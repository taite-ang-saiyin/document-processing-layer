import datetime
import uuid
from typing import Dict, Optional
import cv2
import numpy as np

from app.config import settings
from app.core.quality import QualityChecker
from app.core.alignment import ImageAligner
from app.core.crop_preprocessor import CropPreprocessor
from app.services.crop_engine import CropEngine
from app.services.ocr_router import OCRRouter
from app.services.validation_engine import ValidationEngine
from app.services.exporter import StructuredExporter
from app.services.persistence import PersistenceService
from app.services.template_references import template_reference_store
from app.models.schemas import (
    DocumentProcessingJob,
    ProcessingStatus,
    TemplateDefinition,
    ExtractedFieldResult,
    QualityCheckResult,
)


# In-memory Template & Job Registry Store
TEMPLATES_REGISTRY: Dict[str, TemplateDefinition] = {}
JOBS_STORE: Dict[str, DocumentProcessingJob] = {}


class DocumentProcessingPipeline:
    """Orchestrates the entire document processing pipeline from ingestion to structured output."""

    def __init__(self):
        self.quality_checker = QualityChecker()
        self.image_aligner = ImageAligner()
        self.crop_engine = CropEngine()
        self.crop_preprocessor = CropPreprocessor()
        self.ocr_router = OCRRouter()
        self.validation_engine = ValidationEngine()
        self.exporter = StructuredExporter()
        self.persistence = PersistenceService()
        self.template_reference_store = template_reference_store

    def process_document(
        self,
        image_np: np.ndarray | list[np.ndarray],
        template_id: str,
        job_id: Optional[str] = None,
        template_reference_np: Optional[np.ndarray] = None,
        template_selection_mode: str = "explicit",
        template_match_score: Optional[float] = None,
        template_match_runner_up_score: Optional[float] = None,
    ) -> DocumentProcessingJob:
        if not job_id:
            job_id = str(uuid.uuid4())[:8]

        created_at = datetime.datetime.now().isoformat()

        # Check template existence
        template = TEMPLATES_REGISTRY.get(template_id)
        if not template:
            job = DocumentProcessingJob(
                job_id=job_id,
                template_id=template_id,
                status=ProcessingStatus.FAILED,
                created_at=created_at,
                template_selection_mode=template_selection_mode,
                template_match_score=template_match_score,
                template_match_runner_up_score=template_match_runner_up_score,
                error=f"Template '{template_id}' not found in registry.",
            )
            self.persistence.save_job(job, JOBS_STORE)
            return job

        page_images = image_np if isinstance(image_np, list) else [image_np]
        expected_pages = max((field.page for field in template.fields), default=1)
        if len(page_images) < expected_pages:
            job = DocumentProcessingJob(
                job_id=job_id,
                template_id=template_id,
                status=ProcessingStatus.FAILED,
                created_at=created_at,
                template_selection_mode=template_selection_mode,
                template_match_score=template_match_score,
                template_match_runner_up_score=template_match_runner_up_score,
                error=(
                    f"Template requires {expected_pages} pages, but the uploaded "
                    f"document contains {len(page_images)}."
                ),
            )
            self.persistence.save_job(job, JOBS_STORE)
            return job

        if template_reference_np is None:
            template_reference_np = self.template_reference_store.load(template_id)
        if template_reference_np is None:
            job = DocumentProcessingJob(
                job_id=job_id,
                template_id=template_id,
                status=ProcessingStatus.FAILED,
                created_at=created_at,
                template_selection_mode=template_selection_mode,
                template_match_score=template_match_score,
                template_match_runner_up_score=template_match_runner_up_score,
                error=f"Template '{template_id}' has no registered reference image.",
            )
            self.persistence.save_job(job, JOBS_STORE)
            return job

        # 1. Quality Check across every required page.
        page_quality = [
            self.quality_checker.evaluate_image(page_images[index])
            for index in range(expected_pages)
        ]
        quality_res = QualityCheckResult(
            is_passed=all(item.is_passed for item in page_quality),
            blur_score=round(float(np.mean([item.blur_score for item in page_quality])), 2),
            is_blurry=any(item.is_blurry for item in page_quality),
            illumination_ok=all(item.illumination_ok for item in page_quality),
            message="; ".join(
                f"Page {index}: {item.message}"
                for index, item in enumerate(page_quality, 1)
                if not item.is_passed
            ) or "All document pages passed quality checks.",
        )

        # 2. Align page 1 to its approved reference.
        aligned_page_one, _, align_score = self.image_aligner.align_images(
            page_images[0], template_reference_np
        )
        if align_score < settings.ALIGNMENT_SCORE_THRESHOLD:
            job = DocumentProcessingJob(
                job_id=job_id,
                template_id=template_id,
                status=ProcessingStatus.FAILED,
                created_at=created_at,
                alignment_score=align_score,
                template_selection_mode=template_selection_mode,
                template_match_score=template_match_score,
                template_match_runner_up_score=template_match_runner_up_score,
                error=(
                    f"Image alignment score {align_score:.3f} is below the required "
                    f"threshold {settings.ALIGNMENT_SCORE_THRESHOLD:.3f}."
                ),
            )
            self.persistence.save_job(job, JOBS_STORE)
            return job

        # Subsequent pages currently use their registered pixel geometry.
        page_dimensions = {
            page.page_number: (page.width, page.height)
            for page in (template.pages or [])
        }
        page_dimensions.setdefault(1, (template.width, template.height))
        aligned_pages: dict[int, np.ndarray] = {1: aligned_page_one}
        for page_number in range(2, expected_pages + 1):
            target_width, target_height = page_dimensions.get(
                page_number, (template.width, template.height)
            )
            aligned_pages[page_number] = cv2.resize(
                page_images[page_number - 1], (target_width, target_height)
            )

        # 3. Field Cropping, Preprocessing, OCR Routing, and Validation
        extracted_fields = []
        needs_human_review = not quality_res.is_passed

        for field_def in template.fields:
            # Crop field ROI
            crop_np, crop_path = self.crop_engine.crop_field(
                aligned_pages[field_def.page], field_def.bbox, job_id, field_def.id
            )

            # Preprocess cropped field ROI (denoise, contrast enhancement, border cleanup, aspect-ratio padding)
            preprocessed_crop_np = self.crop_preprocessor.process(crop_np)

            # Route preprocessed crop to OCR Engine
            raw_text, ocr_conf = self.ocr_router.process_field_crop(preprocessed_crop_np, field_def.field_type)

            # Validate & Score
            field_res = self.validation_engine.process_and_validate(
                field_def, raw_text, ocr_conf, crop_path
            )
            field_res.page = field_def.page

            extracted_fields.append(field_res)

            if field_res.human_review_flag:
                needs_human_review = True

        # Calculate overall confidence
        if extracted_fields:
            overall_conf = float(np.mean([f.final_confidence for f in extracted_fields]))
        else:
            overall_conf = 0.0

        status = (
            ProcessingStatus.HUMAN_REVIEW_REQUIRED
            if needs_human_review
            else ProcessingStatus.COMPLETED
        )

        job = DocumentProcessingJob(
            job_id=job_id,
            template_id=template_id,
            status=status,
            quality_check=quality_res,
            extracted_fields=extracted_fields,
            overall_confidence=round(overall_conf, 3),
            needs_human_review=needs_human_review,
            alignment_score=align_score,
            template_selection_mode=template_selection_mode,
            template_match_score=template_match_score,
            template_match_runner_up_score=template_match_runner_up_score,
            created_at=created_at,
            completed_at=datetime.datetime.now().isoformat(),
        )

        # Auto-export structured data
        self.exporter.export_all(job)

        # Save to jobs store (and database when available)
        self.persistence.save_job(job, JOBS_STORE)
        return job
