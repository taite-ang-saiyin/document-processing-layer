import datetime
import uuid
from typing import Dict, Optional
import cv2
import numpy as np

from app.config import settings
from app.core.quality import QualityChecker
from app.core.alignment import ImageAligner
from app.services.crop_engine import CropEngine
from app.services.ocr_router import OCRRouter
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
        self.ocr_router = OCRRouter()
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
            template_reference_np = self.template_reference_store.load(template_id, 1)
        template_references = {1: template_reference_np}
        for page_number in range(2, expected_pages + 1):
            template_references[page_number] = self.template_reference_store.load(template_id, page_number)
        missing_reference_pages = [
            str(page_number)
            for page_number, reference in template_references.items()
            if reference is None
        ]
        if missing_reference_pages:
            job = DocumentProcessingJob(
                job_id=job_id,
                template_id=template_id,
                status=ProcessingStatus.FAILED,
                created_at=created_at,
                template_selection_mode=template_selection_mode,
                template_match_score=template_match_score,
                template_match_runner_up_score=template_match_runner_up_score,
                error=(
                    f"Template '{template_id}' has no registered reference image for "
                    f"page(s): {', '.join(missing_reference_pages)}."
                ),
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

        # 2. Align every uploaded page with the corresponding approved template page.
        aligned_pages: dict[int, np.ndarray] = {}
        page_alignment_scores: dict[int, float] = {}
        for page_number in range(1, expected_pages + 1):
            aligned_page, _, score = self.image_aligner.align_images(
                page_images[page_number - 1], template_references[page_number]
            )
            aligned_pages[page_number] = aligned_page
            page_alignment_scores[page_number] = float(score)
        aligned_page_dir = settings.ALIGNED_PAGES_DIR / job_id
        aligned_page_dir.mkdir(parents=True, exist_ok=True)
        aligned_page_paths: dict[int, str] = {}
        for page_number, aligned_page in aligned_pages.items():
            path = aligned_page_dir / f"page_{page_number:03d}.png"
            if not cv2.imwrite(str(path), aligned_page):
                raise OSError(f"Could not save aligned document page {page_number}")
            aligned_page_paths[page_number] = str(path)
        lowest_alignment_score = min(page_alignment_scores.values())
        failed_alignment_pages = {
            page_number
            for page_number, score in page_alignment_scores.items()
            if score < settings.ALIGNMENT_SCORE_THRESHOLD
        }
        # A completely unaligned document cannot be extracted safely.  For a
        # multi-page document, however, do not throw away the data from pages
        # that *did* align just because another page needs a human to correct
        # its alignment.
        if len(failed_alignment_pages) == len(page_alignment_scores):
            failed_pages = [str(page_number) for page_number in sorted(failed_alignment_pages)]
            job = DocumentProcessingJob(
                job_id=job_id,
                template_id=template_id,
                status=ProcessingStatus.FAILED,
                created_at=created_at,
                alignment_score=lowest_alignment_score,
                page_alignment_scores=page_alignment_scores,
                page_count=len(page_images),
                aligned_page_paths=aligned_page_paths,
                template_selection_mode=template_selection_mode,
                template_match_score=template_match_score,
                template_match_runner_up_score=template_match_runner_up_score,
                error=(
                    f"Page(s) {', '.join(failed_pages)} alignment score is below the required "
                    f"threshold {settings.ALIGNMENT_SCORE_THRESHOLD:.3f}."
                ),
            )
            self.persistence.save_job(job, JOBS_STORE)
            return job

        # 3. Field cropping and raw OCR.  The review workspace deliberately
        # receives the OCR engine's unmodified output: no crop enhancement,
        # text normalization, or validation rules are applied here.
        extracted_fields = []
        needs_human_review = not quality_res.is_passed or bool(failed_alignment_pages)

        for field_def in template.fields:
            if field_def.page in failed_alignment_pages:
                alignment_score = page_alignment_scores[field_def.page]
                extracted_fields.append(
                    ExtractedFieldResult(
                        field_id=field_def.id,
                        label=field_def.label,
                        field_type=field_def.field_type,
                        page=field_def.page,
                        raw_text="",
                        normalized_text="",
                        ocr_confidence=0.0,
                        validation_passed=False,
                        validation_message=(
                            f"Page {field_def.page} needs alignment review "
                            f"({alignment_score:.3f} is below the required "
                            f"{settings.ALIGNMENT_SCORE_THRESHOLD:.3f})."
                        ),
                        final_confidence=0.0,
                        human_review_flag=True,
                    )
                )
                continue

            # Crop field ROI
            crop_np, crop_path = self.crop_engine.crop_field(
                aligned_pages[field_def.page], field_def.bbox, job_id, field_def.id
            )

            # Send the original crop directly to OCR.  Do not normalize or
            # validate its returned text; use its native confidence verbatim.
            raw_text, ocr_conf = self.ocr_router.process_field_crop(crop_np, field_def.field_type)
            field_res = ExtractedFieldResult(
                field_id=field_def.id,
                label=field_def.label,
                field_type=field_def.field_type,
                page=field_def.page,
                raw_text=raw_text,
                normalized_text=raw_text,
                ocr_confidence=ocr_conf,
                validation_passed=True,
                validation_message=None,
                final_confidence=ocr_conf,
                human_review_flag=ocr_conf < settings.CONFIDENCE_THRESHOLD,
                crop_image_path=crop_path,
            )

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
            alignment_score=round(float(np.mean(list(page_alignment_scores.values()))), 3),
            page_alignment_scores=page_alignment_scores,
            page_count=len(page_images),
            aligned_page_paths=aligned_page_paths,
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
