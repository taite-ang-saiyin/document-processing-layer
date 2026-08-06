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
from app.services.validation_engine import ValidationEngine
from app.services.exporter import StructuredExporter
from app.models.schemas import (
    DocumentProcessingJob,
    ProcessingStatus,
    TemplateDefinition,
    ExtractedFieldResult,
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
        self.validation_engine = ValidationEngine()
        self.exporter = StructuredExporter()

    def process_document(
        self,
        image_np: np.ndarray,
        template_id: str,
        job_id: Optional[str] = None,
        template_reference_np: Optional[np.ndarray] = None,
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
                error=f"Template '{template_id}' not found in registry.",
            )
            JOBS_STORE[job_id] = job
            return job

        # 1. Quality Check
        quality_res = self.quality_checker.evaluate_image(image_np)

        # 2. Image Alignment
        if template_reference_np is not None:
            aligned_img, homography, align_score = self.image_aligner.align_images(image_np, template_reference_np)
        else:
            # Resize image to baseline template dimensions if no reference image
            aligned_img = cv2.resize(image_np, (template.width, template.height))

        # 3. Field Cropping, OCR Routing, and Validation
        extracted_fields = []
        needs_human_review = not quality_res.is_passed

        for field_def in template.fields:
            # Crop field ROI
            crop_np, crop_path = self.crop_engine.crop_field(
                aligned_img, field_def.bbox, job_id, field_def.id
            )

            # Route to OCR Engine
            raw_text, ocr_conf = self.ocr_router.process_field_crop(crop_np, field_def.field_type)

            # Validate & Score
            field_res = self.validation_engine.process_and_validate(
                field_def, raw_text, ocr_conf, crop_path
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
            created_at=created_at,
            completed_at=datetime.datetime.now().isoformat(),
        )

        # Auto-export structured data
        self.exporter.export_all(job)

        # Save to jobs store
        JOBS_STORE[job_id] = job
        return job
