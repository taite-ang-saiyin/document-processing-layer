from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


SAFE_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"


class FieldType(str, Enum):
    PRINTED_TEXT = "printed_text"
    HANDWRITING = "handwriting"
    CHECKBOX = "checkbox"
    TABLE = "table"
    SIGNATURE = "signature"


class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    FAILED = "FAILED"


class BoundingBox(BaseModel):
    x: int = Field(..., ge=0, description="Top-left X coordinate in pixels")
    y: int = Field(..., ge=0, description="Top-left Y coordinate in pixels")
    width: int = Field(..., gt=0, description="Bounding box width")
    height: int = Field(..., gt=0, description="Bounding box height")


class TemplateField(BaseModel):
    id: str = Field(..., pattern=SAFE_IDENTIFIER_PATTERN, description="Unique field ID e.g. field_001")
    label: str = Field(..., min_length=1, description="Label description e.g. Claimant Name")
    field_type: FieldType = Field(default=FieldType.PRINTED_TEXT)
    bbox: BoundingBox
    required: bool = True
    validation_regex: Optional[str] = None


class TemplateDefinition(BaseModel):
    template_id: str = Field(..., pattern=SAFE_IDENTIFIER_PATTERN, description="Unique template identifier e.g. claim_form_v1")
    name: str = Field(..., min_length=1, description="Form template title")
    width: int = Field(1200, gt=0, description="Baseline reference image width")
    height: int = Field(1600, gt=0, description="Baseline reference image height")
    fields: List[TemplateField] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_field_bounds_and_ids(self) -> "TemplateDefinition":
        field_ids = [field.id for field in self.fields]
        duplicate_ids = sorted({field_id for field_id in field_ids if field_ids.count(field_id) > 1})
        if duplicate_ids:
            raise ValueError(f"Template field IDs must be unique; duplicates: {', '.join(duplicate_ids)}")

        out_of_bounds = [
            field.id
            for field in self.fields
            if field.bbox.x + field.bbox.width > self.width
            or field.bbox.y + field.bbox.height > self.height
        ]
        if out_of_bounds:
            raise ValueError(
                "Field bounding boxes must fit within the template dimensions; "
                f"out of bounds: {', '.join(out_of_bounds)}"
            )
        return self


class QualityCheckResult(BaseModel):
    is_passed: bool
    blur_score: float
    is_blurry: bool
    illumination_ok: bool
    message: str


class ExtractedFieldResult(BaseModel):
    field_id: str
    label: str
    field_type: FieldType
    raw_text: str
    normalized_text: str
    ocr_confidence: float
    validation_passed: bool
    validation_message: Optional[str] = None
    final_confidence: float
    human_review_flag: bool
    crop_image_path: Optional[str] = None


class OcrCropResult(BaseModel):
    field_type: FieldType
    raw_text: str
    normalized_text: str
    confidence: float


class DocumentProcessingJob(BaseModel):
    job_id: str
    template_id: str
    status: ProcessingStatus
    quality_check: Optional[QualityCheckResult] = None
    extracted_fields: List[ExtractedFieldResult] = Field(default_factory=list)
    overall_confidence: float = 0.0
    needs_human_review: bool = False
    alignment_score: Optional[float] = None
    template_selection_mode: str = "explicit"
    template_match_score: Optional[float] = None
    template_match_runner_up_score: Optional[float] = None
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
