from enum import Enum
from typing import Any, Dict, List, Literal, Optional
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
    page: int = Field(default=1, ge=1, description="One-based template page number")
    required: bool = True
    validation_regex: Optional[str] = None
    choice_group_id: Optional[str] = Field(default=None, pattern=SAFE_IDENTIFIER_PATTERN)
    choice_option_value: Optional[str] = Field(default=None, max_length=256)
    choice_mode: Optional[Literal["single_choice", "multiple_choice"]] = None


class TemplatePage(BaseModel):
    page_number: int = Field(..., ge=1)
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class TemplateDefinition(BaseModel):
    template_id: str = Field(..., pattern=SAFE_IDENTIFIER_PATTERN, description="Unique template identifier e.g. claim_form_v1")
    name: str = Field(..., min_length=1, description="Form template title")
    width: int = Field(1200, gt=0, description="Baseline reference image width")
    height: int = Field(1600, gt=0, description="Baseline reference image height")
    pages: Optional[List[TemplatePage]] = None
    fields: List[TemplateField] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_page_geometry(self) -> "TemplateDefinition":
        field_ids = [field.id for field in self.fields]
        duplicate_ids = sorted({field_id for field_id in field_ids if field_ids.count(field_id) > 1})
        if duplicate_ids:
            raise ValueError(f"Template field IDs must be unique; duplicates: {', '.join(duplicate_ids)}")

        dimensions = {1: (self.width, self.height)}
        if self.pages:
            page_numbers = [page.page_number for page in self.pages]
            if page_numbers != list(range(1, len(page_numbers) + 1)):
                raise ValueError("template pages must be sequential starting at page 1")
            dimensions = {
                page.page_number: (page.width, page.height) for page in self.pages
            }
        for field in self.fields:
            if field.page not in dimensions:
                raise ValueError(f"field {field.id} references unknown page {field.page}")
            width, height = dimensions[field.page]
            if (
                field.bbox.x + field.bbox.width > width
                or field.bbox.y + field.bbox.height > height
            ):
                raise ValueError(
                    f"field {field.id} bbox lies outside page {field.page}"
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
    page: int = Field(default=1, ge=1)
    raw_text: str
    normalized_text: str
    ocr_confidence: float
    validation_passed: bool
    validation_message: Optional[str] = None
    final_confidence: float
    human_review_flag: bool
    crop_image_path: Optional[str] = None
    preprocessed_crop_path: Optional[str] = None
    line_crop_paths: List[str] = Field(default_factory=list)
    ocr_mode: str = "full_field_fallback"
    llm_post_correction_applied: bool = False
    llm_post_correction_reason: Optional[str] = None
    choice_group_id: Optional[str] = None
    choice_option_value: Optional[str] = None
    choice_mode: Optional[Literal["single_choice", "multiple_choice"]] = None
    choice_selected: Optional[bool] = None
    choice_group_status: Optional[str] = None


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
    page_alignment_scores: Dict[int, float] = Field(default_factory=dict)
    page_count: int = Field(default=1, ge=1)
    aligned_page_paths: Dict[int, str] = Field(default_factory=dict)
    alignment_method: str = "template_homography"
    template_selection_mode: str = "explicit"
    template_match_score: Optional[float] = None
    template_match_runner_up_score: Optional[float] = None
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
