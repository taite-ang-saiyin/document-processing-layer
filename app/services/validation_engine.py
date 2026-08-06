import re
from typing import Tuple, Optional
from app.config import settings
from app.core.normalizer import BurmeseTextNormalizer
from app.models.schemas import TemplateField, ExtractedFieldResult


class ValidationEngine:
    """Validates extracted field text against business rules, regex formats, required flags, and confidence thresholds."""

    def __init__(self, confidence_threshold: float = settings.CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.normalizer = BurmeseTextNormalizer()

    def process_and_validate(
        self,
        field_def: TemplateField,
        raw_text: str,
        ocr_confidence: float,
        crop_path: Optional[str] = None,
    ) -> ExtractedFieldResult:
        # 1. Normalize text
        normalized_text = self.normalizer.normalize(raw_text)

        # 2. Required field check
        if field_def.required and not normalized_text:
            return ExtractedFieldResult(
                field_id=field_def.id,
                label=field_def.label,
                field_type=field_def.field_type,
                raw_text=raw_text,
                normalized_text=normalized_text,
                ocr_confidence=ocr_confidence,
                validation_passed=False,
                validation_message="Required field is empty.",
                final_confidence=0.0,
                human_review_flag=True,
                crop_image_path=crop_path,
            )

        # 3. Regex pattern validation if specified
        validation_passed = True
        validation_msg = "Passed"

        if field_def.validation_regex and normalized_text:
            if not re.search(field_def.validation_regex, normalized_text):
                validation_passed = False
                validation_msg = f"Format validation failed for pattern: {field_def.validation_regex}"

        # 4. Calculate combined final confidence
        val_score = 1.0 if validation_passed else 0.4
        final_confidence = round(0.7 * ocr_confidence + 0.3 * val_score, 3)

        # 5. Determine Human Review flag
        human_review_flag = (
            (not validation_passed)
            or (final_confidence < self.confidence_threshold)
            or (field_def.required and not normalized_text)
        )

        return ExtractedFieldResult(
            field_id=field_def.id,
            label=field_def.label,
            field_type=field_def.field_type,
            raw_text=raw_text,
            normalized_text=normalized_text,
            ocr_confidence=ocr_confidence,
            validation_passed=validation_passed,
            validation_message=validation_msg,
            final_confidence=final_confidence,
            human_review_flag=human_review_flag,
            crop_image_path=crop_path,
        )
