import cv2
import numpy as np

from app.models.schemas import ExtractedFieldResult, FieldType
from app.services.mark_detector import TemplateMarkDetector
from app.services.pipeline_orchestrator import DocumentProcessingPipeline


def _control() -> np.ndarray:
    image = np.full((48, 48, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (5, 5), (42, 42), (0, 0, 0), 2)
    return image


def test_template_mark_detector_ignores_blank_control_border():
    reference = _control()
    result = TemplateMarkDetector().detect(reference.copy(), reference)

    assert not result.checked
    assert not result.uncertain


def test_template_mark_detector_detects_added_tick():
    reference = _control()
    completed = reference.copy()
    cv2.line(completed, (14, 25), (22, 34), (0, 0, 0), 3)
    cv2.line(completed, (22, 34), (35, 14), (0, 0, 0), 3)

    result = TemplateMarkDetector().detect(completed, reference)

    assert result.checked
    assert result.confidence >= 0.75


def _choice(option: str, selected: bool) -> ExtractedFieldResult:
    return ExtractedFieldResult(
        field_id=f"claim_type_{option}", label=option, field_type=FieldType.CHECKBOX,
        raw_text="[X] Checked" if selected else "[ ] Unchecked",
        normalized_text="", ocr_confidence=0.95, validation_passed=True,
        final_confidence=0.95, human_review_flag=False,
        choice_group_id="claim_type", choice_option_value=option,
        choice_mode="single_choice", choice_selected=selected,
    )


def test_radio_group_requires_exactly_one_selection():
    fields = [_choice("cash", True), _choice("credit", True)]

    needs_review = DocumentProcessingPipeline._resolve_choice_groups(fields)

    assert needs_review
    assert {field.choice_group_status for field in fields} == {"ambiguous_multiple_selected"}
    assert all(field.human_review_flag for field in fields)
