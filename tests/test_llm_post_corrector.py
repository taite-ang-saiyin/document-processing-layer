import json

from app.config import settings
from app.models.schemas import BoundingBox, FieldType, TemplateField
from app.services.llm_post_corrector import LlmPostCorrector


def _field(**overrides):
    values = {
        "id": "policy_number",
        "label": "Policy number",
        "field_type": FieldType.PRINTED_TEXT,
        "bbox": BoundingBox(x=0, y=0, width=10, height=10),
        "validation_regex": r"^POL-\d{3}$",
    }
    values.update(overrides)
    return TemplateField(**values)


def test_applies_page_level_suggestions_with_bbox_and_ocr_confidence(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    monkeypatch.setattr(settings, "LLM_POST_CORRECTION_ENABLED", True)

    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "available": True,
                "corrections": [{
                    "field_id": "policy_number",
                    "changed": True,
                    "suggested_text": "POL-123",
                    "confidence": 0.99,
                    "reason": "VISUAL_CONFIRMATION",
                }],
            }

    def post(*args, **kwargs):
        captured["url"] = args[0]
        captured["fields"] = kwargs["data"]["fields_json"]
        return Response()

    monkeypatch.setattr("app.services.llm_post_corrector.requests.post", post)
    result = LlmPostCorrector().correct_page(str(image), [(_field(), "POL-I23", 0.72)])

    assert result["policy_number"].text == "POL-123"
    assert result["policy_number"].applied is True
    assert captured["url"].endswith("/api/v1/ocr-post-correction/page")
    request_field = json.loads(captured["fields"])[0]
    assert request_field["bbox_px"] == {"x": 0, "y": 0, "width": 10, "height": 10}
    assert request_field["ocr_confidence"] == 0.72


def test_applies_a_page_level_suggestion_regardless_of_confidence_or_field_validation(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    monkeypatch.setattr(settings, "LLM_POST_CORRECTION_ENABLED", True)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "available": True,
                "corrections": [{
                    "field_id": "policy_number",
                    "changed": False,
                    "suggested_text": "invented value",
                    "confidence": 0.01,
                }],
            }

    monkeypatch.setattr("app.services.llm_post_corrector.requests.post", lambda *args, **kwargs: Response())
    result = LlmPostCorrector().correct_page(str(image), [(_field(), "POL-I23", 0.9)])

    assert result["policy_number"].text == "invented value"
    assert result["policy_number"].applied is True
