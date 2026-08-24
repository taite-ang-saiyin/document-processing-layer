from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from app.config import settings
from app.core.normalizer import BurmeseTextNormalizer
from app.models.schemas import FieldType, TemplateField


@dataclass(frozen=True)
class PostCorrectionResult:
    text: str
    applied: bool = False
    reason: str | None = None


class LlmPostCorrector:
    """Correct every eligible OCR field on a page in one context-aware VLM request."""

    def __init__(self) -> None:
        self.normalizer = BurmeseTextNormalizer()

    def normalized_raw(self, raw_text: str) -> PostCorrectionResult:
        return PostCorrectionResult(self.normalizer.normalize(raw_text))

    @staticmethod
    def _status(page_path: str | None, status: str, **details: object) -> None:
        page_name = Path(page_path).name if page_path else "unknown"
        suffix = " ".join(f"{key}={value}" for key, value in details.items())
        print(f"[post-correction] page={page_name} status={status}" + (f" {suffix}" if suffix else ""), flush=True)

    def correct_page(
        self,
        page_path: str | None,
        fields: Iterable[tuple[TemplateField, str, float]],
    ) -> dict[str, PostCorrectionResult]:
        candidates = [
            (field, raw_text, confidence)
            for field, raw_text, confidence in fields
            if raw_text.strip()
            and field.field_type not in {FieldType.CHECKBOX, FieldType.SIGNATURE}
        ]
        results = {
            field.id: self.normalized_raw(raw_text)
            for field, raw_text, _ in candidates
        }
        if not settings.LLM_POST_CORRECTION_ENABLED or not candidates:
            self._status(
                page_path,
                "skipped",
                reason=("disabled" if not settings.LLM_POST_CORRECTION_ENABLED else "no_eligible_fields"),
                fields=len(candidates),
            )
            return results
        if not page_path or not Path(page_path).is_file():
            self._status(page_path, "skipped", reason="page_unavailable", fields=len(candidates))
            return {
                field.id: PostCorrectionResult(self.normalizer.normalize(raw_text), reason="page_unavailable")
                for field, raw_text, _ in candidates
            }

        payload = [
            {
                "field_id": field.id,
                "label": field.label,
                "field_type": field.field_type.value,
                "bbox_px": {
                    "x": field.bbox.x,
                    "y": field.bbox.y,
                    "width": field.bbox.width,
                    "height": field.bbox.height,
                },
                "raw_text": raw_text,
                "ocr_confidence": round(float(confidence), 4),
                "validation_regex": field.validation_regex,
            }
            for field, raw_text, confidence in candidates
        ]
        self._status(page_path, "requesting", fields=len(payload))
        try:
            with Path(page_path).open("rb") as image:
                response = requests.post(
                    settings.LLM_POST_CORRECTION_URL.rstrip("/") + "/api/v1/ocr-post-correction/page",
                    headers={"X-API-Key": settings.LLM_POST_CORRECTION_API_KEY},
                    data={"fields_json": json.dumps(payload, ensure_ascii=False)},
                    files={"image": (Path(page_path).name, image, "image/png")},
                    timeout=settings.LLM_POST_CORRECTION_TIMEOUT_SECONDS,
                )
            response.raise_for_status()
            page_result = response.json()
        except (OSError, requests.RequestException, ValueError) as exc:
            details: dict[str, object] = {"reason": "vlm_unavailable", "fields": len(candidates)}
            response = getattr(exc, "response", None)
            if response is not None:
                details["reason"] = "vlm_request_failed"
                details["http_status"] = response.status_code
                try:
                    detail = response.json().get("detail")
                    if isinstance(detail, dict) and isinstance(detail.get("code"), str):
                        details["error_code"] = detail["code"]
                except (ValueError, AttributeError):
                    pass
            self._status(page_path, "skipped", **details)
            return {
                field.id: PostCorrectionResult(self.normalizer.normalize(raw_text), reason="vlm_unavailable")
                for field, raw_text, _ in candidates
            }

        if not page_result.get("available"):
            reason = str(page_result.get("reason") or "engine_unavailable")
            self._status(page_path, "skipped", reason=reason, fields=len(candidates))
            return {
                field.id: PostCorrectionResult(self.normalizer.normalize(raw_text), reason=reason)
                for field, raw_text, _ in candidates
            }
        suggestions = {
            item.get("field_id"): item
            for item in page_result.get("corrections", [])
            if isinstance(item, dict) and isinstance(item.get("field_id"), str)
        }
        results = {
            field.id: self._evaluate(field, raw_text, suggestions.get(field.id))
            for field, raw_text, _ in candidates
        }
        applied = sum(result.applied for result in results.values())
        self._status(
            page_path,
            "completed",
            fields=len(candidates),
            suggestions=len(suggestions),
            applied=applied,
        )
        return results

    def _evaluate(
        self,
        _field: TemplateField,
        raw_text: str,
        suggestion: dict | None,
    ) -> PostCorrectionResult:
        original = self.normalizer.normalize(raw_text)
        if not suggestion:
            return PostCorrectionResult(original, reason="no_suggestion")
        candidate = suggestion.get("suggested_text")
        if not isinstance(candidate, str):
            return PostCorrectionResult(original, reason="invalid_suggestion")
        candidate = self.normalizer.normalize(candidate)
        if not candidate or candidate == original:
            return PostCorrectionResult(original, reason="empty_or_unchanged")
        return PostCorrectionResult(candidate, applied=True, reason=str(suggestion.get("reason") or "gemini"))
