"""Text-line detection for OCR field crops.

PaddleOCR detects text polygons.  This adapter turns those polygons into a
small, ordered set of line crops and intentionally returns no lines when the
detection is unreliable so callers can use their complete field-crop fallback.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import settings


@dataclass(frozen=True)
class DetectedLine:
    x: int
    y: int
    width: int
    height: int
    score: float


class TextLineDetector:
    """Lazy PaddleOCR detector that groups text polygons into visual lines."""

    def __init__(self, min_score: float = settings.LINE_DETECTION_MIN_SCORE):
        self.min_score = min_score
        self._model: Any | None = None
        self._unavailable = False
        self._lock = threading.Lock()

    def _load(self) -> Any | None:
        if self._unavailable:
            return None
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None or self._unavailable:
                return self._model
            try:
                # PaddleOCR's standalone text detector is script agnostic and
                # returns polygons plus confidence scores.  Loading remains lazy
                # because direct/local deployments may opt out of this model.
                from paddleocr import TextDetection

                # Paddle 3.3's oneDNN mode cannot execute the PP-OCRv6
                # detector graph on this CPU image.  Use Paddle's compatible
                # plain CPU runtime explicitly instead of silently falling
                # back for every crop.
                self._model = TextDetection(
                    engine="paddle_static",
                    engine_config={"device_type": "cpu", "run_mode": "paddle"},
                )
            except Exception:
                self._unavailable = True
        return self._model

    @staticmethod
    def _plain(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return TextLineDetector._plain(value.tolist())
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, dict):
            return {str(key): TextLineDetector._plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [TextLineDetector._plain(item) for item in value]
        return value

    @classmethod
    def _prediction_payload(cls, prediction: Any) -> dict[str, Any]:
        if hasattr(prediction, "json"):
            payload = prediction.json
            payload = payload() if callable(payload) else payload
            if isinstance(payload, str):
                payload = json.loads(payload)
        elif isinstance(prediction, dict):
            payload = prediction
        else:
            payload = dict(prediction)
        payload = cls._plain(payload)
        return payload.get("res", payload)

    @staticmethod
    def _bbox(polygon: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 2:
            return None
        x1 = max(0, min(width - 1, int(np.floor(points[:, 0].min()))))
        y1 = max(0, min(height - 1, int(np.floor(points[:, 1].min()))))
        x2 = max(x1 + 1, min(width, int(np.ceil(points[:, 0].max()))))
        y2 = max(y1 + 1, min(height, int(np.ceil(points[:, 1].max()))))
        return x1, y1, x2, y2

    def detect(self, image_np: np.ndarray) -> list[DetectedLine]:
        if image_np is None or image_np.size == 0:
            return []
        model = self._load()
        if model is None:
            return []
        try:
            predictions = list(model.predict(image_np, batch_size=1))
            if len(predictions) != 1:
                return []
            payload = self._prediction_payload(predictions[0])
            polygons = payload.get("dt_polys")
            scores = payload.get("dt_scores")
            if polygons is None:
                polygons = []
            if scores is None:
                scores = []
            height, width = image_np.shape[:2]
            regions: list[tuple[int, int, int, int, float]] = []
            for index, polygon in enumerate(polygons):
                score = float(scores[index]) if index < len(scores) else 1.0
                bbox = self._bbox(polygon, width, height)
                if bbox is None or score < self.min_score:
                    continue
                x1, y1, x2, y2 = bbox
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                regions.append((x1, y1, x2, y2, score))
            return self._group_regions_into_lines(regions, width, height)
        except Exception:
            # Recognition should not fail just because optional line detection
            # is temporarily unavailable or a model rejects a crop.
            return []

    @staticmethod
    def _group_regions_into_lines(
        regions: list[tuple[int, int, int, int, float]], width: int, height: int
    ) -> list[DetectedLine]:
        if not regions:
            return []
        regions.sort(key=lambda item: ((item[1] + item[3]) / 2, item[0]))
        heights = [bottom - top for _, top, _, bottom, _ in regions]
        median_height = max(4.0, float(np.median(heights)))
        rows: list[list[tuple[int, int, int, int, float]]] = []
        for region in regions:
            center_y = (region[1] + region[3]) / 2
            for row in rows:
                row_center = np.mean([(item[1] + item[3]) / 2 for item in row])
                if abs(center_y - row_center) <= max(median_height * 0.75, (region[3] - region[1]) * 0.75):
                    row.append(region)
                    break
            else:
                rows.append([region])

        lines: list[DetectedLine] = []
        for row in rows:
            x1 = min(item[0] for item in row)
            y1 = min(item[1] for item in row)
            x2 = max(item[2] for item in row)
            y2 = max(item[3] for item in row)
            score = round(float(np.mean([item[4] for item in row])), 3)
            # Reject narrow/noisy fragments that cannot represent an OCR line.
            if (x2 - x1) < 8 or (y2 - y1) < 6:
                continue
            lines.append(DetectedLine(x1, y1, x2 - x1, y2 - y1, score))
        lines.sort(key=lambda line: (line.y, line.x))
        return lines
