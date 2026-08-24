from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkDetection:
    checked: bool
    confidence: float
    uncertain: bool
    changed_pixel_ratio: float
    mean_ink_delta: float


class TemplateMarkDetector:
    """Detect ink added to a checkbox/radio control relative to its blank template."""

    MIN_CHANGED_RATIO = 0.035
    MIN_MEAN_INK_DELTA = 3.0
    PIXEL_DELTA_THRESHOLD = 28

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _interior(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        # Exclude the printed square/circle outline but keep the tick/dot area.
        border = max(2, min(height, width) // 7)
        if height <= border * 2 + 2 or width <= border * 2 + 2:
            return image
        return image[border:-border, border:-border]

    def detect(self, crop: np.ndarray, reference_crop: np.ndarray | None) -> MarkDetection:
        if crop is None or crop.size == 0:
            return MarkDetection(False, 0.0, True, 0.0, 0.0)

        current = self._gray(crop)
        if reference_crop is None or reference_crop.size == 0:
            # Direct callers without a template reference get a deliberately
            # conservative result that must be reviewed.
            ratio = float(np.mean(self._interior(current) < 100))
            return MarkDetection(ratio >= 0.08, 0.6, True, ratio, 0.0)

        reference = self._gray(reference_crop)
        if reference.shape != current.shape:
            reference = cv2.resize(reference, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_AREA)
        current = self._interior(current).astype(np.int16)
        reference = self._interior(reference).astype(np.int16)

        # Only newly-added dark ink counts; printed control borders are shared
        # by the completed form and blank template and thus cancel out.
        ink_delta = np.maximum(reference - current, 0)
        changed_ratio = float(np.mean(ink_delta >= self.PIXEL_DELTA_THRESHOLD))
        mean_ink_delta = float(np.mean(ink_delta))
        checked = changed_ratio >= self.MIN_CHANGED_RATIO and mean_ink_delta >= self.MIN_MEAN_INK_DELTA
        near_ratio = self.MIN_CHANGED_RATIO * 0.5 < changed_ratio < self.MIN_CHANGED_RATIO * 1.5
        near_delta = self.MIN_MEAN_INK_DELTA * 0.5 < mean_ink_delta < self.MIN_MEAN_INK_DELTA * 1.5
        uncertain = near_ratio or near_delta
        strength = max(changed_ratio / self.MIN_CHANGED_RATIO, mean_ink_delta / self.MIN_MEAN_INK_DELTA)
        confidence = min(0.99, 0.55 + 0.18 * min(abs(strength - 1.0), 2.0))
        if uncertain:
            confidence = min(confidence, 0.74)
        return MarkDetection(checked, round(confidence, 3), uncertain, round(changed_ratio, 4), round(mean_ink_delta, 3))
