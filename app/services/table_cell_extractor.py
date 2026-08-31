from dataclasses import dataclass

import cv2
import numpy as np

from app.config import settings


@dataclass(frozen=True)
class TableCellExtraction:
    image: np.ndarray
    difference: np.ndarray
    is_empty: bool
    change_ratio: float
    reference_has_content: bool = False
    used_raw_fallback: bool = False
    is_header: bool = False


class TemplateCellExtractor:
    """Isolate ink added to a fixed-layout table cell relative to its blank template."""

    def isolate(self, current: np.ndarray, reference: np.ndarray) -> TableCellExtraction:
        if current.size == 0:
            return TableCellExtraction(current, current, True, 0.0)
        if reference.shape[:2] != current.shape[:2]:
            reference = cv2.resize(
                reference,
                (current.shape[1], current.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        current_gray_raw = self._gray(current)
        reference_gray_raw = self._gray(reference)
        current_gray = cv2.GaussianBlur(current_gray_raw, (3, 3), 0)
        reference_gray = cv2.GaussianBlur(reference_gray_raw, (3, 3), 0)

        # Directional subtraction retains newly added dark ink while removing
        # unchanged borders, labels, and pre-printed table content.
        dark_delta = cv2.subtract(reference_gray, current_gray)
        mask = np.where(
            dark_delta >= settings.TABLE_CELL_DELTA_THRESHOLD, 255, 0
        ).astype(np.uint8)

        height, width = mask.shape[:2]
        margin = max(
            2,
            int(round(min(height, width) * settings.TABLE_CELL_BORDER_MARGIN_RATIO)),
        )
        if height > margin * 2 and width > margin * 2:
            mask[:margin, :] = 0
            mask[-margin:, :] = 0
            mask[:, :margin] = 0
            mask[:, -margin:] = 0

        analysis_margin_y = max(2, int(round(height * 0.12)))
        analysis_margin_x = max(2, int(round(width * 0.08)))
        if height > analysis_margin_y * 2 and width > analysis_margin_x * 2:
            current_interior = current_gray_raw[
                analysis_margin_y : height - analysis_margin_y,
                analysis_margin_x : width - analysis_margin_x,
            ]
            reference_interior = reference_gray_raw[
                analysis_margin_y : height - analysis_margin_y,
                analysis_margin_x : width - analysis_margin_x,
            ]
        else:
            current_interior = current_gray_raw
            reference_interior = reference_gray_raw

        current_ink_ratio = float(
            np.mean(current_interior < settings.TABLE_CELL_INK_THRESHOLD)
        )
        reference_ink_ratio = float(
            np.mean(reference_interior < settings.TABLE_CELL_INK_THRESHOLD)
        )
        reference_has_content = (
            reference_ink_ratio >= settings.TABLE_CELL_REFERENCE_CONTENT_RATIO
        )
        is_header = reference_ink_ratio >= settings.TABLE_CELL_HEADER_DARK_RATIO

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cleaned = np.zeros_like(mask)
        minimum_component_area = max(3, int(round(height * width * 0.00015)))
        for component in range(1, component_count):
            if stats[component, cv2.CC_STAT_AREA] >= minimum_component_area:
                cleaned[labels == component] = 255
        if np.any(cleaned):
            cleaned = cv2.dilate(cleaned, np.ones((2, 2), np.uint8), iterations=1)

        interior_area = max(1, (height - 2 * margin) * (width - 2 * margin))
        change_ratio = min(1.0, float(np.count_nonzero(cleaned)) / interior_area)
        is_empty = change_ratio < settings.TABLE_CELL_MIN_CHANGE_RATIO

        # A previously registered template may contain example or completed
        # values. Subtracting that reference erases a matching current value.
        # Use the current cell directly and leave a review warning instead.
        # Dark table headers are inverted to black text on white for TrOCR.
        if is_header:
            _, header = cv2.threshold(
                current_gray_raw,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            return TableCellExtraction(
                image=cv2.cvtColor(header, cv2.COLOR_GRAY2BGR),
                difference=cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR),
                is_empty=False,
                change_ratio=change_ratio,
                reference_has_content=True,
                used_raw_fallback=True,
                is_header=True,
            )
        if reference_has_content:
            current_has_content = (
                current_ink_ratio >= settings.TABLE_CELL_REFERENCE_CONTENT_RATIO
            )
            return TableCellExtraction(
                image=(
                    current.copy()
                    if current_has_content
                    else np.full_like(current, 255)
                ),
                difference=cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR),
                is_empty=not current_has_content,
                change_ratio=change_ratio,
                reference_has_content=True,
                used_raw_fallback=True,
                is_header=False,
            )

        isolated = np.full_like(current, 255)
        if not is_empty:
            if current.ndim == 2:
                isolated[cleaned > 0] = current[cleaned > 0]
            else:
                isolated[cleaned > 0, :] = current[cleaned > 0, :]
        difference = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        return TableCellExtraction(
            image=isolated,
            difference=difference,
            is_empty=is_empty,
            change_ratio=change_ratio,
        )

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
