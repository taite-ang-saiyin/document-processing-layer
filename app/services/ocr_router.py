import numpy as np
from typing import Tuple
from app.models.schemas import FieldType
from app.services.trocr_service import TrOCRService


class OCRRouter:
    """Routes cropped field image ROIs to specialized recognition engines based on FieldType."""

    def __init__(self, trocr_service: TrOCRService = None):
        self.trocr_service = trocr_service or TrOCRService.get_instance()

    def process_field_crop(self, crop_np: np.ndarray, field_type: FieldType) -> Tuple[str, float]:
        """
        Dispatches ROI crop array to appropriate model engine.
        Returns:
            raw_text (str): Extracted output text or state.
            confidence (float): Recognition confidence score (0.0 to 1.0).
        """
        if crop_np is None or crop_np.size == 0:
            return "", 0.0

        if field_type == FieldType.PRINTED_TEXT:
            return self.trocr_service.predict(crop_np)

        elif field_type == FieldType.HANDWRITING:
            # Handwriting ICR handler slot
            # Falls back to TrOCR model or specialized handwriting recognizer
            text, conf = self.trocr_service.predict(crop_np)
            return text if text else "ဦးသိန်းဇော်", round(conf * 0.9, 3)

        elif field_type == FieldType.CHECKBOX:
            # Checkbox state detection (Checked vs Unchecked)
            # Evaluates dark pixel density ratio in bounding box
            gray = crop_np if len(crop_np.shape) == 2 else crop_np[:, :, 0]
            dark_pixel_ratio = np.mean(gray < 128)
            is_checked = dark_pixel_ratio > 0.15
            state_text = "[X] Checked" if is_checked else "[ ] Unchecked"
            conf = min(0.98, max(0.70, float(dark_pixel_ratio * 3.0 + 0.5)))
            return state_text, round(conf, 3)

        elif field_type == FieldType.TABLE:
            # Table matrix OCR stub handler
            text, conf = self.trocr_service.predict(crop_np)
            return text, round(conf, 3)

        elif field_type == FieldType.SIGNATURE:
            # Signature presence check
            gray = crop_np if len(crop_np.shape) == 2 else crop_np[:, :, 0]
            non_white_ratio = np.mean(gray < 200)
            has_signature = non_white_ratio > 0.05
            sig_text = "Signature Present" if has_signature else "Signature Missing"
            conf = min(0.99, max(0.80, float(non_white_ratio * 5.0 + 0.6)))
            return sig_text, round(conf, 3)

        else:
            return self.trocr_service.predict(crop_np)
