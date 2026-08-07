import cv2
import numpy as np


class CropPreprocessor:
    """Preprocesses cropped field ROIs before passing them to OCR engines (TrOCR, Handwriting, etc.)."""

    def __init__(
        self,
        enable_denoise: bool = True,
        enable_contrast_enhancement: bool = True,
        enable_border_removal: bool = True,
        target_height: int = 64,
    ):
        self.enable_denoise = enable_denoise
        self.enable_contrast_enhancement = enable_contrast_enhancement
        self.enable_border_removal = enable_border_removal
        self.target_height = target_height
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process(self, crop_np: np.ndarray) -> np.ndarray:
        """
        Executes crop-level preprocessing pipeline:
        1. Color conversion to grayscale
        2. Denoising & background cleaning
        3. Contrast enhancement (CLAHE)
        4. Form box border / underline removal
        5. Aspect-ratio preserving padding
        """
        if crop_np is None or crop_np.size == 0:
            return crop_np

        processed = crop_np.copy()

        # Convert to BGR if grayscale
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        # 1. Border & Line Artifact Removal
        if self.enable_border_removal:
            processed = self._remove_outer_borders(processed)

        # 2. Contrast Enhancement (CLAHE)
        if self.enable_contrast_enhancement:
            processed = self._enhance_contrast(processed)

        # 3. Denoising
        if self.enable_denoise:
            processed = cv2.fastNlMeansDenoisingColored(
                processed, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
            )

        # 4. Aspect-ratio preserving pad
        processed = self._pad_aspect_ratio(processed)

        return processed

    def _remove_outer_borders(self, img_np: np.ndarray, border_pixels: int = 2) -> np.ndarray:
        """Cleans solid outer bounding box lines near the crop edges."""
        h, w = img_np.shape[:2]
        if h <= border_pixels * 2 or w <= border_pixels * 2:
            return img_np

        cleaned = img_np.copy()
        # Fill outer margin with white background (255)
        cleaned[:border_pixels, :] = 255
        cleaned[-border_pixels:, :] = 255
        cleaned[:, :border_pixels] = 255
        cleaned[:, -border_pixels:] = 255
        return cleaned

    def _enhance_contrast(self, img_np: np.ndarray) -> np.ndarray:
        """Applies CLAHE on L-channel of LAB color space to enhance text ink contrast."""
        lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_enhanced = self.clahe.apply(l_chan)
        lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _pad_aspect_ratio(self, img_np: np.ndarray, pad_margin: int = 10) -> np.ndarray:
        """Pads cropped image with white background to maintain natural aspect ratio before resizing."""
        h, w = img_np.shape[:2]
        if h == 0 or w == 0:
            return img_np

        # Add white padding around image
        padded = cv2.copyMakeBorder(
            img_np,
            pad_margin,
            pad_margin,
            pad_margin,
            pad_margin,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        return padded
