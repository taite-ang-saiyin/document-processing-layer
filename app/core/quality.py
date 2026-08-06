import cv2
import numpy as np
from app.config import settings
from app.models.schemas import QualityCheckResult


class QualityChecker:
    """Evaluates input document image quality (blur, contrast, brightness)."""

    def __init__(self, blur_threshold: float = settings.BLUR_LAPLACIAN_THRESHOLD):
        self.blur_threshold = blur_threshold

    def evaluate_image(self, image_np: np.ndarray) -> QualityCheckResult:
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np

        # Laplacian variance for sharpness/blur detection
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = blur_score < self.blur_threshold

        # Illumination assessment (mean brightness)
        mean_brightness = float(np.mean(gray))
        illumination_ok = 30.0 <= mean_brightness <= 230.0

        is_passed = (not is_blurry) and illumination_ok
        
        messages = []
        if is_blurry:
            messages.append(f"Image is too blurry (Laplacian variance {blur_score:.2f} < threshold {self.blur_threshold})")
        if not illumination_ok:
            messages.append(f"Suboptimal illumination (Mean brightness: {mean_brightness:.2f})")
            
        message = "; ".join(messages) if messages else "Image quality passed all checks."

        return QualityCheckResult(
            is_passed=is_passed,
            blur_score=round(blur_score, 2),
            is_blurry=is_blurry,
            illumination_ok=illumination_ok,
            message=message,
        )
