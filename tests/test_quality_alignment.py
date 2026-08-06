import cv2
import numpy as np
from app.core.quality import QualityChecker
from app.core.alignment import ImageAligner


def test_quality_checker():
    checker = QualityChecker(blur_threshold=100.0)

    # Sharp synthetic image with text grid
    sharp_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(sharp_img, "Burmese OCR Test", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    sharp_res = checker.evaluate_image(sharp_img)
    assert not sharp_res.is_blurry
    assert sharp_res.blur_score > 100.0

    # Blurry image created via Gaussian blur
    blurry_img = cv2.GaussianBlur(sharp_img, (25, 25), 0)
    blurry_res = checker.evaluate_image(blurry_img)
    assert blurry_res.is_blurry


def test_image_aligner():
    aligner = ImageAligner()

    # Create synthetic reference template image
    template = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(template, (50, 50), (350, 350), (255, 255, 255), 3)
    cv2.putText(template, "ANCHOR 1", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(template, "ANCHOR 2", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Test alignment with identical image
    aligned, h_mat, score = aligner.align_images(template, template)
    assert aligned.shape == template.shape
