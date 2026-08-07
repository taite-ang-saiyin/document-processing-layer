import cv2
import numpy as np
from app.core.crop_preprocessor import CropPreprocessor


def test_crop_preprocessor_execution():
    preprocessor = CropPreprocessor()

    # Create synthetic raw crop image with border line and noise
    raw_crop = np.full((40, 200, 3), 240, dtype=np.uint8)
    # Add black border line at top and bottom
    raw_crop[:2, :] = 0
    raw_crop[-2:, :] = 0
    # Add text
    cv2.putText(raw_crop, "TEST CROP", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    processed = preprocessor.process(raw_crop)

    assert processed is not None
    assert processed.size > 0
    # Verify padding added around image
    assert processed.shape[0] > raw_crop.shape[0]
    assert processed.shape[1] > raw_crop.shape[1]
    # Verify outer border pixels are cleaned (white background = 255)
    assert np.all(processed[:2, :] == 255)
