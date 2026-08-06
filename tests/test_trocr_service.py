import numpy as np
from app.services.trocr_service import TrOCRService


def test_trocr_service_prediction():
    service = TrOCRService.get_instance()
    
    # Create test synthetic crop ROI
    test_crop = np.full((50, 200, 3), 255, dtype=np.uint8)
    
    text, confidence = service.predict(test_crop)
    assert isinstance(text, str)
    assert 0.0 <= confidence <= 1.0
