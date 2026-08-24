import numpy as np

from app.services.line_detector import DetectedLine, TextLineDetector


class FakeTextDetection:
    def predict(self, _image, batch_size=1):
        assert batch_size == 1
        return [{
            "res": {
                "dt_polys": np.array([
                    [[10, 10], [40, 10], [40, 28], [10, 28]],
                    [[48, 11], [92, 11], [92, 29], [48, 29]],
                    [[12, 48], [85, 48], [85, 67], [12, 67]],
                ]),
                "dt_scores": np.array([0.91, 0.87, 0.95]),
            }
        }]


def test_text_detector_groups_word_regions_into_ordered_lines():
    detector = TextLineDetector()
    detector._model = FakeTextDetection()
    image = np.full((100, 120, 3), 255, dtype=np.uint8)

    lines = detector.detect(image)

    assert lines == [
        DetectedLine(x=10, y=10, width=82, height=19, score=0.89),
        DetectedLine(x=12, y=48, width=73, height=19, score=0.95),
    ]
