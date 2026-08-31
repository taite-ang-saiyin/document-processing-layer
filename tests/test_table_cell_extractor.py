import cv2
import numpy as np

from app.services.table_cell_extractor import TemplateCellExtractor


def blank_cell() -> np.ndarray:
    image = np.full((80, 180, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (179, 79), (0, 0, 0), 2)
    return image


def test_unchanged_template_cell_is_empty():
    reference = blank_cell()
    result = TemplateCellExtractor().isolate(reference.copy(), reference)

    assert result.is_empty is True
    assert result.change_ratio == 0.0
    assert np.all(result.image == 255)


def test_added_cell_text_isolated_from_static_border():
    reference = blank_cell()
    filled = reference.copy()
    cv2.putText(
        filled,
        "1250",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    result = TemplateCellExtractor().isolate(filled, reference)

    assert result.is_empty is False
    assert result.change_ratio > 0.0015
    assert np.count_nonzero(result.difference) > 0
    assert np.all(result.image[0:2, :] == 255)



def test_populated_reference_uses_current_cell_instead_of_erasing_value():
    reference = blank_cell()
    current = blank_cell()
    cv2.putText(reference, "OLD", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(current, "NEW", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    result = TemplateCellExtractor().isolate(current, reference)

    assert result.is_empty is False
    assert result.reference_has_content is True
    assert result.used_raw_fallback is True
    assert result.is_header is False
    assert np.array_equal(result.image, current)


def test_populated_reference_does_not_make_a_blank_current_cell_nonempty():
    reference = blank_cell()
    cv2.putText(reference, "OLD", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    result = TemplateCellExtractor().isolate(blank_cell(), reference)

    assert result.is_empty is True
    assert result.reference_has_content is True
    assert result.used_raw_fallback is True
    assert np.all(result.image == 255)


def test_dark_table_header_is_inverted_for_ocr():
    header = np.full((80, 180, 3), (160, 70, 35), dtype=np.uint8)
    cv2.putText(header, "DATE", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    result = TemplateCellExtractor().isolate(header.copy(), header)

    assert result.is_header is True
    assert result.is_empty is False
    assert result.used_raw_fallback is True
    assert np.mean(result.image) > 180
