import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.endpoints.documents import pipeline
from app.services.line_detector import DetectedLine


client = TestClient(app)


def _template_payload(template_id: str) -> dict:
    return {
        "template_id": template_id,
        "name": "Alignment test template",
        "width": 480,
        "height": 320,
        "fields": [
            {
                "id": "claim_number",
                "label": "Claim number",
                "field_type": "printed_text",
                "bbox": {"x": 40, "y": 180, "width": 240, "height": 60},
                "required": True,
            }
        ],
    }


def _reference_image() -> np.ndarray:
    image = np.full((320, 480, 3), 215, dtype=np.uint8)
    cv2.rectangle(image, (15, 15), (465, 305), (0, 0, 0), 3)
    cv2.putText(image, "INSURANCE CLAIM", (55, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(image, "POLICY 998877", (55, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "CLAIM 12345", (55, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return image


def _png_bytes(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update(template_id="invalid template id"),
        lambda payload: payload.update(width=0),
        lambda payload: payload["fields"][0]["bbox"].update(width=0),
        lambda payload: payload["fields"][0]["bbox"].update(x=-1),
        lambda payload: payload["fields"][0]["bbox"].update(x=300, width=240),
        lambda payload: payload["fields"].append({**payload["fields"][0]}),
    ],
)
def test_template_registration_rejects_invalid_geometry_and_duplicate_ids(mutator):
    payload = _template_payload("invalid_template")
    mutator(payload)

    response = client.post("/api/v1/templates/register", json=payload)

    assert response.status_code == 422


def test_document_processing_requires_a_reference_image():
    template_id = "requires_reference"
    response = client.post("/api/v1/templates/register", json=_template_payload(template_id))
    assert response.status_code == 201

    process_response = client.post(
        "/api/v1/documents/process",
        files={"file": ("completed.png", _png_bytes(_reference_image()), "image/png")},
        data={"template_id": template_id},
    )

    assert process_response.status_code == 409
    assert "reference image" in process_response.json()["detail"]


def test_template_reference_upload_rejects_wrong_dimensions_and_bad_content_type():
    template_id = "reference_validation"
    response = client.post("/api/v1/templates/register", json=_template_payload(template_id))
    assert response.status_code == 201

    wrong_size = np.full((100, 100, 3), 255, dtype=np.uint8)
    dimensions_response = client.post(
        f"/api/v1/templates/{template_id}/reference",
        files={"file": ("wrong-size.png", _png_bytes(wrong_size), "image/png")},
    )
    assert dimensions_response.status_code == 422

    mime_response = client.post(
        f"/api/v1/templates/{template_id}/reference",
        files={"file": ("not-an-image.txt", b"not an image", "text/plain")},
    )
    assert mime_response.status_code == 415

    corrupt_image_response = client.post(
        f"/api/v1/templates/{template_id}/reference",
        files={"file": ("corrupt.png", b"not a PNG", "image/png")},
    )
    assert corrupt_image_response.status_code == 400


def test_processing_uses_reference_and_rejects_low_alignment():
    template_id = "alignment_enforced"
    reference = _reference_image()
    register_response = client.post("/api/v1/templates/register", json=_template_payload(template_id))
    assert register_response.status_code == 201

    reference_response = client.post(
        f"/api/v1/templates/{template_id}/reference",
        files={"file": ("reference.png", _png_bytes(reference), "image/png")},
    )
    assert reference_response.status_code == 201
    assert reference_response.json()["stored"] is True
    assert reference_response.json()["reference_filename"] == f"{template_id}.png"

    corrupt_document_response = client.post(
        "/api/v1/documents/process",
        files={"file": ("corrupt.png", b"not a PNG", "image/png")},
        data={"template_id": template_id},
    )
    assert corrupt_document_response.status_code == 400

    aligned_response = client.post(
        "/api/v1/documents/process",
        files={"file": ("completed.png", _png_bytes(reference), "image/png")},
        data={"template_id": template_id},
    )
    assert aligned_response.status_code == 200, aligned_response.text
    aligned_job = aligned_response.json()
    assert aligned_job["status"] in ["COMPLETED", "HUMAN_REVIEW_REQUIRED"]
    assert aligned_job["alignment_score"] >= 0.5
    assert len(aligned_job["extracted_fields"]) == 1

    unaligned_image = np.full(reference.shape, 215, dtype=np.uint8)
    unaligned_response = client.post(
        "/api/v1/documents/process",
        files={"file": ("unaligned.png", _png_bytes(unaligned_image), "image/png")},
        data={"template_id": template_id},
    )
    assert unaligned_response.status_code == 200
    unaligned_job = unaligned_response.json()
    assert unaligned_job["status"] == "FAILED"
    assert unaligned_job["alignment_score"] == 0.0
    assert "alignment score" in unaligned_job["error"]


def test_canonicalized_pages_skip_feature_based_template_warp(monkeypatch):
    template_id = "canonical_page_processing"
    reference = _reference_image()
    assert client.post("/api/v1/templates/register", json=_template_payload(template_id)).status_code == 201
    assert client.post(
        f"/api/v1/templates/{template_id}/reference",
        files={"file": ("reference.png", _png_bytes(reference), "image/png")},
    ).status_code == 201

    def should_not_align(*_args, **_kwargs):
        raise AssertionError("canonical pages must not be template-warped")

    monkeypatch.setattr(pipeline.image_aligner, "align_images", should_not_align)
    response = client.post(
        "/api/v1/documents/process",
        files={"file": ("canonical.png", _png_bytes(reference), "image/png")},
        data={"template_id": template_id, "canonicalized_pages": "true"},
    )

    assert response.status_code == 200, response.text
    job = response.json()
    assert job["alignment_method"] == "canonical_page_resize"
    assert job["alignment_score"] == 1.0


def test_text_field_uses_detected_lines_and_preserves_all_crop_artifacts(monkeypatch):
    template_id = "line_aware_processing"
    reference = _reference_image()
    assert client.post("/api/v1/templates/register", json=_template_payload(template_id)).status_code == 201
    assert client.post(
        f"/api/v1/templates/{template_id}/reference",
        files={"file": ("reference.png", _png_bytes(reference), "image/png")},
    ).status_code == 201
    monkeypatch.setattr(
        pipeline.line_detector,
        "detect",
        lambda _crop: [DetectedLine(5, 5, 80, 20, 0.9), DetectedLine(5, 30, 80, 20, 0.9)],
    )
    monkeypatch.setattr(pipeline.ocr_router, "process_field_crop", lambda _crop, _kind: ("line", 0.9))

    response = client.post(
        "/api/v1/documents/process",
        files={"file": ("canonical.png", _png_bytes(reference), "image/png")},
        data={"template_id": template_id, "canonicalized_pages": "true"},
    )

    assert response.status_code == 200, response.text
    field = response.json()["extracted_fields"][0]
    assert field["raw_text"] == "line\nline"
    assert field["ocr_mode"] == "detected_lines"
    assert field["crop_image_path"]
    assert field["preprocessed_crop_path"]
    assert len(field["line_crop_paths"]) == 2


def test_automatic_template_matching_selects_a_clear_reference_match():
    template_id = "automatic_match"
    reference = _reference_image()
    assert client.post("/api/v1/templates/register", json=_template_payload(template_id)).status_code == 201
    assert client.post(
        f"/api/v1/templates/{template_id}/reference",
        files={"file": ("reference.png", _png_bytes(reference), "image/png")},
    ).status_code == 201

    response = client.post(
        "/api/v1/documents/process",
        files={"file": ("completed.png", _png_bytes(reference), "image/png")},
    )

    assert response.status_code == 200, response.text
    job = response.json()
    assert job["template_id"] == template_id
    assert job["template_selection_mode"] == "automatic"
    assert job["template_match_score"] >= 0.5
    assert job["alignment_score"] >= 0.5


def test_automatic_template_matching_rejects_ambiguous_references():
    reference = _reference_image()
    for template_id in ("ambiguous_match_a", "ambiguous_match_b"):
        assert client.post("/api/v1/templates/register", json=_template_payload(template_id)).status_code == 201
        assert client.post(
            f"/api/v1/templates/{template_id}/reference",
            files={"file": ("reference.png", _png_bytes(reference), "image/png")},
        ).status_code == 201

    response = client.post(
        "/api/v1/documents/process",
        files={"file": ("completed.png", _png_bytes(reference), "image/png")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "TEMPLATE_MATCH_UNCERTAIN"
    assert len(detail["candidates"]) == 2
    assert detail["candidates"][0]["score"] == detail["candidates"][1]["score"]


def test_automatic_template_matching_rejects_unmatched_document():
    template_id = "unmatched_document"
    assert client.post("/api/v1/templates/register", json=_template_payload(template_id)).status_code == 201
    assert client.post(
        f"/api/v1/templates/{template_id}/reference",
        files={"file": ("reference.png", _png_bytes(_reference_image()), "image/png")},
    ).status_code == 201
    unrelated = np.full((320, 480, 3), 215, dtype=np.uint8)

    response = client.post(
        "/api/v1/documents/process",
        files={"file": ("unrelated.png", _png_bytes(unrelated), "image/png")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "TEMPLATE_MATCH_UNCERTAIN"
    assert detail["candidates"][0]["score"] == 0.0
