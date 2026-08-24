import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.api.endpoints.documents import pipeline

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "HEALTHY"


def test_list_templates():
    response = client.get("/api/v1/templates")
    assert response.status_code == 200
    templates = response.json()
    assert len(templates) >= 1
    assert any(template["template_id"] == "claim_form_v1" for template in templates)


def test_document_processing_pipeline_e2e():
    # 1. Generate synthetic claim form image in memory
    img = np.full((1600, 1200, 3), 255, dtype=np.uint8)
    cv2.putText(img, "BURMESE INSURANCE CLAIM FORM", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "NAME: U AYE MAUNG", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "POLICY: POL-998877", (550, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    success, img_encoded = cv2.imencode(".png", img)
    assert success
    png_bytes = img_encoded.tobytes()

    # 2. Register the approved template's canonical alignment reference.
    reference_response = client.post(
        "/api/v1/templates/claim_form_v1/reference",
        files={"file": ("claim_form_reference.png", png_bytes, "image/png")},
    )
    assert reference_response.status_code == 201, reference_response.text

    # 3. Post document to /api/v1/documents/process
    response = client.post(
        "/api/v1/documents/process",
        files={"file": ("test_claim_form.png", png_bytes, "image/png")},
        data={"template_id": "claim_form_v1"},
    )

    assert response.status_code == 200, f"Error detail: {response.text}"
    job_data = response.json()
    assert "job_id" in job_data
    assert job_data["template_id"] == "claim_form_v1"
    assert len(job_data["extracted_fields"]) == 6
    assert job_data["status"] in ["COMPLETED", "HUMAN_REVIEW_REQUIRED"]
    assert job_data["alignment_score"] is not None
    assert job_data["alignment_score"] >= 0.5
    assert job_data["template_selection_mode"] == "explicit"

    job_id = job_data["job_id"]

    # 4. Test job status retrieval
    job_response = client.get(f"/api/v1/documents/jobs/{job_id}")
    assert job_response.status_code == 200
    assert job_response.json()["job_id"] == job_id

    # 5. Test export endpoints
    for fmt in ["json", "csv", "excel"]:
        export_res = client.get(f"/api/v1/documents/jobs/{job_id}/export/{fmt}")
        assert export_res.status_code == 200


def test_multi_page_document_aligns_each_page_and_uses_page_specific_fields(monkeypatch):
    template = {
        "template_id": "two_page_claim_v1",
        "name": "Two-page claim",
        "width": 600,
        "height": 800,
        "pages": [
            {"page_number": 1, "width": 600, "height": 800},
            {"page_number": 2, "width": 600, "height": 800},
        ],
        "fields": [
            {
                "id": "field_page_one",
                "label": "Page one field",
                "field_type": "printed_text",
                "page": 1,
                "bbox": {"x": 50, "y": 100, "width": 200, "height": 60},
            },
            {
                "id": "field_page_two",
                "label": "Page two field",
                "field_type": "printed_text",
                "page": 2,
                "bbox": {"x": 50, "y": 100, "width": 200, "height": 60},
            },
        ],
    }
    registered = client.post("/api/v1/templates/register", json=template)
    assert registered.status_code == 201, registered.text
    reference = np.full((800, 600, 3), 255, dtype=np.uint8)
    success, encoded_reference = cv2.imencode(".png", reference)
    assert success
    reference_response = client.post(
        "/api/v1/templates/two_page_claim_v1/reference",
        files={"file": ("reference.png", encoded_reference.tobytes(), "image/png")},
    )
    assert reference_response.status_code == 201, reference_response.text
    second_reference_response = client.post(
        "/api/v1/templates/two_page_claim_v1/reference?page_number=2",
        files={"file": ("reference-page-2.png", encoded_reference.tobytes(), "image/png")},
    )
    assert second_reference_response.status_code == 201, second_reference_response.text
    monkeypatch.setattr(
        "app.api.endpoints.documents.pipeline.ocr_router.process_field_crop",
        lambda *_args, **_kwargs: ("value", 0.95),
    )
    monkeypatch.setattr(
        "app.api.endpoints.documents.pipeline.image_aligner.align_images",
        lambda image, _reference: (cv2.resize(image, (600, 800)), None, 1.0),
    )

    page_one = np.full((400, 300, 3), 255, dtype=np.uint8)
    page_two = np.full((400, 300, 3), 255, dtype=np.uint8)
    result = pipeline.process_document(
        image_np=[page_one, page_two],
        template_id="two_page_claim_v1",
    )

    fields = [field.model_dump() for field in result.extracted_fields]
    assert [field["field_id"] for field in fields] == [
        "field_page_one", "field_page_two"
    ]
    assert [field["page"] for field in fields] == [1, 2]
    assert result.page_count == 2
    assert result.page_alignment_scores == {1: 1.0, 2: 1.0}
    aligned_page_response = client.get(f"/api/v1/documents/jobs/{result.job_id}/pages/2")
    assert aligned_page_response.status_code == 200
    assert aligned_page_response.headers["content-type"] == "image/png"


def test_multi_page_document_keeps_aligned_page_data_when_another_page_needs_review(monkeypatch):
    template = {
        "template_id": "partial_alignment_claim_v1",
        "name": "Partial alignment claim",
        "width": 600,
        "height": 800,
        "pages": [
            {"page_number": 1, "width": 600, "height": 800},
            {"page_number": 2, "width": 600, "height": 800},
        ],
        "fields": [
            {
                "id": "page_one_value",
                "label": "Page one value",
                "field_type": "printed_text",
                "page": 1,
                "bbox": {"x": 50, "y": 100, "width": 200, "height": 60},
            },
            {
                "id": "page_two_value",
                "label": "Page two value",
                "field_type": "printed_text",
                "page": 2,
                "bbox": {"x": 50, "y": 100, "width": 200, "height": 60},
            },
        ],
    }
    assert client.post("/api/v1/templates/register", json=template).status_code == 201
    reference = np.full((800, 600, 3), 255, dtype=np.uint8)
    success, encoded_reference = cv2.imencode(".png", reference)
    assert success
    for page_number in (1, 2):
        response = client.post(
            f"/api/v1/templates/partial_alignment_claim_v1/reference?page_number={page_number}",
            files={"file": ("reference.png", encoded_reference.tobytes(), "image/png")},
        )
        assert response.status_code == 201, response.text

    alignment_scores = iter((1.0, 0.4))
    monkeypatch.setattr(
        "app.api.endpoints.documents.pipeline.image_aligner.align_images",
        lambda image, _reference: (cv2.resize(image, (600, 800)), None, next(alignment_scores)),
    )
    ocr_crop_shapes = []

    def raw_ocr(crop, *_args, **_kwargs):
        ocr_crop_shapes.append(crop.shape[:2])
        return "extracted page one", 0.95

    monkeypatch.setattr(
        "app.api.endpoints.documents.pipeline.ocr_router.process_field_crop", raw_ocr
    )

    page = np.full((400, 300, 3), 255, dtype=np.uint8)
    result = pipeline.process_document(
        image_np=[page, page], template_id="partial_alignment_claim_v1"
    )

    fields = {field.field_id: field for field in result.extracted_fields}
    assert result.status.value == "HUMAN_REVIEW_REQUIRED"
    assert result.needs_human_review is True
    assert fields["page_one_value"].raw_text == "extracted page one"
    assert fields["page_one_value"].human_review_flag is False
    assert fields["page_two_value"].raw_text == ""
    assert fields["page_two_value"].human_review_flag is True
    assert "alignment review" in fields["page_two_value"].validation_message
    # OCR receives the cleaned/padded crop when line detection falls back.
    assert ocr_crop_shapes == [(80, 220)]
