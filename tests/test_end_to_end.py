import io
import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

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
    assert templates[0]["template_id"] == "claim_form_v1"


def test_document_processing_pipeline_e2e():
    # 1. Generate synthetic claim form image in memory
    img = np.full((1600, 1200, 3), 255, dtype=np.uint8)
    cv2.putText(img, "BURMESE INSURANCE CLAIM FORM", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "NAME: U AYE MAUNG", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "POLICY: POL-998877", (550, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    success, img_encoded = cv2.imencode(".png", img)
    assert success
    png_bytes = img_encoded.tobytes()

    # 2. Post document to /api/v1/documents/process
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

    job_id = job_data["job_id"]

    # 3. Test job status retrieval
    job_response = client.get(f"/api/v1/documents/jobs/{job_id}")
    assert job_response.status_code == 200
    assert job_response.json()["job_id"] == job_id

    # 4. Test export endpoints
    for fmt in ["json", "csv", "excel"]:
        export_res = client.get(f"/api/v1/documents/jobs/{job_id}/export/{fmt}")
        assert export_res.status_code == 200


def test_multi_page_pdf_uses_page_specific_template_fields(monkeypatch):
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
    monkeypatch.setattr(
        "app.api.endpoints.documents.pipeline.ocr_router.process_field_crop",
        lambda *_args, **_kwargs: ("value", 0.95),
    )

    page_one = Image.new("RGB", (300, 400), "white")
    page_two = Image.new("RGB", (300, 400), "white")
    output = io.BytesIO()
    page_one.save(output, format="PDF", save_all=True, append_images=[page_two])
    pdf = output.getvalue()

    response = client.post(
        "/api/v1/documents/process",
        files={"file": ("two-page.pdf", pdf, "application/pdf")},
        data={"template_id": "two_page_claim_v1"},
    )

    assert response.status_code == 200, response.text
    fields = response.json()["extracted_fields"]
    assert [field["field_id"] for field in fields] == [
        "field_page_one", "field_page_two"
    ]
    assert [field["page"] for field in fields] == [1, 2]
