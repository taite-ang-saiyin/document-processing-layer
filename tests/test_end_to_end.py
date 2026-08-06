import io
import cv2
import numpy as np
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
