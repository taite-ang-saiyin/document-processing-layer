import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from fastapi.responses import FileResponse

from app.models.schemas import DocumentProcessingJob
from app.services.pipeline_orchestrator import (
    DocumentProcessingPipeline,
    JOBS_STORE,
    TEMPLATES_REGISTRY,
)
from app.services.exporter import StructuredExporter

router = APIRouter(prefix="/documents", tags=["Document Processing"])
pipeline = DocumentProcessingPipeline()
exporter = StructuredExporter()


@router.post("/process", response_model=DocumentProcessingJob, status_code=status.HTTP_200_OK)
async def process_document(
    file: UploadFile = File(..., description="Scanned or photographed completed insurance claim form image"),
    template_id: str = Form(..., description="Registered template ID to match against e.g. claim_form_v1"),
):
    """Ingests a completed claim form, runs OCR pipeline, and returns extracted structured data."""
    if template_id not in TEMPLATES_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Template '{template_id}' is not registered. Please register template first.",
        )

    # Read uploaded file into OpenCV image matrix
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_np is None:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file format.")

    # Execute pipeline
    job = pipeline.process_document(image_np=img_np, template_id=template_id)
    return job


@router.get("/jobs/{job_id}", response_model=DocumentProcessingJob)
def get_job_status(job_id: str):
    """Fetches job processing status and extracted fields by job ID."""
    if job_id not in JOBS_STORE:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JOBS_STORE[job_id]


@router.get("/jobs/{job_id}/export/{export_format}")
def download_export(job_id: str, export_format: str):
    """
    Downloads structured extraction output in requested format.
    Allowed formats: json, csv, excel
    """
    if job_id not in JOBS_STORE:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    job = JOBS_STORE[job_id]
    paths = exporter.export_all(job)

    fmt = export_format.lower()
    if fmt == "json":
        return FileResponse(paths["json"], media_type="application/json", filename=f"{job_id}.json")
    elif fmt == "csv":
        return FileResponse(paths["csv"], media_type="text/csv", filename=f"{job_id}.csv")
    elif fmt in ["excel", "xlsx"]:
        return FileResponse(
            paths["excel"],
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"{job_id}.xlsx",
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format. Use 'json', 'csv', or 'excel'.")
