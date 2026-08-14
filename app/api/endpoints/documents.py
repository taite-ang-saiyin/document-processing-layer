import cv2
import numpy as np
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from fastapi.responses import FileResponse

from app.models.schemas import DocumentProcessingJob, OcrCropResult, FieldType
from app.services.pipeline_orchestrator import (
    DocumentProcessingPipeline,
    JOBS_STORE,
    TEMPLATES_REGISTRY,
)
from app.services.exporter import StructuredExporter
from app.services.template_references import template_reference_store
from app.services.template_matcher import TemplateMatcher
from app.core.normalizer import BurmeseTextNormalizer

router = APIRouter(prefix="/documents", tags=["Document Processing"])
pipeline = DocumentProcessingPipeline()
exporter = StructuredExporter()
template_matcher = TemplateMatcher()


@router.post("/ocr-crop", response_model=OcrCropResult, status_code=status.HTTP_200_OK)
async def ocr_single_crop(
    file: UploadFile = File(..., description="A single cropped field image to recognize"),
    field_type: FieldType = Form(default=FieldType.PRINTED_TEXT, description="Field type to route to the matching OCR engine"),
):
    """Recognizes text/state from a single pre-cropped field image using the field-type-specific engine."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_np is None:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file format.")

    # Route the crop directly to the appropriate recognition engine (no preprocessing)
    raw_text, confidence = pipeline.ocr_router.process_field_crop(img_np, field_type)

    return OcrCropResult(
        field_type=field_type,
        raw_text=raw_text,
        normalized_text=BurmeseTextNormalizer.normalize(raw_text),
        confidence=confidence,
    )


@router.post("/process", response_model=DocumentProcessingJob, status_code=status.HTTP_200_OK)
async def process_document(
    file: UploadFile = File(..., description="Scanned or photographed completed insurance claim form image"),
    template_id: Optional[str] = Form(
        default=None,
        description="Optional registered template ID. Omit to select a template by reference-image matching.",
    ),
):
    """Processes a document with an explicit template or conservative automatic template matching."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_np is None:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file format.")

    template_selection_mode = "explicit"
    template_match_score = None
    template_match_runner_up_score = None
    if template_id:
        if template_id not in TEMPLATES_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Template '{template_id}' is not registered. Please register template first.",
            )
        if not template_reference_store.exists(template_id):
            raise HTTPException(
                status_code=409,
                detail=f"Template '{template_id}' requires a registered reference image before processing.",
            )
    else:
        match_result = template_matcher.match(img_np, TEMPLATES_REGISTRY)
        if match_result.selected_template_id is None:
            raise HTTPException(status_code=422, detail=match_result.as_error_detail())
        template_id = match_result.selected_template_id
        template_selection_mode = "automatic"
        template_match_score = match_result.top_score
        template_match_runner_up_score = match_result.runner_up_score

    job = pipeline.process_document(
        image_np=img_np,
        template_id=template_id,
        template_selection_mode=template_selection_mode,
        template_match_score=template_match_score,
        template_match_runner_up_score=template_match_runner_up_score,
    )
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
