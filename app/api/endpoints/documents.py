import cv2
import numpy as np
import subprocess
import tempfile
from pathlib import Path
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


def _load_document_images(contents: bytes, content_type: str, filename: str) -> np.ndarray | list[np.ndarray]:
    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        try:
            with tempfile.TemporaryDirectory(prefix="document-pages-") as temp_dir:
                temp_path = Path(temp_dir)
                source_path = temp_path / "source.pdf"
                source_path.write_bytes(contents)
                output_prefix = temp_path / "page"
                subprocess.run(
                    ["pdftoppm", "-png", "-r", "144", str(source_path), str(output_prefix)],
                    check=True,
                    capture_output=True,
                )
                rendered_paths = sorted(
                    temp_path.glob("page-*.png"), key=lambda path: int(path.stem.rsplit("-", 1)[1])
                )
                page_images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in rendered_paths]
                if not page_images or any(page is None for page in page_images):
                    raise ValueError("PDF page rendering produced an unreadable image")
                return page_images
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid or unreadable PDF file.") from exc
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file format.")
    return image


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
    """Processes an image or multi-page PDF with an explicit or automatically matched template."""
    contents = await file.read()
    img_np = _load_document_images(contents, (file.content_type or "").lower(), file.filename or "document")

    template_selection_mode = "explicit"
    template_match_score = None
    template_match_runner_up_score = None
    if template_id:
        if template_id not in TEMPLATES_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Template '{template_id}' is not registered. Please register template first.",
            )
        template = TEMPLATES_REGISTRY[template_id]
        expected_pages = max((field.page for field in template.fields), default=1)
        missing_reference_pages = [
            page_number
            for page_number in range(1, expected_pages + 1)
            if not template_reference_store.exists(template_id, page_number)
        ]
        if missing_reference_pages:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Template '{template_id}' requires registered reference images for "
                    f"page(s): {', '.join(map(str, missing_reference_pages))}."
                ),
            )
    else:
        match_image = img_np[0] if isinstance(img_np, list) else img_np
        match_result = template_matcher.match(match_image, TEMPLATES_REGISTRY)
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


@router.post("/match-template")
async def match_template(
    file: UploadFile = File(..., description="Document to match against approved template references"),
):
    """Matches the first document page and returns candidates without processing OCR."""
    contents = await file.read()
    image = _load_document_images(contents, (file.content_type or "").lower(), file.filename or "document")
    match_image = image[0] if isinstance(image, list) else image
    result = template_matcher.match(match_image, TEMPLATES_REGISTRY)
    return {
        "selected_template_id": result.selected_template_id,
        "score": result.top_score,
        "runner_up_score": result.runner_up_score,
        "reason": result.reason,
        "candidates": [
            {"template_id": candidate.template_id, "score": candidate.score}
            for candidate in result.candidates
        ],
    }


@router.get("/jobs/{job_id}", response_model=DocumentProcessingJob)
def get_job_status(job_id: str):
    """Fetches job processing status and extracted fields by job ID."""
    if job_id not in JOBS_STORE:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JOBS_STORE[job_id]


@router.get("/jobs/{job_id}/pages/{page_number}")
def get_aligned_page(job_id: str, page_number: int):
    """Returns the aligned page used for field cropping and OCR review."""
    job = JOBS_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    page_path = job.aligned_page_paths.get(page_number)
    if not page_path or not Path(page_path).is_file():
        raise HTTPException(status_code=404, detail=f"Aligned page {page_number} is unavailable.")
    return FileResponse(page_path, media_type="image/png")


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
