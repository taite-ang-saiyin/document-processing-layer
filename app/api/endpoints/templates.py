from typing import List
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from app.models.schemas import TemplateDefinition
from app.services.pipeline_orchestrator import TEMPLATES_REGISTRY
from app.services.persistence import PersistenceService
from app.services.template_references import template_reference_store

router = APIRouter(prefix="/templates", tags=["Templates"])
persistence = PersistenceService()


@router.post("/register", response_model=TemplateDefinition, status_code=status.HTTP_201_CREATED)
def register_template(template: TemplateDefinition):
    """Registers a new approved template definition into the Template Registry."""
    persistence.save_template(template, TEMPLATES_REGISTRY)
    return template


@router.post("/{template_id}/reference", status_code=status.HTTP_201_CREATED)
async def upload_template_reference(
    template_id: str,
    file: UploadFile = File(..., description="Canonical blank template image used for alignment"),
):
    """Stores the approved template's alignment reference image as a canonical PNG."""
    template = TEMPLATES_REGISTRY.get(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' is not registered.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Template reference must be an image upload.")

    contents = await file.read()
    image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image_np is None:
        raise HTTPException(status_code=400, detail="Invalid or unreadable template reference image.")
    height, width = image_np.shape[:2]
    if width != template.width or height != template.height:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Template reference dimensions must be {template.width}x{template.height}; "
                f"received {width}x{height}."
            ),
        )

    try:
        reference_path = template_reference_store.save(template_id, image_np)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not store template reference image.") from exc

    return {
        "template_id": template_id,
        "width": width,
        "height": height,
        "stored": True,
        "reference_filename": reference_path.name,
    }


@router.get("/{template_id}", response_model=TemplateDefinition)
def get_template(template_id: str):
    """Retrieves a template definition by ID."""
    if template_id not in TEMPLATES_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found.")
    return TEMPLATES_REGISTRY[template_id]


@router.get("", response_model=List[TemplateDefinition])
def list_templates():
    """Lists all registered templates in the system."""
    return list(TEMPLATES_REGISTRY.values())
