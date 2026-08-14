from typing import List
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import TemplateDefinition
from app.services.pipeline_orchestrator import TEMPLATES_REGISTRY
from app.services.persistence import PersistenceService

router = APIRouter(prefix="/templates", tags=["Templates"])
persistence = PersistenceService()


@router.post("/register", response_model=TemplateDefinition, status_code=status.HTTP_201_CREATED)
def register_template(template: TemplateDefinition):
    """Registers a new approved template definition into the Template Registry."""
    persistence.save_template(template, TEMPLATES_REGISTRY)
    return template


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
