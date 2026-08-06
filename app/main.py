from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.router import api_router
from app.models.schemas import TemplateDefinition, TemplateField, FieldType, BoundingBox
from app.services.pipeline_orchestrator import TEMPLATES_REGISTRY


def seed_default_templates():
    """Seed initial default Burmese Insurance Claim Form template into registry."""
    default_template = TemplateDefinition(
        template_id="claim_form_v1",
        name="Burmese Motor Insurance Claim Form",
        width=1200,
        height=1600,
        fields=[
            TemplateField(
                id="field_claimant_name",
                label="အာမခံထားသူအမည် ( Claimant Name )",
                field_type=FieldType.PRINTED_TEXT,
                bbox=BoundingBox(x=100, y=200, width=400, height=50),
                required=True,
            ),
            TemplateField(
                id="field_policy_number",
                label="ပေါ်လစီအမှတ် ( Policy Number )",
                field_type=FieldType.PRINTED_TEXT,
                bbox=BoundingBox(x=550, y=200, width=350, height=50),
                required=True,
            ),
            TemplateField(
                id="field_nrc_number",
                label="မှတ်ပုံတင်အမှတ် ( NRC Number )",
                field_type=FieldType.PRINTED_TEXT,
                bbox=BoundingBox(x=100, y=280, width=450, height=50),
                required=True,
                validation_regex=r"^[\u1000-\u109F0-9\/\(\)\s\-]+$",
            ),
            TemplateField(
                id="field_handwritten_note",
                label="ဖြစ်စဉ်အကျဉ်း ( Accident Details Note )",
                field_type=FieldType.HANDWRITING,
                bbox=BoundingBox(x=100, y=360, width=800, height=120),
                required=False,
            ),
            TemplateField(
                id="field_checkbox_third_party",
                label="တတိယလူပါဝင်မှု ( Third Party Claim Checkbox )",
                field_type=FieldType.CHECKBOX,
                bbox=BoundingBox(x=100, y=500, width=40, height=40),
                required=False,
            ),
            TemplateField(
                id="field_signature",
                label="လျှောက်ထားသူလက်မှတ် ( Signature )",
                field_type=FieldType.SIGNATURE,
                bbox=BoundingBox(x=600, y=600, width=300, height=100),
                required=True,
            ),
        ],
    )
    TEMPLATES_REGISTRY[default_template.template_id] = default_template


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_default_templates()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Seed immediately on module import as well for tests
seed_default_templates()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR,
    }


@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "model_dir": str(settings.MODEL_DIR)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
