from fastapi import APIRouter
from app.api.endpoints import templates, documents

api_router = APIRouter()
api_router.include_router(templates.router)
api_router.include_router(documents.router)
