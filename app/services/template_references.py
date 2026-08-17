from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import settings


class TemplateReferenceStore:
    """Stores canonical reference images used to align completed documents."""

    def __init__(self, storage_dir: Path = settings.TEMPLATE_REFERENCES_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, template_id: str, page_number: int = 1) -> Path:
        # Template IDs are schema-validated safe identifiers before reaching this store.
        if page_number < 1:
            raise ValueError("page_number must be at least 1")
        # Keep the original page-one filename so previously approved single-page
        # templates continue to work, while every later page has its own reference.
        suffix = "" if page_number == 1 else f".page-{page_number:03d}"
        return self.storage_dir / f"{template_id}{suffix}.png"

    def exists(self, template_id: str, page_number: int = 1) -> bool:
        return self.path_for(template_id, page_number).is_file()

    def save(self, template_id: str, image_np: np.ndarray, page_number: int = 1) -> Path:
        path = self.path_for(template_id, page_number)
        if not cv2.imwrite(str(path), image_np):
            raise OSError(f"Could not save reference image for template '{template_id}'.")
        return path

    def load(self, template_id: str, page_number: int = 1) -> Optional[np.ndarray]:
        path = self.path_for(template_id, page_number)
        if not path.is_file():
            return None
        return cv2.imread(str(path), cv2.IMREAD_COLOR)


template_reference_store = TemplateReferenceStore()
