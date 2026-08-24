import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from app.config import settings
from app.models.schemas import BoundingBox, TemplateField


class CropEngine:
    """Slices field Regions of Interest (ROIs) from aligned document images based on bounding boxes."""

    def __init__(self, output_dir: Path = settings.CROPS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def crop_field(
        self,
        image_np: np.ndarray,
        bbox: BoundingBox,
        job_id: str,
        field_id: str,
        save_to_disk: bool = True,
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Crops a rectangular region from image_np.
        Returns:
            crop_np: BGR numpy image array of cropped field ROI.
            crop_path: Filepath where cropped image is saved.
        """
        img_h, img_w = image_np.shape[:2]

        # Clamp bounding box coordinates within image boundaries
        x1 = max(0, min(bbox.x, img_w - 1))
        y1 = max(0, min(bbox.y, img_h - 1))
        x2 = max(x1 + 1, min(bbox.x + bbox.width, img_w))
        y2 = max(y1 + 1, min(bbox.y + bbox.height, img_h))

        crop_np = image_np[y1:y2, x1:x2]

        crop_path = None
        if save_to_disk and crop_np.size > 0:
            filename = f"{job_id}_{field_id}.png"
            full_path = self.output_dir / filename
            cv2.imwrite(str(full_path), crop_np)
            crop_path = str(full_path)

        return crop_np, crop_path

    @staticmethod
    def save_artifact(
        image_np: np.ndarray,
        output_dir: Path,
        job_id: str,
        field_id: str,
        artifact: str,
        line_number: Optional[int] = None,
    ) -> Optional[str]:
        """Persist a derived crop while retaining the original field crop."""
        if image_np is None or image_np.size == 0:
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        line_suffix = f"_line_{line_number:02d}" if line_number is not None else ""
        path = output_dir / f"{job_id}_{field_id}_{artifact}{line_suffix}.png"
        if not cv2.imwrite(str(path), image_np):
            raise OSError(f"Could not save {artifact} crop for field {field_id}")
        return str(path)
