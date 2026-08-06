import os
import math
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image

from app.config import settings


class TrOCRService:
    """Service wrapper for loading and running inference with fine-tuned trocr-small-printed model."""

    _instance: Optional["TrOCRService"] = None

    def __init__(self, model_dir: Path = settings.MODEL_DIR, device: str = settings.DEVICE):
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.processor = None
        self.is_loaded = False
        self._load_model()

    @classmethod
    def get_instance(cls) -> "TrOCRService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Attempts to load HuggingFace TrOCR model and processor from model_dir."""
        # Check if model files exist
        has_config = (self.model_dir / "config.json").exists()
        has_weights = (self.model_dir / "model.safetensors").exists() or (self.model_dir / "pytorch_model.bin").exists()

        if has_config and has_weights:
            try:
                import torch
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel

                print(f"[TrOCRService] Loading model from {self.model_dir} on device={self.device}...")
                self.processor = TrOCRProcessor.from_pretrained(str(self.model_dir))
                self.model = VisionEncoderDecoderModel.from_pretrained(str(self.model_dir))
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                print("[TrOCRService] Pre-trained trocr-small-printed successfully loaded!")
                return
            except Exception as e:
                print(f"[TrOCRService] Warning: Failed to load PyTorch TrOCR model ({e}). Operating in simulation mode.")
        else:
            print(f"[TrOCRService] Model directory {self.model_dir} does not contain complete weights. Operating in fallback mode.")
        
        self.is_loaded = False

    def predict(self, crop_np: np.ndarray) -> Tuple[str, float]:
        """
        Runs OCR on a cropped field ROI numpy array (BGR or RGB).
        Returns:
            extracted_text (str): Recognized Burmese/English text string.
            confidence (float): Calculated output confidence score between 0.0 and 1.0.
        """
        if crop_np is None or crop_np.size == 0:
            return "", 0.0

        # Convert OpenCV BGR array to PIL RGB Image
        if len(crop_np.shape) == 3 and crop_np.shape[2] == 3:
            pil_image = Image.fromarray(crop_np[:, :, ::-1])
        else:
            pil_image = Image.fromarray(crop_np)

        if self.is_loaded and self.model is not None and self.processor is not None:
            try:
                import torch

                pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        pixel_values,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_length=64,
                        num_beams=2,
                    )
                
                # Decode output sequence
                generated_ids = outputs.sequences[0]
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Compute mean token probability from beam search scores
                if hasattr(outputs, "scores") and outputs.scores:
                    log_probs = []
                    for step_scores in outputs.scores:
                        softmax_probs = torch.softmax(step_scores[0], dim=-1)
                        max_prob = torch.max(softmax_probs).item()
                        log_probs.append(max_prob)
                    confidence = float(np.mean(log_probs)) if log_probs else 0.90
                else:
                    confidence = 0.92

                return text.strip(), round(confidence, 3)

            except Exception as e:
                print(f"[TrOCRService] Inference error: {e}")
                return "Inference Error", 0.0

        # Fallback / Stub simulation mode for testing pipeline flow
        # In mock mode, calculate dummy text based on image contrast/features
        avg_val = np.mean(crop_np)
        simulated_conf = min(0.95, max(0.60, float(avg_val / 255.0 + 0.3)))
        return "ဦးအေးမောင် ( Burmese Printed Text Stub )", round(simulated_conf, 3)
