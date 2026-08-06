from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Burmese Insurance Claim Form OCR Backend"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Base Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    CROPS_DIR: Path = STORAGE_DIR / "crops"
    EXPORTS_DIR: Path = STORAGE_DIR / "exports"
    
    # Pre-trained Model Configuration
    MODEL_DIR: Path = BASE_DIR / "models_weights" / "trocr-small-printed"
    DEVICE: str = "cpu"  # 'cuda' or 'cpu'
    
    # Quality & Threshold Settings
    BLUR_LAPLACIAN_THRESHOLD: float = 100.0  # Variance below this is flagged blurry
    CONFIDENCE_THRESHOLD: float = 0.85        # Scores below this require human review
    
    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")


settings = Settings()

# Ensure directories exist
for path in [settings.STORAGE_DIR, settings.UPLOAD_DIR, settings.CROPS_DIR, settings.EXPORTS_DIR, settings.MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)
