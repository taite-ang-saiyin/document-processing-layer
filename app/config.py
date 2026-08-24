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
    PREPROCESSED_CROPS_DIR: Path = STORAGE_DIR / "preprocessed_crops"
    LINE_CROPS_DIR: Path = STORAGE_DIR / "line_crops"
    EXPORTS_DIR: Path = STORAGE_DIR / "exports"
    ALIGNED_PAGES_DIR: Path = STORAGE_DIR / "aligned_pages"
    TEMPLATE_REFERENCES_DIR: Path = STORAGE_DIR / "template_references"
    
    # Pre-trained Model Configuration
    MODEL_DIR: Path = BASE_DIR / "models_weights" / "trocr-small-printed"
    DEVICE: str = "cpu"  # 'cuda' or 'cpu'
    
    # Quality & Threshold Settings
    BLUR_LAPLACIAN_THRESHOLD: float = 100.0  # Variance below this is flagged blurry
    CONFIDENCE_THRESHOLD: float = 0.80        # Raw OCR scores below this require human review
    ALIGNMENT_SCORE_THRESHOLD: float = 0.50   # Scores below this reject extraction
    TEMPLATE_MATCH_SCORE_THRESHOLD: float = 0.50
    TEMPLATE_MATCH_MARGIN: float = 0.10       # Required lead over the runner-up
    LINE_DETECTION_MIN_SCORE: float = 0.45
    LINE_DETECTION_PADDING_PX: int = 6
    LLM_POST_CORRECTION_ENABLED: bool = False
    LLM_POST_CORRECTION_URL: str = "http://insurance-vlm:8000"
    LLM_POST_CORRECTION_API_KEY: str = "local-vlm-key"
    LLM_POST_CORRECTION_TIMEOUT_SECONDS: float = 180.0
    # The example template is useful in standalone development, but production
    # receives only human-approved registrations from the orchestrator.
    SEED_DEFAULT_TEMPLATE: bool = True
    
    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env", extra="ignore")


settings = Settings()

# Ensure directories exist
for path in [
    settings.STORAGE_DIR,
    settings.UPLOAD_DIR,
    settings.CROPS_DIR,
    settings.PREPROCESSED_CROPS_DIR,
    settings.LINE_CROPS_DIR,
    settings.EXPORTS_DIR,
    settings.ALIGNED_PAGES_DIR,
    settings.TEMPLATE_REFERENCES_DIR,
    settings.MODEL_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
