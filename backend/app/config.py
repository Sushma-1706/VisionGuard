from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings; values can be overridden through environment variables."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "VisionGuard"
    database_url: str = "sqlite:///./visionguard.db"
    model_path: Path = Path("../ml/saved_models/visionguard_resnet18.pt")
    max_upload_bytes: int = 10 * 1024 * 1024
    image_size: int = 224
    allowed_origins: str = "http://localhost:5173"


settings = Settings()
