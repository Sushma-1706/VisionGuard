from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings; values can be overridden through environment variables."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "VisionGuard"
    database_url: str = "sqlite:///./visionguard.db"
    model_path: Path = Path("../ml/saved_models/visionguard_resnet18.pt")
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 4_000_000
    max_image_dimension: int = 4_096
    image_size: int = 224
    allowed_origins: str = "http://localhost:5173"

    @field_validator("database_url")
    @classmethod
    def require_local_sqlite_database(cls, value: str) -> str:
        """Development deployments only support SQLite database URLs.

        Restricting the scheme avoids accidentally accepting an attacker- or
        deployment-controlled connection string for an arbitrary database.
        """
        if not value.startswith("sqlite:///"):
            raise ValueError("DATABASE_URL must use the sqlite:/// scheme.")
        return value


settings = Settings()
