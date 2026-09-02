from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str | None = None
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agentdiff"
    model_id: str = "claude-opus-5"
    effort: str = "high"
    workspace_dir: Path = Path(".")
