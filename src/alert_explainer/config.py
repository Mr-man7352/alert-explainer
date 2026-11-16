"""Project paths and settings. Override any path with an AE_* environment variable."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AE_", env_file=".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    docs_dir: Path = PROJECT_ROOT / "docs"
    seed: int = 42
    # PyTorch device: auto (CUDA > MPS > CPU), cpu, mps or cuda. `ae bench-mlp` says which is faster.
    device: str = "auto"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def get_settings() -> Settings:
    return Settings()
