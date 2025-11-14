"""Configuration settings for the project."""

import os
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Project settings."""

    model_config = SettingsConfigDict(
        env_file="./.env", env_file_encoding="utf-8", extra="allow"
    )

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = DATA_DIR / "indexes"

