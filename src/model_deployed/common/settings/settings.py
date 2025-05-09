from __future__ import annotations

from dotenv import find_dotenv
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from .models import CardAlignSettings
from .models import CardDetectorSettings
from .models import TextDetectorSettings
from .models import TextOCRSettings

# test in local
load_dotenv(find_dotenv('.env'), override=True)


class Settings(BaseSettings):
    # embedding: EmbeddingSettings
    card_detector: CardDetectorSettings
    text_detector: TextDetectorSettings
    text_ocr: TextOCRSettings
    card_align: CardAlignSettings

    class Config:
        env_nested_delimiter = '__'
