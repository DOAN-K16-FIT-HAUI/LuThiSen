from __future__ import annotations

from common.bases import BaseModel


class TextOCRSettings(BaseModel):
    config_name: str = 'vgg_transformer'
    device: str = 'cpu'
    model_path: str
    cnn_pretrained: bool
    vocab: str
