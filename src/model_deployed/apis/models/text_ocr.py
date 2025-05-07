"""Face detector API format input """
from __future__ import annotations

from typing import Any
from typing import List

from common.bases import BaseModel

# from typing import Any


class APIInput(BaseModel):
    img: List[List[List[int]]]
    classes: List[str]
    bboxes: List[List[Any]]


class APIOutput(BaseModel):
    cls: str
    course: str
    date: str
    hktt: str
    msv: str
    name: str
