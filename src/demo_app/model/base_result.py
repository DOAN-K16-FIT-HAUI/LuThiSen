from __future__ import annotations

from common.bases import BaseModel


class BaseInfo(BaseModel):
    cls: str
    course: str
    date: str
    hktt: str
    msv: str
    name: str
