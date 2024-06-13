from typing import List

from pydantic import BaseModel, validator  # noqa pylint: disable=W0611


class ReplicaSchema(BaseModel):
    """Схема: реплика"""

    text: str
    start: float
    end: float


class DialogSchema(BaseModel):
    """Схема: диалог"""

    dialog: List[ReplicaSchema]
