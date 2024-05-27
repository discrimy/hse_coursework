from enum import Enum


class Device(str, Enum):
    """Устройство для работы моделей"""

    CPU = "cpu"
    CUDA = "cuda"
