import functools
from enum import Enum


class Device(str, Enum):
    """Устройство для работы моделей"""

    CPU = "cpu"
    CUDA = "cuda"


@functools.cache
def get_device() -> Device:
    import torch

    if torch.cuda.is_available():
        device = Device.CUDA
    else:
        device = Device.CPU
    return device
