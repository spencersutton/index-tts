import torch
from dataclasses import dataclass
from typing import Optional, Union

log = ...

@dataclass(frozen=True)
class DeviceInfo:
    tops: dict[Union[torch.dtype, str], float]
    dram_bw_gbs: float
    dram_gb: float

_device_mapping: dict[str, DeviceInfo] = ...

def lookup_device_info(name: str) -> Optional[DeviceInfo]: ...
def datasheet_tops(dtype: torch.dtype, is_tf32: bool = ...) -> Optional[float]: ...
