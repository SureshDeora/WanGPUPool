from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any, Union
import uuid

class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"

class WorkerType(Enum):
    LOCAL = "local"
    KAGGLE = "kaggle"
    COLAB = "colab"

class TensorPrecision(Enum):
    FP32 = 32
    FP16 = 16
    BF16 = 16
    FP8 = 8
    INT8 = 8

@dataclass
class ModelSpec:
    name: str
    size_gb: float
    precision: TensorPrecision
    supports_sharding: bool = True

@dataclass
class Task:
    id: str
    name: str
    op_type: str  # e.g., "TextEncode", "UNet", "VAE"
    inputs: List[str]  # IDs of input tasks/data
    model_requirement: Optional[ModelSpec] = None
    estimated_compute_cost: float = 1.0  # Abstract units
    assigned_worker_ids: List[str] = field(default_factory=list)
    status: str = "PENDING"
    output_size_mb: float = 0.0

@dataclass
class Job:
    id: str
    workflow_json: Dict[str, Any]
    tasks: Dict[str, Task] = field(default_factory=dict)
    
@dataclass
class WorkerState:
    id: str
    name: str
    type: WorkerType
    device_type: DeviceType
    total_vram_gb: float
    used_vram_gb: float = 0.0
    loaded_models: Set[str] = field(default_factory=set) # Model names
    current_task_id: Optional[str] = None
    
    @property
    def free_vram_gb(self) -> float:
        return self.total_vram_gb - self.used_vram_gb
