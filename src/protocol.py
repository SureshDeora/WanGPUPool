from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from enum import Enum

class DeviceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"

class WorkerStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"

# Messages from Worker -> Server
class WorkerRegistration(BaseModel):
    worker_id: str
    name: str  # e.g. "Kaggle-T4-1"
    device_type: DeviceType
    vram_gb: float
    supported_ops: List[str] # ["TextEncode", "WanUNet", "VAE"]

class TaskResult(BaseModel):
    task_id: str
    worker_id: str
    status: str # "success", "error"
    output_data: Dict[str, Any] # e.g. {"s3_url": "..."}
    error_msg: Optional[str] = None

# Messages from Server -> Worker
class TaskPayload(BaseModel):
    task_id: str
    op_type: str
    model_uri: Optional[str]
    input_data: Dict[str, Any] # e.g. {"latents_url": "...", "prompt_embeds": "..."}
    config: Dict[str, Any] # parameters like steps, cfg

class ServerMessage(BaseModel):
    type: str # "task", "ack", "heartbeat"
    payload: Optional[TaskPayload] = None
