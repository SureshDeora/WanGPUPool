import asyncio
import logging
from typing import List, Optional
from .schema import WorkerState, WorkerType, DeviceType, ModelSpec

logger = logging.getLogger(__name__)

class Worker:
    def __init__(self, 
                 worker_id: str, 
                 name: str, 
                 w_type: WorkerType, 
                 vram_gb: float, 
                 device: DeviceType = DeviceType.GPU):
        self.state = WorkerState(
            id=worker_id,
            name=name,
            type=w_type,
            device_type=device,
            total_vram_gb=vram_gb
        )
        self.network_latency_ms = 0 if w_type == WorkerType.LOCAL else 100
        self.bandwidth_mbps = 10000 if w_type == WorkerType.LOCAL else 500 # Simulated WAN bandwidth

    def can_accommodate(self, size_gb: float) -> bool:
        """Checks if the worker has enough free VRAM."""
        return self.state.free_vram_gb >= size_gb

    async def allocate(self, size_gb: float, model_name: Optional[str] = None):
        """Simulate memory allocation."""
        if not self.can_accommodate(size_gb):
            raise MemoryError(f"Worker {self.state.name} OOM! Free: {self.state.free_vram_gb}GB, Requested: {size_gb}GB")
        
        self.state.used_vram_gb += size_gb
        if model_name:
            self.state.loaded_models.add(model_name)
        logger.info(f"[{self.state.name}] Allocated {size_gb:.2f}GB. Used: {self.state.used_vram_gb:.2f}/{self.state.total_vram_gb}GB")

    async def free(self, size_gb: float, model_name: Optional[str] = None):
        """Simulate memory release."""
        self.state.used_vram_gb = max(0.0, self.state.used_vram_gb - size_gb)
        if model_name and model_name in self.state.loaded_models:
            self.state.loaded_models.remove(model_name)
        logger.info(f"[{self.state.name}] Freed {size_gb:.2f}GB. Used: {self.state.used_vram_gb:.2f}/{self.state.total_vram_gb}GB")

    async def execute_task(self, task_name: str, duration: float):
        """Simulate task execution."""
        logger.info(f"[{self.state.name}] execution started: {task_name} ({duration}s)")
        self.state.current_task_id = task_name
        await asyncio.sleep(duration)
        self.state.current_task_id = None
        logger.info(f"[{self.state.name}] execution finished: {task_name}")

class VirtualWorkerGroup:
    """Represents a group of workers acting as one (FSDP/Sharding)."""
    def __init__(self, workers: List[Worker]):
        self.workers = workers
        self.id = "group_" + "_".join([w.state.id for w in workers])
        self.name = "FSDP_Group[" + ",".join([w.state.name for w in workers]) + "]"

    @property
    def total_vram_gb(self) -> float:
        return sum(w.state.total_vram_gb for w in self.workers)

    @property
    def free_vram_gb(self) -> float:
        return sum(w.state.free_vram_gb for w in self.workers)

    async def allocate(self, total_size_gb: float, model_name: Optional[str] = None):
        # Naive sharding: Split equally
        per_worker = total_size_gb / len(self.workers)
        logger.info(f"[{self.name}] Allocating split model {total_size_gb}GB ({per_worker:.2f}GB per worker)")
        
        futures = []
        for w in self.workers:
            futures.append(w.allocate(per_worker, model_name))
        await asyncio.gather(*futures)

    async def free(self, total_size_gb: float, model_name: Optional[str] = None):
        per_worker = total_size_gb / len(self.workers)
        logger.info(f"[{self.name}] Freeing split model")
        futures = []
        for w in self.workers:
            futures.append(w.free(per_worker, model_name))
        await asyncio.gather(*futures)

    async def execute_task(self, task_name: str, base_duration: float):
        # Distributed overhead penalty
        # Latency = max(worker_latency) + synchronization overhead
        overhead = 0.5 * len(self.workers) # 0.5s per extra worker overhead
        duration = (base_duration / len(self.workers)) + overhead
        
        logger.info(f"[{self.name}] execution started (Distributed): {task_name} (Est: {duration:.2f}s)")
        futures = []
        for w in self.workers:
            futures.append(w.execute_task(task_name + "_shard", duration))
        await asyncio.gather(*futures)
        logger.info(f"[{self.name}] execution finished: {task_name}")
