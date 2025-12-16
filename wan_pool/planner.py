import json
import logging
from typing import List, Dict, Union, Tuple
import networkx as nx

from .schema import Job, Task, ModelSpec, TensorPrecision
from .estimator import VRAMEstimator
from .worker import Worker, VirtualWorkerGroup

logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self, json_data: Dict):
        self.data = json_data
        self.graph = nx.DiGraph()
        
    def build(self) -> Job:
        job_id = self.data.get("id", "unknown_job")
        nodes = self.data.get("nodes", [])
        links = self.data.get("links", [])
        
        # 1. Create Tasks from Nodes
        tasks = {}
        link_map = {} # link_id -> source_task_id
        
        for n in nodes:
            node_id = str(n["id"])
            node_type = n["type"]
            
            # Determine Op Type and Model Requirements
            op_type = "Generic"
            model_spec = None
            compute_cost = 1.0
            
            if "Loader" in node_type:
                # Just a loader, low cost, but registers the model
                pass 
            elif "CLIPTextEncode" in node_type:
                op_type = "TextEncode"
                compute_cost = 2.0
                # Finds inputs later
            elif "KSampler" in node_type or "ModelSampling" in node_type:
                op_type = "Diffusion"
                compute_cost = 20.0
            elif "VAEDecode" in node_type:
                op_type = "VAEDecode"
                compute_cost = 5.0
            elif "WanImageToVideo" in node_type:
                 op_type = "PipelineInit"
            
            tasks[node_id] = Task(
                id=node_id,
                name=f"{node_type}_{node_id}",
                op_type=op_type,
                inputs=[],
                estimated_compute_cost=compute_cost
            )
            
            # Extract widgets for model info (naive parsing)
            if "widgets_values" in n:
                for w in n["widgets_values"]:
                    if isinstance(w, str) and (".safetensors" in w or ".pt" in w):
                        # Detect model size/precision
                        prec = TensorPrecision.FP16
                        if "fp8" in w.lower():
                            prec = TensorPrecision.FP8
                        
                        size = VRAMEstimator.estimate_model_size(w, prec)
                        tasks[node_id].model_requirement = ModelSpec(
                            name=w,
                            size_gb=size,
                            precision=prec
                        )
                        logger.info(f"Node {node_id} requires model {w} ({size:.2f}GB)")

        # 2. Build Links (Dependencies)
        # Link structure: [id, origin_id, origin_slot, target_id, target_slot, type]
        for l in links:
            link_id = l[0]
            origin_node_id = str(l[1])
            target_node_id = str(l[3])
            
            if target_node_id in tasks:
                tasks[target_node_id].inputs.append(origin_node_id)
            
            self.graph.add_edge(origin_node_id, target_node_id)
            
        return Job(id=job_id, workflow_json=self.data, tasks=tasks)

class PartitionEngine:
    def __init__(self, workers: List[Worker]):
        self.workers = workers
        
    def find_worker_for_task(self, task: Task) -> Union[Worker, VirtualWorkerGroup, None]:
        req = task.model_requirement
        if not req:
            # No model required, find least busy worker (or generic CPU for text)
            # Prefer CPU for text encoding if available to save GPU VRAM
            if "Text" in task.op_type:
                for w in self.workers:
                    if w.state.device_type.value == "cpu":
                        return w
            # Otherwise pick first available
            return self.workers[0]
        
        required_size = req.size_gb
        LARGE_MODEL_THRESHOLD = 10.0 # GB
        
        if required_size > LARGE_MODEL_THRESHOLD:
            # --- LARGE MODEL STRATEGY (GPU ONLY) ---
            
            # 1. Try Single GPU
            valid_gpu_workers = [w for w in self.workers 
                                 if w.state.device_type.value == 'gpu' and w.can_accommodate(required_size)]
            
            if valid_gpu_workers:
                def score(w):
                    has_model = 1 if req.name in w.state.loaded_models else 0
                    return (has_model, w.state.free_vram_gb)
                
                return sorted(valid_gpu_workers, key=score, reverse=True)[0]
                
            # 2. Try FSDP (Multi-GPU)
            logger.warning(f"Task {task.name} requires {required_size:.2f}GB. Single GPU fit not found. Attempting FSDP split.")
            
            # Simple greedy bin packing for FSDP
            sorted_gpus = sorted([w for w in self.workers if w.state.device_type.value == 'gpu'], 
                                 key=lambda w: w.state.free_vram_gb, reverse=True)
            
            current_sum = 0
            group = []
            for w in sorted_gpus:
                group.append(w)
                current_sum += w.state.free_vram_gb
                
                # Check if the split fits everyone
                shard_size = required_size / len(group)
                fits_all = all(member.can_accommodate(shard_size) for member in group)
                
                if current_sum >= required_size and fits_all:
                    logger.info(f"Found FSDP Group: {[w.state.name for w in group]} (Total: {current_sum:.2f}GB, Shard: {shard_size:.2f}GB)")
                    return VirtualWorkerGroup(group)
            
            # 3. Fail (Do not fallback to CPU)
            return None

        else:
            # --- SMALL MODEL STRATEGY (ANY FIT) ---
            valid_workers = [w for w in self.workers if w.can_accommodate(required_size)]
            
            if valid_workers:
                # Prefer GPU if available even for small tasks to avoid CPU bottleneck, unless it's text
                # But for now, just sort by model residency > free ram
                def score(w):
                    has_model = 1 if req.name in w.state.loaded_models else 0
                    is_gpu = 1 if w.state.device_type.value == 'gpu' else 0
                    return (has_model, is_gpu, w.state.free_vram_gb)
                
                best_worker = sorted(valid_workers, key=score, reverse=True)[0]
                
                # Regression Guard: Ensure we didn't accidentally pick CPU for a heavy task
                if required_size > 5.0 and best_worker.state.device_type.value == 'cpu':
                    if any(w.state.device_type.value == 'gpu' for w in self.workers):
                        logger.warning(f"Task {task.name} ({required_size}GB) assigned to CPU despite GPUs existing. Check logic.")
                        
                return best_worker
            
            return None
