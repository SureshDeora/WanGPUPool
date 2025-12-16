import asyncio
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
import uuid
import json

from .protocol import WorkerRegistration, ServerMessage, TaskPayload, TaskResult, DeviceType
# We will reuse the Planner logic logic from the simulation, but adapted for real objects
# For now, we implement a simplified Scheduler inside the server
from wan_pool.schema import Task, ModelSpec, TensorPrecision
from wan_pool.estimator import VRAMEstimator

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WanController")

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.worker_specs: Dict[str, WorkerRegistration] = {}
        self.worker_status: Dict[str, str] = {} # ID -> Status

    async def connect(self, websocket: WebSocket, registration: WorkerRegistration):
        # WebSocket is already accepted by the endpoint
        self.active_connections[registration.worker_id] = websocket
        self.worker_specs[registration.worker_id] = registration
        self.worker_status[registration.worker_id] = "idle"
        logger.info(f"Worker connected: {registration.name} ({registration.vram_gb}GB VRAM)")

    def disconnect(self, worker_id: str):
        if worker_id in self.active_connections:
            del self.active_connections[worker_id]
        if worker_id in self.worker_specs:
            del self.worker_specs[worker_id]
        if worker_id in self.worker_status:
            del self.worker_status[worker_id]
        logger.info(f"Worker disconnected: {worker_id}")

    async def send_task(self, worker_id: str, task: TaskPayload):
        if worker_id in self.active_connections:
            msg = ServerMessage(type="task", payload=task)
            await self.active_connections[worker_id].send_text(msg.model_dump_json())
            self.worker_status[worker_id] = "busy"
            return True
        return False

    def mark_idle(self, worker_id: str):
        if worker_id in self.worker_status:
            self.worker_status[worker_id] = "idle"

    def get_idle_workers(self) -> List[WorkerRegistration]:
        return [self.worker_specs[wid] for wid, status in self.worker_status.items() if status == "idle"]

manager = ConnectionManager()
task_queue: List[TaskPayload] = []

# --- API Endpoints ---

@app.websocket("/ws/worker/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    # Handshake: Expect Registration JSON first
    try:
        data = await websocket.receive_text()
        reg_dict = json.loads(data)
        registration = WorkerRegistration(**reg_dict)
        
        await manager.connect(websocket, registration)
        
        try:
            while True:
                # Listen for results
                data = await websocket.receive_text()
                result_dict = json.loads(data)
                result = TaskResult(**result_dict)
                
                logger.info(f"Received result for Task {result.task_id} from {result.worker_id}: {result.status}")
                manager.mark_idle(result.worker_id)
                
                # In a real system, we would trigger the next step in the DAG here
                
        except WebSocketDisconnect:
            manager.disconnect(registration.worker_id)
            
    except Exception as e:
        logger.error(f"Connection error: {e}")
        await websocket.close()

class JobSubmission(BaseModel):
    workflow_json: Dict[str, Any]

@app.post("/submit_job")
async def submit_job(job: JobSubmission):
    # Simplified parsing -> In real impl, this uses GraphBuilder
    # Here we mock creating a single Task
    
    # Check if we have nodes
    nodes = job.workflow_json.get("nodes", [])
    task_id = str(uuid.uuid4())
    
    # Mock Task Creation
    # Real logic: Traverse graph, topological sort, create TaskPayloads
    logger.info(f"Received Job with {len(nodes)} nodes. Queueing mock task.")
    
    # Extract Prompt and Config from input
    prompt = job.workflow_json.get("prompt", "a cinematic video")
    steps = job.workflow_json.get("steps", 20)
    
    mock_task = TaskPayload(
        task_id=task_id,
        op_type="WanUNet",
        model_uri="wan2.2_14b_fp8.safetensors",
        input_data={"latents": "s3://bucket/latents_0.pt"},
        config={"prompt": prompt, "steps": steps}
    )
    
    task_queue.append(mock_task)
    return {"status": "queued", "job_id": "mock_job", "task_id": task_id}

# --- Background Scheduler ---

async def scheduler_loop():
    logger.info("Scheduler started.")
    while True:
        if task_queue:
            # Simple FIFO
            task = task_queue[0]
            
            # Simple Planner: Find First Idle Worker that fits
            # (Real impl: Use planner.py logic here)
            idle_workers = manager.get_idle_workers()
            candidate = None
            
            # Use Estimator to guess size
            # Hardcoded mock for now
            req_size = 15.0 if "fp8" in task.model_uri else 30.0
            
            for w in idle_workers:
                if w.vram_gb >= req_size:
                    candidate = w
                    break
            
            if candidate:
                logger.info(f"Scheduling Task {task.task_id} to {candidate.name}")
                success = await manager.send_task(candidate.worker_id, task)
                if success:
                    task_queue.pop(0)
            else:
                # No worker available, wait
                # In real FSDP: Check for groups here
                pass
                
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler_loop())
