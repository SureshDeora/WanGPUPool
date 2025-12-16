import asyncio
import json
import logging
import uuid
import sys
import os
import websockets
from typing import Optional

# Import the Real Inference Engine
try:
    from .inference import WanInferenceEngine
except ImportError:
    print("Warning: 'diffusers' not found. Install it via: pip install diffusers transformers accelerate")
    WanInferenceEngine = None

from .protocol import WorkerRegistration, ServerMessage, TaskPayload, TaskResult, DeviceType

# Configuration via Environment Variables
SERVER_URL = os.getenv("CONTROLLER_URL", "ws://localhost:8000/ws/worker")
WORKER_NAME = os.getenv("WORKER_NAME", "Kaggle-Worker-Generic")
MODEL_ID = os.getenv("MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

# Detect VRAM
import torch
try:
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        vram = 0.0
except:
    vram = 0.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WorkerClient")

class WorkerClient:
    def __init__(self, server_url: str, worker_id: str):
        self.server_url = f"{server_url}/{worker_id}"
        self.worker_id = worker_id
        self.ws = None
        
        # Initialize Engine only if we have GPUs
        if WanInferenceEngine and vram > 0:
            self.engine = WanInferenceEngine(model_id=MODEL_ID)
        else:
            self.engine = None
            logger.warning("Running in NO-GPU mode. Tasks will fail if they require inference.")

    async def connect(self):
        async with websockets.connect(self.server_url) as websocket:
            self.ws = websocket
            logger.info(f"Connected to Controller at {self.server_url}")
            
            # 1. Send Registration
            reg = WorkerRegistration(
                worker_id=self.worker_id,
                name=WORKER_NAME,
                device_type=DeviceType.GPU if vram > 0 else DeviceType.CPU,
                vram_gb=float(f"{vram:.1f}"),
                supported_ops=["WanUNet", "VAE"]
            )
            await websocket.send(reg.model_dump_json())
            logger.info("Sent Registration.")
            
            # 2. Event Loop
            try:
                while True:
                    msg_text = await websocket.recv()
                    msg = ServerMessage(**json.loads(msg_text))
                    
                    if msg.type == "task" and msg.payload:
                        await self.handle_task(msg.payload)
                    elif msg.type == "heartbeat":
                        pass
                        
            except websockets.ConnectionClosed:
                logger.warning("Connection closed by server.")

    async def handle_task(self, task: TaskPayload):
        logger.info(f"Received Task: {task.op_type} (ID: {task.task_id})")
        
        if not self.engine:
            await self.send_error(task.task_id, "No GPU/Engine available on this worker.")
            return

        try:
            # 1. Parse Config
            prompt = task.config.get("prompt", "a cinematic video")
            steps = task.config.get("steps", 20)
            
            # 2. Run Inference
            # We run this in a separate thread/process to not block the event loop
            # But for simplicity here, we run blocking (or use run_in_executor)
            output_file = f"output_{task.task_id}.mp4"
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                lambda: self.engine.generate(prompt, output_file, num_inference_steps=steps)
            )
            
            # 3. Upload Result (Mock for now, normally Upload to S3/R2)
            # In a real Kaggle/Colab setup, you'd push to HuggingFace Hub or S3
            logger.info(f"Video generated at {output_file}")
            
            result = TaskResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status="success",
                output_data={"video_path": output_file, "message": "Video generated locally. Check disk."}
            )
            
            if self.ws:
                await self.ws.send(result.model_dump_json())
                logger.info("Sent Result.")
                
        except Exception as e:
            logger.error(f"Task Failed: {e}")
            await self.send_error(task.task_id, str(e))

    async def send_error(self, task_id, msg):
        res = TaskResult(task_id=task_id, worker_id=self.worker_id, status="error", error_msg=msg, output_data={})
        if self.ws:
            await self.ws.send(res.model_dump_json())

async def main():
    # Generate unique ID
    wid = str(uuid.uuid4())[:8]
    client = WorkerClient(SERVER_URL, wid)
    
    # Retry loop
    while True:
        try:
            await client.connect()
        except Exception as e:
            logger.error(f"Connection failed: {e}. Retrying in 5s...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
