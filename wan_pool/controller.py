import asyncio
import logging
import time
from typing import List, Dict, Union

from .schema import Job, Task
from .planner import GraphBuilder, PartitionEngine
from .worker import Worker, VirtualWorkerGroup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Controller:
    def __init__(self, workers: List[Worker]):
        self.workers = workers
        self.planner = PartitionEngine(workers)
        self.queue: List[Job] = []
        
    def submit_job(self, workflow_json: Dict):
        builder = GraphBuilder(workflow_json)
        job = builder.build()
        self.queue.append(job)
        logger.info(f"Job {job.id} submitted with {len(job.tasks)} tasks.")
        
    async def run(self):
        while self.queue:
            job = self.queue.pop(0)
            await self.execute_job(job)
            
    async def execute_job(self, job: Job):
        logger.info(f"Starting execution of Job {job.id}")
        
        # Simple topological sort execution (BFS-ish)
        # We need to track completed tasks
        completed = set()
        
        # Continuously find ready tasks
        while len(completed) < len(job.tasks):
            ready_tasks = []
            for tid, task in job.tasks.items():
                if tid in completed:
                    continue
                
                # Check dependencies
                deps_met = all(dep in completed for dep in task.inputs)
                if deps_met:
                    ready_tasks.append(task)
            
            if not ready_tasks:
                if len(completed) < len(job.tasks):
                    logger.error("Deadlock detected! Cycles in graph or unreachable tasks.")
                    break
                else:
                    break
                    
            # Execute ready tasks (simulate parallel dispatch)
            futures = []
            for task in ready_tasks:
                # 1. Assign
                executor = self.planner.find_worker_for_task(task)
                if not executor:
                    logger.error(f"Could not schedule task {task.name}. No worker has enough VRAM.")
                    return
                
                # 2. Allocate & Run
                futures.append(self.run_task(task, executor))
                
            # Wait for this batch to finish (simplified sync-barrier for simulation)
            results = await asyncio.gather(*futures)
            for tid in results:
                completed.add(tid)
                
        logger.info(f"Job {job.id} Completed Successfully.")

    async def run_task(self, task: Task, executor: Union[Worker, VirtualWorkerGroup]) -> str:
        # Simulate Allocation
        req_size = task.model_requirement.size_gb if task.model_requirement else 0.5
        model_name = task.model_requirement.name if task.model_requirement else None
        
        try:
            await executor.allocate(req_size, model_name)
            
            # Simulate Data Transfer (Mock)
            # If previous tasks were on different workers, add transfer time
            # For simulation, just add random small delay
            await asyncio.sleep(0.2) 
            
            # Simulate Execution
            await executor.execute_task(task.name, task.estimated_compute_cost)
            
            # Cleanup (Greedy release for this simulation)
            # In real system we might cache
            await executor.free(req_size, model_name)
            
            return task.id
            
        except Exception as e:
            logger.error(f"Task {task.name} failed: {e}")
            raise e
