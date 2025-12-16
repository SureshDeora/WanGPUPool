import torch
import logging
from diffusers import WanPipeline, WanImageToVideoPipeline
from diffusers.utils import export_to_video
import os

logger = logging.getLogger("WanInference")

class WanInferenceEngine:
    def __init__(self, model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", device="cuda"):
        self.device = device
        self.model_id = model_id
        self.pipeline = None
        logger.info(f"Initializing WanInferenceEngine with {model_id}...")

    def load_model(self):
        if self.pipeline is not None:
            return

        logger.info("Loading pipeline... (This may take time)")
        try:
            # Auto-detect pipeline type based on model ID or explicit type
            if "I2V" in self.model_id:
                 self.pipeline = WanImageToVideoPipeline.from_pretrained(
                    self.model_id, torch_dtype=torch.float16
                )
            else:
                self.pipeline = WanPipeline.from_pretrained(
                    self.model_id, torch_dtype=torch.float16
                )
            
            # CPU Offload is CRITICAL for Kaggle/Colab (15GB VRAM)
            # 1.3B fits easily, 14B requires strict offloading
            self.pipeline.enable_model_cpu_offload()
            
            # Optional: Enable VAE tiling to save memory during decode
            self.pipeline.enable_vae_tiling()
            
            logger.info("Model loaded successfully with CPU Offload enabled.")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def generate(self, prompt: str, output_path: str, negative_prompt="", num_inference_steps=30, guidance_scale=5.0, image=None):
        if self.pipeline is None:
            self.load_model()
            
        logger.info(f"Generating video for prompt: '{prompt}'")
        
        generator = torch.Generator(device="cpu").manual_seed(42) # Reproducibility
        
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "output_type": "np" # Numpy array for export_to_video
        }
        
        if image and isinstance(self.pipeline, WanImageToVideoPipeline):
            args["image"] = image

        try:
            output = self.pipeline(**args)
            video_frames = output.frames[0]
            
            logger.info(f"Saving video to {output_path}")
            export_to_video(video_frames, output_path, fps=15)
            return output_path
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise e
