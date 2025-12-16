from typing import Tuple, Dict
from .schema import TensorPrecision, ModelSpec

# Constants for Wan 2.2 Model Sizes (Estimates)
# Assuming FP16 unless specified.
# 14B Params -> ~28GB FP16, ~14GB FP8.
# VAE -> ~300MB
# Text Enc (T5-XXL) -> ~10GB FP16, ~5GB FP8

class VRAMEstimator:
    @staticmethod
    def estimate_model_size(model_name: str, precision: TensorPrecision) -> float:
        """Returns estimated size in GB."""
        name_lower = model_name.lower()
        
        # Base sizes in Parameters (Billions)
        params = 0.0
        if "14b" in name_lower:
            params = 14.0
        elif "t5" in name_lower or "clip" in name_lower:
            params = 4.7 # T5-XXL roughly
        elif "vae" in name_lower:
            params = 0.1 # Small
        else:
            params = 1.0 # Default fallback
            
        # Calculate bytes
        bytes_per_param = precision.value / 8.0
        size_gb = (params * 10**9 * bytes_per_param) / (1024**3)
        
        # Add safety buffer for runtime activations (very rough heuristic)
        # Activation overhead is usually 20-30% of weights for inference
        return size_gb * 1.2

    @staticmethod
    def estimate_latent_size(width: int, height: int, frames: int) -> float:
        # Wan latents are compressed. Assuming standard SD-like or Wan compression.
        # Example: 1280x720 -> Latent 160x90 channels=16
        # Let's assume (W/8 * H/8 * F * C * 2bytes) / 1024^2 for MB
        # Wan uses 3D VAE.
        
        # Rough estimate: 2MB per frame at 720p latents
        size_mb = frames * 2.0
        return size_mb / 1024.0 # to GB
