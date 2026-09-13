# Core dependencies: dataclasses, environment variables, and path handling.
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load .env configuration up front.
load_dotenv()

# Embedding configuration for BGE-M3.
@dataclass
class EmbeddingConfig:
    bge_m3_path: str  # Local model path.
    bge_m3: str       # Model repository identifier.
    bge_device: str   # Runtime device, such as cuda:0 or cpu.
    bge_fp16: bool    # Whether to enable half precision (1=True/0=False).

# Instantiate the config object, keeping the same style as lm_config.
embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH"),
    bge_m3=os.getenv("BGE_M3"),
    bge_device=os.getenv("BGE_DEVICE"),
    # Convert 1/0 values from .env to booleans and support common string formats.
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1)
)
