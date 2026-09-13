# Core dependencies: dataclasses, environment variables, and path handling.
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load .env configuration up front.
load_dotenv()

@dataclass
class RerankerConfig:
    bge_reranker_large: str  # Local model path.
    bge_reranker_device: str       # Runtime device or model repository identifier.
    bge_reranker_fp16: bool    # Whether to enable half precision (1=True/0=False).

# Instantiate the config object, keeping the same style as lm_config.
reranker_config = RerankerConfig(
    bge_reranker_large=os.getenv("BGE_RERANKER_LARGE"),
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE"),
    # Convert 1/0 values from .env to booleans and support common string formats.
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16") in ("1", "True", "true", 1)
)
