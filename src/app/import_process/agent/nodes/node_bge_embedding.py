import sys

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState

@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Embedding generation (node_bge_embedding).
    This node uses the BGE-M3 model to convert text into embeddings.
    Planned implementation:
    1. Load the BGE-M3 model.
    2. Generate dense and sparse vectors for each chunk.
    3. Prepare the data format required for writing to Milvus.
    """
    return state
