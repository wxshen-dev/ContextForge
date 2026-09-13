import sys

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState

@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Import vector store (node_import_milvus).
    This node writes processed vector data into Milvus.
    Planned implementation:
    1. Connect to Milvus.
    2. Delete old data by item_name for idempotency.
    3. Batch-insert new vector data.
    """
    return state
