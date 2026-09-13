import sys

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Item-name recognition (node_item_name_recognition).
    This node identifies the item or product name described by the document.
    Planned implementation:
    1. Read the first few paragraphs of the document.
    2. Call an LLM to identify what the document is about, such as "Fluke 17B+ multimeter".
    3. Store the result in state["item_name"] for later idempotent cleanup.
    """
    return state
