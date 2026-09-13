import sys

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState

@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Document splitting (node_document_split).
    This node splits long documents into smaller chunks for retrieval.
    Planned implementation:
    1. Recursively split by Markdown heading hierarchy.
    2. Split overly long paragraphs a second time.
    3. Generate chunk lists with metadata such as heading paths.
    """
    return state
