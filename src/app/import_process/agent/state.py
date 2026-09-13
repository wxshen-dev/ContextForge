from typing import TypedDict
import copy
from app.core.logger import logger

class ImportGraphState(TypedDict):
    """
    Graph state definition containing all fields produced and consumed by nodes.
    TypedDict provides autocomplete and type checking in code.
    Use dictionary-style access, such as state["session_id"] or state.get("embedding_chunks").
    """
    task_id: str          # Unique task ID used for log tracing.

    # --- Flow-control flags ---
    is_md_read_enabled: bool   # Whether to enable the Markdown input path.
    is_pdf_read_enabled: bool  # Whether to enable the PDF input path.

    # --- Path fields ---
    local_dir: str        # Current working directory or output directory.
    local_file_path: str  # Original input file path.
    file_title: str       # File title without extension.
    pdf_path: str         # PDF file path when the input is a PDF.
    md_path: str          # Markdown file path after conversion or direct input.

    # --- Content data ---
    md_content: str       # Full Markdown content.
    chunks: list          # Split text chunks with metadata.
    item_name: str        # Identified item name, used to improve retrieval.

    # --- Database fields ---
    embeddings_content: list # List containing vector data ready to write to Milvus.


# Default initial graph state.
graph_default_state: ImportGraphState = {
    "task_id":"",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}

def create_default_state(**overrides) -> ImportGraphState:
    """
    Create a default state with optional overrides.
    Args:
        **overrides: Fields to override.
    Returns:
        A new state instance.
    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # Start from the default state.
    state = copy.deepcopy(graph_default_state)
    # Override default values.
    state.update(overrides)
    # Return the created state dictionary.
    return state

def get_default_state() -> ImportGraphState:
    """
    Return a new state instance to avoid mutating the global default.
    """
    return copy.deepcopy(graph_default_state)

if __name__ == "__main__":
    """
    Test.
    """
    # Create a default state.
    state = create_default_state(local_file_path="multimeter_rs_12_usage.pdf")
    logger.info(state)
