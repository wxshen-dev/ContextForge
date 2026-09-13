import sys

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState

@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Image processing (node_md_img).
    This node processes image assets referenced in Markdown.
    Planned implementation:
    1. Scan image links in Markdown.
    2. Upload images to MinIO object storage.
    3. Optionally call a multimodal model to generate image descriptions.
    4. Replace Markdown image links with MinIO URLs.
    """
    return state
