import os

from pathlib import Path
from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Entry node (node_entry).
    This graph entry point receives external input and decides the flow path.
    Planned implementation:
        1. Track task status in running and completed lists.
        2. Detect the input type from state["local_file_path"] and update related fields:
           is_md_read_enabled or is_pdf_read_enabled, plus md_path or pdf_path.
        3. Log unsupported file types and let the routing node handle termination.
        4. Derive file_title as a fallback identifier for later item_name recognition.
    """

    # 1. Track task status.
    add_running_task(state['task_id'],'node_entry')

    # 2. Detect file type.
    local_file_path = state['local_file_path']
    if not local_file_path:
        logger.warning("No input file path was provided; skipping directly to the end node.")
        add_done_task(state['task_id'], 'node_entry')
        return state

    if local_file_path.endswith(".md"):
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    elif local_file_path.endswith(".pdf"):
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    else:
        logger.warning(
            f"local_file_path was provided, but the file type is unsupported. "
            f"Only .md and .pdf files are currently supported. Please check the input file: {local_file_path}"
        )
        add_done_task(state['task_id'], 'node_entry')
        return state

    # 3. Derive the file identifier.
    # os.path-based approach.
    file_title_os = os.path.basename(local_file_path).split(".")[0]
    # pathlib-based approach.
    file_title = Path(local_file_path).stem # File name: .name, folder name: .parent, file extension: .suffix.
    state['file_title']= file_title

    add_done_task(state['task_id'], 'node_entry')
    return state

if __name__ == '__main__':

    # Unit test covering unsupported, Markdown, and PDF inputs.
    logger.info("===== Starting node_entry unit test =====")

    # Test 1: Unsupported TXT file.
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="lenovo_dolphin_user_manual.txt"
    )
    node_entry(test_state1)

    # Test 2: Markdown file.
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="xiaomi_user_manual.md"
    )
    node_entry(test_state2)

    # Test 3: PDF file.
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="multimeter_usage.pdf"
    )
    node_entry(test_state3)

    logger.info("===== Finished node_entry unit test =====")
