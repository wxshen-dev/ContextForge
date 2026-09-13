import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config

@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    Node: PDF to Markdown (node_pdf_to_md).
    This node converts unstructured PDF data into structured Markdown data.
    """
    # 1. Track task status with running and completed markers.
    add_running_task(state['task_id'], 'node_pdf_to_md')
    # 2. Validate that required paths are complete and exist.
    pdf_path_obj , output_dir_pbj = step_1_validate_paths(state)
    # 3. step_2_upload_and_poll
    zip_url = step_2_upload_and_poll(pdf_path_obj, output_dir_pbj)
    # 4. step_3_download_and_extract
    md_path = step_3_download_and_extract(zip_url, output_dir_pbj, pdf_path_obj.stem)
    # 5. Store the response result back into state.
    state['md_path'] = md_path
    # Original file-reading approach.
    with open(md_path, 'r', encoding='utf-8') as f:
        state['md_content'] = f.read()
    add_done_task(state['task_id'], 'node_pdf_to_md')
    return state

@step_log("step_1_validate_paths")
def step_1_validate_paths(state):
    """
    Step 1: Validate and initialize paths.
    Validate the PDF input file and output directory:
    1. PDF path must be non-empty and point to an existing file.
    2. Output directory defaults to project_root/output and is created when missing.
    3. Paths are converted to Path objects for consistent cross-platform handling.
    :param state: Workflow state containing pdf_path and local_dir.
    :return: Tuple of validated PDF path and output directory path.
    """
    # 1. Read path parameters.
    pdf_path = state.get("pdf_path", "").strip()
    local_dir = state.get("local_dir", "").strip()
    # 2. Validate required parameters.
    if not pdf_path:
        raise ValueError("pdf_path cannot be empty. Please provide a valid PDF file path.")
    if not local_dir:
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = str(local_dir)
        logger.warning(f"No output directory was specified. Using default path: {local_dir}")
    # 3. Convert to Path objects for normalized path handling.
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    # 4. Validate paths: strict input validation and automatic output repair.
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path_obj}. Please check the file path.")
    if not local_dir_obj.exists():
        logger.warning(f"Output directory does not exist; creating it automatically: {local_dir_obj}")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj

@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj, output_dir_obj):
    """
    Step 2: Upload the PDF to MinerU and poll the parsing task status.
    Flow: validate configuration, get upload URL, upload file, and poll until done, failed, or timed out.
    Args:
        pdf_path_obj: Validated PDF Path object.
        output_dir_obj: Output directory Path object.
    Returns:
        full_zip_url for the parsing result package.
    Raises:
        ValueError for missing configuration, RuntimeError for request/upload failures, and TimeoutError for timeouts.
    """
    # 1. Validate configuration before making requests.
    if not mineru_config.base_url or not mineru_config.api_key:
        raise ValueError("MinerU base_url or api_key is not configured. Please check the configuration file.")

    # 2. Build headers and call the batch API to get a signed upload URL and batch ID.
    request_headers = {
        "Authorization": f"Bearer {mineru_config.api_key}",
        "Content-Type": "application/json"
    }
    url_get_upload = f"{mineru_config.base_url}/file-urls/batch"
    req_data = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": "vlm"
    }

    # Request the signed upload URL.
    response = requests.post(url_get_upload, json=req_data, headers=request_headers)
    if response.status_code != 200:
        raise RuntimeError(f"MinerU request failed. Status code: {response.status_code}, response: {response.text}")

    resp_data = response.json()
    if resp_data["code"] != 0:
        raise RuntimeError(f"MinerU API returned failure. code: {resp_data['code']}, msg: {resp_data['msg']}")

    # Extract the signed upload URL and task batch ID.
    signed_url = resp_data["data"]["file_urls"][0]
    batch_id = resp_data["data"]["batch_id"]

    # 3. Read PDF binary content.
    file_data = pdf_path_obj.read_bytes()

    # 4. Upload with a stable Session and ignore environment proxies for signed URLs.
    with requests.Session() as session:
        session.trust_env = False
        put_response = session.put(signed_url, data=file_data, timeout=60)

        if put_response.status_code != 200:
            raise RuntimeError(f"PDF upload failed. Status code: {put_response.status_code}, response: {put_response.text}")
    # 5. Poll task status with timeout control and automatic retry for server errors.
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600  # Maximum timeout: 10 minutes.
    poll_interval = 3      # Polling interval: 3 seconds.
    start_time = time.time()

    logger.debug("Started polling MinerU parsing results.")

    while True:
        # Timeout check.
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"MinerU parsing task timed out after {timeout_seconds}s. Please check the service or file size.")
        try:
            poll_response = requests.get(poll_url, headers=request_headers, timeout=10)
        except Exception as e:
            logger.warning(f"Polling request error; retrying: {str(e)}")
            time.sleep(poll_interval)
            continue
        status_code = poll_response.status_code
        # Retry server-side 5xx errors.
        if status_code != 200:
            if 500 <= status_code < 600:
                logger.warning(f"MinerU server error {status_code}; sleeping before retry.")
                time.sleep(poll_interval)
                continue
            else:
                raise RuntimeError(f"Polling request failed with client error {status_code}. Please check API_KEY and service URL.")
        # Parse the business response.
        poll_data = poll_response.json()
        if poll_data["code"] != 0:
            raise RuntimeError(f"MinerU polling API error. code: {poll_data['code']}, msg: {poll_data['msg']}")
        extract_results = poll_data["data"]["extract_result"]
        if not extract_results:
            logger.debug("No parsing result yet; continuing to poll.")
            time.sleep(poll_interval)
            continue
        # 6. Handle each task state.
        task_state = extract_results[0]["state"]
        if task_state == "done":
            full_zip_url = extract_results[0]["full_zip_url"]
            if not full_zip_url:
                raise RuntimeError("MinerU parsing completed, but no valid ZIP download URL was returned.")
            logger.info("PDF parsing task completed; preparing to download the result package.")
            return full_zip_url
        elif task_state == "failed":
            raise RuntimeError("MinerU parsing task failed. Please check whether the PDF is corrupted or contains abnormal content.")
        else:
            logger.debug(f"Task is still processing. Current state: {task_state}; continuing to poll.")
            time.sleep(poll_interval)

@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url:str, output_dir_pbj:Path, stem:str):
    """
    Step 3: Download and extract the MinerU result ZIP, then locate the target Markdown file.
    Flow: download ZIP, clean old extraction directory, extract files, locate Markdown by priority,
    and rename it to match the source PDF name.
    Args:
        zip_url: ZIP package download URL.
        output_dir_obj: Output directory Path.
        pdf_stem: PDF filename without extension.
    Returns:
        Absolute path to the final Markdown file as a string.
    Raises:
        RuntimeError for download failures and FileNotFoundError when no Markdown file exists.
    """
    # 1. Download the resource at zip_url.
    response = requests.get(zip_url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"ZIP package download failed. Status code: {response.status_code}, response: {response.text}")

    # 2. Save the ZIP file as output_dir_obj / source_filename_result.zip.
    zip_save_path =output_dir_pbj / f"{stem}_result.zip"
    zip_save_path.write_bytes(response.content)

    # 3. Clean the old extraction directory and extract the ZIP package.
    # Define the target extraction directory.
    extract_target_dir = output_dir_pbj / stem
    if extract_target_dir.exists():
        # shutil.copy(source, target)   # Copy a file.
        # shutil.move(source, target)   # Move or rename a file.
        # extract_target_dir.unlink(missing_ok=True)   # Delete a single file.
        # extract_target_dir.rmdir()                   # Delete an empty directory.
        shutil.rmtree(extract_target_dir)

    # Ensure the output directory exists.
    extract_target_dir.mkdir(parents=True, exist_ok=True)
    # Extract with ZipFile.
    with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_target_dir)
    # with ZipFile("test.zip", "w") as zip_write:
    #     zip_write.write("a.txt")   # Add a.txt to the archive.
    # shutil.unpack_archive(zip_file_path_obj, extract_dir_path_obj)
    # 4. Normalize the Markdown filename and return its string path.
    # Get all Markdown files.
    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list or len(md_file_list) == 0:
        raise FileNotFoundError("No Markdown file corresponding to the PDF was found.")

    target_md_file = None
    # Prefer a Markdown file with the same stem as the PDF.
    for md_file in md_file_list:
        if md_file.stem == stem:
            target_md_file = md_file
            break
    # Then prefer a file named full.md.
    if  not target_md_file :
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break

    if not target_md_file:
        target_md_file = md_file_list[0]

    # Rename the file when needed to simplify downstream processing.
    if target_md_file.stem != stem:
        # target_md_file.with_name(f"{stem}.md") returns a modified Path object without touching disk.
        # target_md_file.rename(...) changes the filename on disk and returns the new Path.
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))

    # Get the final Markdown absolute path and return it as a string.
    final_md_str_path = str(target_md_file.resolve())
    return final_md_str_path

if __name__ == "__main__":

    # Unit test: verify the full PDF-to-Markdown flow.
    logger.info("===== Starting node_pdf_to_md unit test =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"Resolved project root: {PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180_product_safety_manual.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # Build test state.
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== Finished node_pdf_to_md unit test =====")
