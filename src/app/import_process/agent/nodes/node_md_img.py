import os
import re
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO dependencies
from minio.deleteobjects import DeleteObject

# Use LangChain utilities and multimodal message support instead of native OpenAI calls.
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from langchain_core.output_parsers import StrOutputParser
# Shared LLM client helper.
from app.lm.lm_utils import get_llm_client
# LangChain multimodal message dependency.
from langchain.messages import HumanMessage
# Project configuration.
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# Project logging helpers.
from app.core.logger import logger, node_log, step_log
# API rate limiting helper.
from app.utils.rate_limit_utils import apply_api_rate_limit
# Prompt loading helper.
from app.core.load_prompt import load_prompt

# Image formats supported by the import pipeline.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    Check whether a file uses a supported image extension.
    :param filename: File name with extension.
    :return: True when the image format is supported.
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS



@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Markdown image processing (node_md_img).
    This node normalizes image references before document splitting and import.
    """
    # 1. Update task status and logs.
    add_running_task(state['task_id'],'node_md_img')
    # 2. Validate core parameters and resolve Markdown/image paths.
    md_content,md_path_obj,images_dir_obj = step_1_get_content(state)
    # 3. Locate images used in the Markdown and collect nearby context.
    image_targets = step_2_scan_images(md_content, images_dir_obj)
    # 4. Summarize image content with the multimodal model.
    image_summaries = step_3_image_summary(image_targets,md_path_obj.stem)
    # 5. Upload images to MinIO and replace local Markdown image references.
    new_md_content = step_4_upload_images_replace(image_summaries, image_targets , md_content, md_path_obj.stem)
    # 6. Save the updated Markdown as <original_name>_new.md.
    new_md_file_path_str = step_5_backup_md_file(md_path_obj, new_md_content)
    # 7. Update md_path and md_content in state.
    state['md_path'] = new_md_file_path_str
    state['md_content'] = new_md_content
    # 8. Return the updated state.
    add_done_task(state['task_id'], 'node_md_img')
    return state

@step_log("step_1_get_content")
def step_1_get_content(state) -> Tuple[str, Path, Path]:
    """
    Extract and validate Markdown content, then resolve the image directory.
    :param state: Import graph state.
    :return: Markdown content, Markdown path, and image directory path.
    """
    # 1. Get md_path.
    md_file_path = state.get("md_path")
    if not md_file_path:
        raise ValueError("Invalid md_path. Please check the input parameters.")
    md_file_obj = Path(md_file_path)
    if not md_file_obj.exists():
        raise FileNotFoundError(f"Invalid md_path. Please check the input parameters: {md_file_path}")

    # 2. Read md_content when it is not already present in state.
    if not state['md_content']:
        state['md_content'] = md_file_obj.read_text(encoding="utf-8")

    # 3. Resolve the image storage directory.
    images_dir_obj = md_file_obj.parent / "images"

    return state['md_content'], md_file_obj, images_dir_obj

@step_log("step_2_scan_images")
def step_2_scan_images(md_content: str, images_dir_obj: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    Scan Markdown image references, match local image files, and capture nearby context.
    :param md_content: Raw Markdown content.
    :param images_dir_obj: Directory containing local images.
    :return: List of (image file name, absolute image path, (text before, text after)).
    """
    # Store the final image processing targets.
    image_targets = []
    # Use pathlib to iterate through the image directory.
    for image_file in images_dir_obj.iterdir():
        img_name = image_file.name
        # Skip unsupported file formats.
        if not is_supported_image(img_name):
            logger.warning(f"Skipping non-image file: {img_name}")
            continue
        # Match Markdown image syntax: ![...](...image_name...).
        # re.escape prevents special characters in file names from breaking the regex.
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        items = list(pattern.finditer(md_content))
        # Skip files that are not referenced by the Markdown document.
        if not items:
            logger.warning(f"Image {img_name} is not referenced in Markdown. Skipping.")
            continue
        # Get the first image reference position in the Markdown document.
        start, end = items[0].span()
        # Capture nearby context, up to 100 characters before and after the image reference.
        pre_text = md_content[max(start - 100, 0): start]
        post_text = md_content[end: min(end + 100, len(md_content))]
        context = (pre_text, post_text)
        # Build the processing target: file name, full path, and surrounding context.
        image_targets.append((img_name, str(image_file), context))
    return image_targets

@step_log("step_3_generate_img_summaries")
def step_3_image_summary(image_targets, stem) -> Dict[str, str]:
    """
    Generate a concise textual summary for each image.
    :param image_targets: Image data [(file name, file path, (text before, text after))].
    :param stem: Markdown file stem used by the prompt.
    :return: Mapping from image file name to image summary.
    """
    # 1. Store generated summaries.
    summaries = {}
    # 2. Track model requests for rate limiting.
    requests_limiter = deque()

    for image_file,image_path,context in image_targets:
        # Apply API rate limiting according to the configured model limits.
        apply_api_rate_limit(requests_limiter,max_requests=100)
        # Get the multimodal model client.
        vm_model = get_llm_client(model=lm_config.lv_model)
        # Prepare the prompt.
        prompt = load_prompt(name="image_summary", root_folder=stem, image_content=context)
        # Convert the image binary data into a base64 string for multimodal input.
        if isinstance(image_path, str):
            image_path = Path(image_path)
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )

        # Invoke the model and parse the result as a string.
        chain = vm_model | StrOutputParser()
        summary = chain.invoke([message])
        summaries[image_file] = summary

    return summaries

@step_log("step_4_upload_images")
def step_4_upload_images_replace(image_summaries, image_targets, md_content, stem) -> str:
    """
    Upload images to MinIO and replace local Markdown image references.
    This makes images accessible outside the local filesystem while preserving
    the image content through remote URLs and searchable alt text.
    :param image_summaries: Mapping from image file name to generated summary.
    :param image_targets: Image data with file names, paths, and context.
    :param md_content: Original Markdown content.
    :param stem: Markdown file stem.
    :return: Updated Markdown content.
    """
    # 1. Get the MinIO client.
    minio_client = get_minio_client()
    # 2. Remove existing images for this Markdown file from MinIO.
    # Path format: MinIO / bucket / file stem / images.
    object_list = minio_client.list_objects(
        bucket_name=minio_config.bucket_name,
        # Strip the leading slash before using the directory as a MinIO prefix.
        prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",
        recursive=True
    )
    # Convert matched MinIO objects into delete requests.
    delete_object_list = [
        DeleteObject(obj.object_name)
        for obj in object_list
    ]
    # Execute the batch delete request.
    errors = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=delete_object_list
    )
    for error in errors:
        logger.warning(f"Failed to delete image from MinIO: {error}")

    # 3. Upload images to MinIO.
    # Store the generated URL for each image: {image_file: minio_url}.
    images_urls = {}
    for image_file,image_path,_ in image_targets:
        # Keep processing other images if one upload fails.
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_file}",
                file_path=image_path,
                content_type="image/jpeg"
            )
            # Build the full image URL: protocol + endpoint + bucket + object name.
            images_urls[
                image_file] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.debug(f"Uploaded image {image_file}. URL: {images_urls[image_file]}")
        except Exception as e:
            logger.exception(f"Failed to upload image {image_file}. Reason: {e}")
            logger.debug("Continuing with the next image.")
    # 4. Build replacement metadata: {image_file: (summary, url)}.
    images_infos = {}
    for image_file,summary in image_summaries.items():
        images_infos[image_file] = (summary,images_urls[image_file])

    # 5. Replace local Markdown image references with summarized remote references.
    if images_infos:
        for image_file,(summary,url) in images_infos.items():

            rep = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")

            md_content = rep.sub(lambda _: f"![{summary}]({url})", md_content)
    logger.debug(f"Completed Markdown image reference replacement. Preview: {md_content[:200]}")
    return md_content

@step_log("step_5_backup_md_file")
def step_5_backup_md_file(md_path_obj, new_md_content) -> str:
    """
    Save the updated Markdown file and return its new path.
    Naming rule: <original_name>_new.md.
    :param md_path_obj: Original Markdown path.
    :param new_md_content: Updated Markdown content.
    :return: New Markdown file path.
    """
    # new_md_path_obj = md_path_obj.with_stem(f"{md_path_obj.stem}_new")
    new_md_path_obj = md_path_obj.parent / (md_path_obj.stem + "_new" + md_path_obj.suffix)

    new_md_path_obj.write_text(new_md_content,encoding="utf-8")

    return str(new_md_path_obj)

if __name__ == "__main__":
    """Local test entry point for the Markdown image processing workflow."""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"Local test - project root: {PROJECT_ROOT}")

    # Test Markdown file path. Place the test file in the target directory manually.
    test_md_name = os.path.join(r"output\sample_product_safety_manual", "sample_product_safety_manual.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # Validate that the test file exists.
    if not os.path.exists(test_md_path):
        logger.error(f"Local test - test file does not exist: {test_md_path}")
        logger.info("Check the file path, or manually place the test Markdown file under the project output directory.")
    else:
        # Build a test state object to simulate workflow input.
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("Starting local test - Markdown image processing workflow")
        # Run the core processing workflow.
        result_state = node_md_img(test_state)
        logger.info(f"Local test completed - result state: {result_state}")
