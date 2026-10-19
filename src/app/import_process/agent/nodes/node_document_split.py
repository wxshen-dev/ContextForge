import json
import os
import re
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task



CHUNK_SIZE = 200

CHUNK_OVERLAP = 20

@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """Process input and return the updated state."""

    add_running_task(state['task_id'],'node_document_split')

    md_content, file_title = step_1_get_content(state)


    sections,title_count,lines_count = step_2_split_by_title(md_content,file_title)


    final_chunks = step_3_refine_chunks(sections)

    state['chunks'] = final_chunks
    step_4_backup_chunks(final_chunks,state)
    add_done_task(state['task_id'], 'node_document_split')
    return state

@step_log("step_1_get_content")
def step_1_get_content(state) -> Tuple[str, str]:
    """Process input and return the updated state."""

    md_content = state['md_content']
    if not md_content:
        logger.error(f"没有输入内容,请检查输入内容是否正确!")
        raise RuntimeError("没有输入内容,请检查输入内容是否正确!")

    pass
    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')
    file_title = state.get("file_title", "default_file")
    return md_content, file_title


@step_log("step_2_split_by_title")
def step_2_split_by_title(md_content, file_title) -> List[Dict]:
    """Process input and return the updated state."""





    title_pattern = re.compile(r'^\s*#{1,6}\s+.+')
    lines = md_content.split('\n')

    chunks = []
    current_title = ""
    current_lines = []
    in_code_block = False
    title_count = 0


    for line in lines:
        strip_line = line.strip()

        if strip_line.startswith("```") or strip_line.startswith("~~~"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        if not in_code_block and title_pattern.match(strip_line):

            if current_title:

                chunks.append({
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "file_title": file_title
                })

            current_title = strip_line
            current_lines = [strip_line]
            title_count += 1
        else:

            current_lines.append(strip_line)

    if current_title:
        chunks.append({
            "title": current_title.strip(),
            "content": "\n".join(current_lines),
            "file_title": file_title
        })

    # --------------------

    # --------------------
    if not chunks:
        chunks = [{
            "title": "无主题",
            "content": md_content,
            "file_title": file_title
        }]

    pass
    return chunks, title_count, len(lines)

@step_log("step_3_refine_chunks")
def step_3_refine_chunks(sections) -> List[Dict]:
    """Process input and return the updated state."""
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,

        separators=["\n\n", "\n", "。", "！", "；", " "]
    )

    final_chunks = []
    for section in sections:

        sub_chunks = spliter.split_text(section["content"])
        has_multiple_chunks = len(sub_chunks) > 1

        for idx,chunk in enumerate(sub_chunks,start=1):
            current_title = f"{section['title']}_{idx}" if has_multiple_chunks else section["title"]
            final_chunks.append({
                "title": current_title,
                "content": chunk.strip(),
                "file_title": section["file_title"],
                "parent_title": section["title"],
                "part": idx
            })
    return final_chunks

def step_4_backup_chunks(final_chunks, state):
    """Process input and return the updated state."""
    backup_file_path = Path(state['md_path']).parent / "backup_chunks.json"

    with open(backup_file_path, "w", encoding="utf-8") as f:
        json.dump(
            final_chunks,
            f,
            ensure_ascii=False,
            indent=4
        )
    logger.debug(f"数据备份成功,备份地址:{backup_file_path}")

if __name__ == '__main__':
    pass

    pass
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")


    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)


    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:

        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")

        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")
