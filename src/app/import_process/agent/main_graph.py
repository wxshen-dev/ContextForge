# Load environment variables from .env, such as Milvus URL, KG service URL, and BGE model path.
from dotenv import load_dotenv
# Import LangGraph core dependencies: StateGraph and built-in START/END constants.
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
# Import the custom state class that stores all data shared or modified by workflow nodes.
from app.import_process.agent.state import ImportGraphState, create_default_state
# Import custom business nodes for each knowledge-base import step.
from app.import_process.agent.nodes.node_entry import node_entry  # Entry node: initialize parameters and validate input.
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md  # PDF to Markdown conversion.
from app.import_process.agent.nodes.node_md_img import node_md_img  # Markdown image processing.
from app.import_process.agent.nodes.node_document_split import node_document_split  # Split long documents into chunks.
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  # Identify core item names.
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # Convert text chunks into BGE vectors.
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # Write vector data into Milvus.


# Initialize environment variables before reading configuration.
load_dotenv()

# ===================== 1. Initialize the LangGraph State Graph =====================
# StateGraph is the LangGraph core class for building stateful workflows.
# ImportGraphState is a custom TypedDict that defines all workflow state fields.
# Each node receives this state object, and returned key-value pairs are merged back into it.
workflow = StateGraph(ImportGraphState)

# ===================== 2. Register Business Nodes =====================
# Syntax: add_node("unique_node_id", node_function).
# Node functions must receive the state object and return a dict used to update state.
# Nodes are registered in knowledge-base import order for easier maintenance.
workflow.add_node("node_entry", node_entry)  # Flow entry: parameter initialization and input validation.
workflow.add_node("node_pdf_to_md", node_pdf_to_md)  # PDF to Markdown preprocessing.
workflow.add_node("node_md_img", node_md_img)  # Ensure images in Markdown are accessible.
workflow.add_node("node_document_split", node_document_split)  # Split large text for embedding and reasoning.
workflow.add_node("node_item_name_recognition", node_item_name_recognition)  # Extract the core business identifier.
workflow.add_node("node_bge_embedding", node_bge_embedding)  # Convert text to vectors for Milvus storage.
workflow.add_node("node_import_milvus", node_import_milvus)  # Persist vector data into Milvus.

# ===================== 3. Set the Workflow Entry Node =====================
# set_entry_point("node_id") directly specifies the first node in the flow.
# Equivalent to workflow.add_edge(START, "node_entry"), but more concise.
workflow.set_entry_point("node_entry")

# ===================== 4. Define Conditional Routing After Entry =====================
# Dynamically choose the next path from state flags for PDF import or direct Markdown import.
# The function receives the state object and returns a target node ID or END.
def route_after_entry(state: ImportGraphState) -> str:
    """
    Conditional routing logic after the entry node.
    :param state: Full workflow state containing configuration and intermediate results.
    :return: Target node ID or END. LangGraph will jump to the returned node.
    """
    # Branch 1: direct Markdown import; skip PDF conversion and process Markdown images.
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    # Branch 2: PDF import; convert PDF to Markdown before continuing.
    elif state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    # Branch 3: no import path enabled; terminate the workflow.
    else:
        return END

# Register conditional edges by binding the entry node to the routing function.
# After the source node finishes, LangGraph calls the router and jumps to the returned target.
workflow.add_conditional_edges(
    "node_entry",
    route_after_entry,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END
    }
)

# ===================== 5. Register Static Sequential Edges =====================
# All branches merge into a fixed sequence from Markdown image processing through Milvus import.
# add_edge("source_node_id", "target_node_id_or_END") defines a static route without branching.
workflow.add_edge("node_pdf_to_md", "node_md_img")  # PDF conversion complete -> Markdown image processing.
workflow.add_edge("node_md_img", "node_document_split")  # Markdown processing complete -> document splitting.
workflow.add_edge("node_document_split", "node_item_name_recognition")  # Splitting complete -> item-name recognition.
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")  # Item-name recognition complete -> BGE embedding.
workflow.add_edge("node_bge_embedding", "node_import_milvus")  # Embedding complete -> import into Milvus.
workflow.add_edge("node_import_milvus", END)  # Milvus import complete -> workflow ends.

# ===================== 6. Compile the Workflow =====================
# compile() turns the StateGraph flow into an executable LangGraph application.
# kb_import_app can be invoked repeatedly with different initial states.
kb_import_app = workflow.compile()


if __name__ == "__main__":
    from app.utils.path_util import PROJECT_ROOT
    import os

    # 全流程测试：验证PDF导入→Milvus入库→KG导入完整链路
    logger.info("===== 开始执行知识图谱导入全流程测试 =====")
    # 1. 构造测试文件路径（复用你项目的doc目录，和pdf2md测试文件一致）
    test_pdf_name = os.path.join("doc", "万用表RS-12的使用.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)
    # 2. 构造输出目录（存放MD/图片等中间文件）
    test_output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(test_output_dir, exist_ok=True)  # 不存在则创建

    # 3. 校验测试PDF文件是否存在
    if not os.path.exists(test_pdf_path):
        logger.error(f"全流程测试失败：测试PDF文件不存在，路径：{test_pdf_path}")
        logger.info("请检查文件路径，或手动将测试文件放入项目根目录的doc文件夹中")
    else:
        # 4. 构造测试状态（贴合实际业务入参，开启PDF解析开关）
        test_state = ImportGraphState({
            "task_id": "test_kg_import_workflow_001",  # 测试任务ID
            "user_id": "test_user",  # 测试用户ID
            "local_file_path": test_pdf_path,  # 测试PDF文件路径
            "local_dir": test_output_dir,  # 中间文件输出目录
            "is_pdf_read_enabled": False,  # 开启PDF解析（核心开关）
            "is_md_read_enabled": False  # 关闭MD解析
        })
        try:
            logger.info(f"测试任务启动，PDF文件路径：{test_pdf_path}")
            logger.info(f"中间文件输出目录：{test_output_dir}")
            logger.info("开始执行全流程节点，依次执行：entry→pdf2md→md_img→split→item_name→embedding→milvus→kg")

            # 5. 执行LangGraph全流程（流式执行，打印节点执行进度）
            final_state = None
            for step in kb_import_app.stream(test_state, stream_mode="values"):
                # 打印当前执行完成的节点（流式输出更直观）
                current_node = list(step.keys())[-1] if step else "未知节点"
                logger.info(f"✅ 节点执行完成：{current_node}")
                final_state = step  # 保存最终状态

            # 6. 全流程执行完成，结果预览和核心指标打印
            if final_state:
                logger.info("-" * 80)
                logger.info("===== 全流程测试执行成功，核心结果预览 =====")
                # 提取核心结果指标
                chunks = final_state.get("chunks", [])
                chunk_count = len(chunks)
                md_content = final_state.get("md_content", "")[:150]  # MD内容前150字符
                has_embedding = all("dense_vector" in c and "sparse_vector" in c for c in chunks) if chunks else False
                has_chunk_id = all("chunk_id" in c for c in chunks) if chunks else False
                kg_id = final_state.get("kg_id", "未生成")  # KG导入生成的ID（按实际业务字段调整）

                # 打印核心指标
                logger.info(f"📄 PDF转MD内容预览（前150字符）：{md_content}...")
                logger.info(f"📝 文档切分总切片数：{chunk_count}")
                logger.info(f"🔍 所有切片是否完成向量化：{'是' if has_embedding else '否'}")
                logger.info(f"🗄️  所有切片是否完成Milvus入库（含chunk_id）：{'是' if has_chunk_id else '否'}")
                logger.info(f"🧠 知识图谱导入ID：{kg_id}")
                logger.info(f"📂 最终状态包含的核心键：{list(final_state.keys())}")
                logger.info("-" * 80)
        except Exception as e:
            logger.exception(f"===== 全流程测试运行失败 =====")
    logger.info("===== 知识图谱导入全流程测试结束 =====")
