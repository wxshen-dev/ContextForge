
import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def step_1_create_hyde_doc(rewritten_query: str) -> str:

    if not rewritten_query:
        logger.error("Step 1 Error: rewritten_query 为空")
        raise ValueError("rewritten_query 不能为空")

    logger.info(f"Step 1: 开始生成假设性文档 (HyDE), Query: {rewritten_query}")

    try:
        llm = get_llm_client()

        hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
        logger.debug(f"Step 1: Prompt加载成功, 长度: {len(hyde_prompt)}")

        response = llm.invoke(hyde_prompt)
        hyde_doc = response.content

        logger.info(f"Step 1: 假设文档生成完成, 长度: {len(hyde_doc)} 字符")
        logger.debug(f"Step 1: 文档预览: {hyde_doc[:50]}...")

        return hyde_doc

    except Exception as e:
        logger.error(f"Step 1: 生成假设文档失败: {e}")
        raise e

def step_2_search_embedding_hyde(
    rewritten_query: str,
    hyde_doc: str,
    item_names=None,
    req_limit: int = 10,
    top_k: int = 5,
    ranker_weights=(0.8, 0.2),
    norm_score: bool = True,
    output_fields=["chunk_id", "content", "item_name"],
):

    if not rewritten_query:
        raise ValueError("rewritten_query 不能为空")
    if not hyde_doc:
        raise ValueError("hypothetical_doc 不能为空")

    combined_text = rewritten_query + " " + hyde_doc
    logger.info(f"Step 2: 拼接 Query + HyDE Doc, 总长度: {len(combined_text)}")

    logger.info("Step 2: 正在生成混合向量 (Embedding)...")
    embeddings = generate_embeddings([combined_text])

    collection_name = os.environ.get("CHUNKS_COLLECTION")
    if not collection_name:
        logger.error("Step 2 Error: 环境变量 CHUNKS_COLLECTION 未设置")
        return []

    logger.info(f"Step 2: 准备在集合 '{collection_name}' 中执行混合检索")

    expr = None
    if item_names:

        quoted = ", ".join(f'"{v}"' for v in item_names)
        expr = f"item_name in [{quoted}]"
        logger.info(f"Step 2: 应用过滤条件: {expr}")
    else:
        logger.info("Step 2: 未指定商品名过滤，将全库检索")

    try:

        reqs = create_hybrid_search_requests(
            dense_vector=embeddings.get("dense")[0],
            sparse_vector=embeddings.get("sparse")[0],
            expr=expr,
            limit=req_limit,
        )

        client = get_milvus_client()
        if not client:
            logger.error("Step 2 Error: 无法连接到 Milvus")
            return []

        logger.info(f"Step 2: 执行 Hybrid Search, Weights={ranker_weights}, TopK={top_k}")
        res = hybrid_search(
            client=client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=top_k,
            output_fields=list(output_fields),
        )

        hit_count = len(res[0]) if res and len(res) > 0 else 0
        logger.info(f"Step 2: 检索完成, 找到 {hit_count} 个匹配切片")

        return res

    except Exception as e:
        logger.error(f"Step 2: 检索过程发生异常: {e}")
        return []

def node_search_embedding_hyde(state):

    logger.info("---HyDE (假设文档检索) 节点开始处理---")

    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        rewritten_query = state.get("original_query")

    if not rewritten_query:
        logger.error("HyDE节点错误: 未找到有效的用户查询 (rewritten_query/original_query 均为空)")
        return {}

    item_names = state.get("item_names")
    logger.info(f"HyDE检索入参: query='{rewritten_query}', item_names={item_names}")

    hyde_doc = ""
    try:
        logger.info("Step 1: 开始生成假设性文档 (HyDE Doc)...")
        hyde_doc = step_1_create_hyde_doc(rewritten_query)
        logger.info(f"Step 1: 假设文档生成成功 (长度: {len(hyde_doc)})")
        logger.debug(f"假设文档预览: {hyde_doc[:100]}...")
    except Exception as e:
        logger.error(f"Step 1 (生成假设文档) 发生异常: {e}", exc_info=True)

        return {}

    try:
        logger.info("Step 2: 基于假设文档执行 Milvus 混合检索...")
        res = step_2_search_embedding_hyde(
            rewritten_query=rewritten_query,
            hyde_doc=hyde_doc,
            item_names=item_names,
            top_k=5,
        )

        hit_count = len(res[0]) if res and len(res) > 0 else 0
        logger.info(f"Step 2: 检索完成，召回 {hit_count} 条相关切片")

        if hit_count > 0:

            first_hit = res[0][0]
            score = first_hit.get("distance")
            content_preview = first_hit.get("entity", {}).get("content", "")[:30]
            logger.debug(f"Top1 结果: Score={score}, Content='{content_preview}...'")

        return {
            "hyde_embedding_chunks": res[0] if res else [],
            "hyde_doc": hyde_doc,
        }
    except Exception as e:
        logger.error(f"Step 2 (向量生成与检索) 发生异常: {e}", exc_info=True)
        return {}
    finally:

        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        logger.info("---HyDE 节点处理结束---")
