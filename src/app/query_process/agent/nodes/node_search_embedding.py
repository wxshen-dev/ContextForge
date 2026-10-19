import sys
import os
from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

def node_search_embedding(state):

    logger.info("---search_milvus 开始处理---")
    add_running_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])

    query = state.get("rewritten_query") or state.get("original_query")
    item_names = state.get("item_names")

    if not query or not item_names:
        logger.info("Skipping vector search: query or product names are missing")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"embedding_chunks": []}

    logger.info(f"核心入参提取: query='{query}', item_names={item_names}")

    logger.info(f"开始为文本获取嵌入值: {query[:50]}..." if len(query) > 50 else f"开始为“{query}”文本获取嵌入值...")

    embeddings = generate_embeddings([query])

    dense_vec = embeddings.get("dense")[0]
    sparse_vec = embeddings.get("sparse")[0]

    logger.debug(f"向量生成成功: dense_dim={len(dense_vec)}, sparse_len={len(sparse_vec)}")

    collection_name = os.environ.get("CHUNKS_COLLECTION")
    logger.info(f"正在连接到 Milvus 并准备集合 '{collection_name}'...")

    quoted = ", ".join(f'"{v}"' for v in item_names)

    expr = f"item_name in [{quoted}]"
    logger.info(f"创建搜索请求过滤表达式: {expr}")

    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        expr=expr,
        limit=10
    )

    logger.info("开始执行 Milvus 混合检索...")
    client = get_milvus_client()
    res = hybrid_search(
        client=client,
        collection_name=collection_name,
        reqs=reqs,
        ranker_weights=(0.8, 0.2),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id", "content", "item_name"]
    )

    hit_count = len(res[0]) if res and len(res) > 0 else 0
    logger.info(f"节点 search_embedding 处理成功，检索到 {hit_count} 条相关片段")
    if hit_count > 0:
        logger.debug(f"Top1 检索结果示例: {res[0][0]}")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    return {"embedding_chunks": res[0] if res else []}
