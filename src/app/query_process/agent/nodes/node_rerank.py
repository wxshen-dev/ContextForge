from app.utils.task_utils import *
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger
import sys

# -----------------------------

# -----------------------------

RERANK_MAX_TOPK: int = 10

RERANK_MIN_TOPK: int = 1

RERANK_GAP_RATIO: float = 0.25

RERANK_GAP_ABS: float = 0.5

def step_1_merge_docs(state):

    rrf_docs = state.get("rrf_chunks") or []
    web_docs = state.get("web_search_docs") or []

    logger.info(f"Step 1: 开始合并文档 - 本地RRF源: {len(rrf_docs)}条, 联网Web源: {len(web_docs)}条")
    doc_items = []
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    for i, doc in enumerate(rrf_docs):

        entity = doc.get("entity") if isinstance(doc, dict) and "entity" in doc else doc

        if not isinstance(entity, dict):
            logger.warning(f"本地文档格式异常 (index={i}): {type(entity)}")
            continue

        content = entity.get("content")
        if not content:

            logger.debug(f"跳过无内容文档 (index={i}, keys={list(entity.keys())})")
            continue

        doc_id = entity.get("chunk_id") or entity.get("id")
        title = entity.get("title") or entity.get("item_name") or ""

        doc_items.append({
            "text": content,
            "doc_id": doc_id,
            "chunk_id": doc_id,
            "title": title,
            "url": "",
            "source": "local",
        })

    # ---------------------------------------------------------

    # ---------------------------------------------------------
    for i, doc in enumerate(web_docs):

        text = (doc.get("snippet") or doc.get("content") or "").strip()
        url = (doc.get("url") or "").strip()
        title = (doc.get("title") or "").strip()

        if not text:
            logger.debug(f"跳过无内容联网结果 (index={i})")
            continue

        doc_items.append({
            "text": text,
            "doc_id": None,
            "chunk_id": None,
            "title": title,
            "url": url,
            "source": "web",
        })

    logger.info(f"Step 1: 文档合并完成，共输出 {len(doc_items)} 条标准化文档")
    return doc_items

def step_2_rerank_docs(state, doc_items):

    question = state.get("rewritten_query") or state.get("original_query") or ""

    if not doc_items or not question:
        logger.warning("Step 2: 跳过重排序 (无文档或无问题)")
        return []

    logger.info(f"Step 2: 开始重排序 (Rerank), 待排序文档数: {len(doc_items)}")

    texts = [x["text"] for x in doc_items]
    try:
        reranker = get_reranker_model()

        sentence_pairs = [[question, t] for t in texts]

        logger.info("Step 2: 正在计算相关性得分...")
        scores = reranker.compute_score(sentence_pairs)

        scored_docs = []
        for item, text, score in zip(doc_items, texts, scores):

            score_val = float(score)
            scored_docs.append(
                {
                    "text": text,
                    "score": score_val,
                    "source": item.get("source") or "",
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "url": item.get("url") or "",
                    "title": item.get("title") or "",
                }
            )

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs
    except Exception as e:
        logger.error(f"Step 2: 重排序过程发生异常: {e}", exc_info=True)

        fallback_docs = [
            {
                "text": x.get("text"),
                "score": 0.0,
                "source": x.get("source") or "",
                "chunk_id": x.get("chunk_id"),
                "doc_id": x.get("doc_id"),
                "url": x.get("url") or "",
                "title": x.get("title") or "",
            }
            for x in doc_items
        ]

        return fallback_docs

def step_3_topk(scored_docs):

    max_topk = min(RERANK_MAX_TOPK, len(scored_docs))
    min_topk = RERANK_MIN_TOPK
    gap_ratio = RERANK_GAP_RATIO
    gap_abs = RERANK_GAP_ABS

    topk = max_topk

    if topk > min_topk:

        for i in range(min_topk - 1, max_topk - 1):
            s1 = scored_docs[i].get("score")
            s2 = scored_docs[i + 1].get("score")

            gap = s1 - s2

            rel = gap / (abs(s1) + 1e-6)

            if gap >= gap_abs or rel >= gap_ratio:
                logger.info(f"Step 3: 触发断崖截断 @ index={i} (Score {s1:.4f} -> {s2:.4f}, Gap={gap:.4f})")
                topk = i + 1
                break

    topk_docs = scored_docs[:topk]

    logger.info(f"Step 3: 截断完成，保留前 {len(topk_docs)} 条文档 (TopK={topk})")

    if topk_docs:
        preview = ", ".join([f"{d.get('chunk_id') or 'Web'}({d.get('score'):.3f})" for d in topk_docs[:3]])
        logger.debug(f"Step 3: Top3 文档预览: {preview}")

    return topk_docs

def node_rerank(state):

  logger.info("---Rerank (重排序) 节点开始处理---")
  add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

  doc_items = step_1_merge_docs(state)

  scored_docs = step_2_rerank_docs(state, doc_items)

  topk_docs = step_3_topk(scored_docs)

  logger.info(f"Rerank 节点处理结束, 最终输出 {len(topk_docs)} 条文档")

  add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
  return {"reranked_docs": topk_docs}
