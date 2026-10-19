import sys
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

def _as_entity_list(state_list) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    for doc in (state_list or []):
        if not doc:
            continue

        final_ent = {}

        if hasattr(doc, "entity") and hasattr(doc, "id"):

            entity_content = doc.entity
            if hasattr(entity_content, "to_dict"):
                 final_ent = entity_content.to_dict()
            elif isinstance(entity_content, dict):
                 final_ent = entity_content.copy()
            else:

                 try:
                     final_ent = dict(entity_content)
                 except:
                     pass

            if "id" not in final_ent and "chunk_id" not in final_ent:
                final_ent["id"] = doc.id

            if hasattr(doc, "distance"):
                final_ent["score"] = doc.distance

        elif isinstance(doc, dict):

             if "entity" in doc:
                 ent = doc["entity"]
                 if isinstance(ent, dict):
                     final_ent = ent.copy()

                 if "id" in doc and "id" not in final_ent:
                     final_ent["id"] = doc["id"]
                 if "distance" in doc:
                     final_ent["score"] = doc["distance"]
             else:

                 final_ent = doc

        elif hasattr(doc, "get"):
             ent = doc.get("entity") or doc
             if isinstance(ent, dict):
                 final_ent = ent

        if final_ent and isinstance(final_ent, dict):
            out.append(final_ent)

    return out

def reciprocal_rank_fusion(
        source_weights: list,
        k: int = 60,
        max_results: int = None,
) -> List[tuple]:

    score_map = {}

    chunk_map = {}

    for docs, weight in source_weights:

        for rank, item in enumerate(docs, start=1):

            chunk_id = item.get("chunk_id") or item.get("id")

            if not chunk_id:

                logger.warning(
                    f"RRF Warning: item missing chunk_id/id: {list(item.keys()) if isinstance(item, dict) else item}")
                continue

            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight * (1.0 / (k + rank))

            chunk_map.setdefault(chunk_id, item)

    merged = []
    for chunk_id, score in score_map.items():
        doc_item = chunk_map[chunk_id]
        merged.append((doc_item, score))

    merged.sort(key=lambda x: x[1], reverse=True)

    if max_results is not None:
        merged = merged[:max_results]

    return merged

def node_rrf(state):

    logger.info("---RRF (倒数排名融合) 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    #  {"entity": {...fields...}, "distance": ...}

    embedding_chunks = _as_entity_list(state.get("embedding_chunks"))
    hyde_embedding_chunks = _as_entity_list(state.get("hyde_embedding_chunks"))

    logger.info(f"RRF 输入统计: Embedding源={len(embedding_chunks)}条, HyDE源={len(hyde_embedding_chunks)}条")

    if embedding_chunks:
        logger.debug(f"Embedding源 chunk_ids (前5个): {[c.get('chunk_id') for c in embedding_chunks[:5]]}")
    if hyde_embedding_chunks:
        logger.debug(f"HyDE源 chunk_ids (前5个): {[c.get('chunk_id') for c in hyde_embedding_chunks[:5]]}")

    source_weights = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0)
    ]

    rrf_res = reciprocal_rank_fusion(source_weights, k=60, max_results=10)

    rrf_chunks = [doc for doc, score in rrf_res]

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    return {"rrf_chunks": rrf_chunks}
