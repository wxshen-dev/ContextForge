import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv, find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())

def step_3_extract_info(query: str, history: List[Dict]) -> Dict:

    logger.info("Step 3: 开始提取信息 (LLM)")

    client = get_llm_client(json_mode=True)

    history_text = ""
    for msg in history:
        history_text += f"{msg.get('role', 'unknown')}: {msg.get('text', '')}\n"

    logger.info(f"Step 3: 历史上下文构建完成，长度: {len(history_text)} 字符")

    try:

        prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)
        logger.debug(f"Step 3: 提示词加载成功，Prompt长度: {len(prompt)}")
    except Exception as e:
        logger.error(f"Step 3: 加载提示词失败: {e}")
        return {"item_names": [], "rewritten_query": query}

    messages = [
        SystemMessage(content="You are a customer support assistant. Identify the user's intent and extract relevant details."),
        HumanMessage(content=prompt)
    ]

    try:
        logger.info("Step 3: 正在调用 LLM 进行提取...")
        response = client.invoke(messages)
        content = response.content
        logger.debug(f"Step 3: LLM 原始响应: {content}")

        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")

        result = json.loads(content)

        if "item_names" not in result:
            result["item_names"] = []
        if "rewritten_query" not in result:
            result["rewritten_query"] = query

        logger.info(f"Step 3: 提取结果解析成功 - 商品名: {result['item_names']}, 重写问题: {result['rewritten_query']}")
        return result

    except Exception as e:
        logger.error(f"Step 3: LLM 提取或解析失败: {e}")
        return {"item_names": [], "rewritten_query": query}

def step_4_vectorize_and_query(item_names: List[str]) -> List[Dict]:

    logger.info(f"Step 4: 开始向量化检索，目标商品: {item_names}")
    results = []

    client = get_milvus_client()
    if not client:
        logger.error("Step 4: 无法连接到 Milvus")
        return results

    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
    if not collection_name:
        logger.error("Step 4: 环境变量中未找到 ITEM_NAME_COLLECTION")
        return results

    try:
        logger.info("Step 4: 正在生成 Embedding (Dense + Sparse)...")
        embeddings = generate_embeddings(item_names)
        logger.info(f"Step 4: 向量生成完成，开始 Milvus 搜索 (Collection: {collection_name})")

        for i, name in enumerate(item_names):
            try:
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]

                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5
                )

                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2),
                    limit=5,
                    norm_score=True,
                    output_fields=["item_name"]
                )

                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        entity = hit.get("entity") or {}
                        item_name = entity.get("item_name")
                        score = hit.get("distance")

                        if item_name:
                            matches.append({
                                "item_name": item_name,
                                "score": score
                            })
                            logger.debug(f"Step 4: '{name}' 匹配项: {item_name} (Score: {score:.4f})")

                results.append({
                    "extracted_name": name,
                    "matches": matches
                })
                logger.info(f"Step 4: 商品 '{name}' 检索完成，找到 {len(matches)} 个匹配项")

            except Exception as inner_e:
                logger.error(f"Step 4: 处理商品 '{name}' 时出错: {inner_e}")
                results.append({"extracted_name": name, "matches": []})

    except Exception as e:
        logger.error(f"Step 4: 向量化或搜索过程发生全局错误: {e}")

    return results

def step_5_align_item_names(query_results: List[Dict]) -> Dict:

    logger.info("Step 5: 开始对齐商品名 (Score Analysis)")

    confirmed_item_names = []
    options = []

    for res in query_results:
        extracted_name = res.get("extracted_name", "").strip()
        matches = res.get("matches", []) or []

        if not matches:
            logger.info(f"Step 5: '{extracted_name}' 无匹配结果")
            continue

        matches.sort(key=lambda x: x.get("score", 0), reverse=True)

        top_matches_log = ", ".join([f"{m['item_name']}({m['score']:.3f})" for m in matches[:3]])
        logger.info(f"Step 5: '{extracted_name}' Top匹配: {top_matches_log}")

        high = [m for m in matches if m.get("score", 0) > 0.85]
        mid = [m for m in matches if m.get("score", 0) >= 0.6]

        if len(high) == 1:
            confirmed_name = high[0].get("item_name")
            confirmed_item_names.append(confirmed_name)
            logger.info(f"Step 5: 规则A命中 (Single High) -> 确认: {confirmed_name}")
            continue

        if len(high) > 1:
            picked = None

            if extracted_name:
                for m in high:
                    if m.get("item_name") == extracted_name:
                        picked = m
                        logger.info(f"Step 5: 规则B命中 (Exact Match in High) -> 确认: {picked.get('item_name')}")
                        break

            if not picked:
                picked = high[0]
                logger.info(f"Step 5: 规则B命中 (Highest Score) -> 确认: {picked.get('item_name')}")

            confirmed_item_names.append(picked.get("item_name"))
            continue

        if len(mid) > 0:
            current_options = [m.get("item_name") for m in mid[:5]]
            options.extend(current_options)
            logger.info(f"Step 5: 规则C命中 (Mid Confidence) -> 添加候选: {current_options}")
            continue

        logger.info(f"Step 5: 规则D命中 (Low Confidence) -> 无匹配")

    result = {
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options": list(set(options))
    }
    logger.info(f"Step 5: 对齐结果: {result}")
    return result

def step_6_check_confirmation(state: Dict, align_result: Dict, session_id: str, history: List[Dict], rewritten_query: str) -> Dict:

    logger.info("Step 6: 检查确认状态并更新 State")

    if align_result is None:
        align_result = {}

    confirmed = align_result.get("confirmed_item_names", [])
    options = align_result.get("options", [])

    if confirmed:
        logger.info(f"Step 6: [分支A] 存在确认商品名: {confirmed}")

        ids_to_update = []
        for msg in history:
            if not msg.get("item_names"):
                mid = msg.get("_id")
                if mid:
                    ids_to_update.append(str(mid))

        if ids_to_update:
            logger.info(f"Step 6: 更新 {len(ids_to_update)} 条历史消息的关联商品名")
            update_message_item_names(ids_to_update, confirmed)

        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query
        if "answer" in state:
            del state["answer"]
        return state

    if options:
        logger.info(f"Step 6: [分支B] 存在候选商品名: {options}")
        options_str = ", ".join(options[:3])
        answer = f"Which product do you mean: {options_str}? Please specify the model."
        state["answer"] = answer
        state["item_names"] = []
        return state

    logger.info("Step 6: [分支C] 无确认也无候选")
    state["answer"] = "I could not find a matching product. Please provide the exact model."
    state["item_names"] = []
    return state

def step_7_write_history(state: Dict, session_id: str, history: List[Dict], rewritten_query: str, message_id: str) -> Dict:

    logger.info("Step 7: 写入会话历史")

    logger.info(f"Step 7: 更新用户消息 (ID: {message_id})")
    save_chat_message(
        session_id=session_id,
        role="user",
        text=state["original_query"],
        rewritten_query=rewritten_query,
        item_names=state.get("item_names", []),
        message_id=message_id
    )

    return state

def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:

    logger.info(">>> node_item_name_confirm: 开始处理")

    session_id = state["session_id"]
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, "node_item_name_confirm", is_stream)

    history = get_recent_messages(session_id, limit=10)
    logger.info(f"Node: 获取到 {len(history)} 条历史消息")

    message_id = save_chat_message(session_id, "user", original_query, "", state.get("item_names", []))
    logger.debug(f"Node: 用户消息已初始保存, ID: {message_id}")

    extract_res = step_3_extract_info(original_query, history)
    item_names = extract_res.get("item_names", [])
    rewritten_query = extract_res.get("rewritten_query", original_query)

    state["rewritten_query"] = rewritten_query

    align_result = {}

    if len(item_names) > 0:
        query_results = step_4_vectorize_and_query(item_names)
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info("Node: 未提取到商品名，跳过向量检索")

    state = step_6_check_confirmation(state, align_result, session_id, history, rewritten_query)

    final_state = step_7_write_history(state, session_id, history, rewritten_query, message_id)

    final_state["history"] = history

    add_done_task(session_id, "node_item_name_confirm", is_stream)

    logger.info(f"Node: 处理结束, Final State Item Names: {final_state.get('item_names')}")
    return final_state
