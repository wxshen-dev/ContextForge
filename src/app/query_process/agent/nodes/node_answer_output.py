import sys
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re

_IMAGE_BLOCK_MARKER = "[Images]"
MAX_CONTEXT_CHARS = 12000

def step_1_check_answer(state) -> bool:

  answer = state.get("answer", None)
  is_stream = state.get("is_stream" )
  if answer:
    if is_stream:
      logger.info("---Step 1: 发现已有答案，执行流式推送---")
      push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
    else:
      set_task_result(state["session_id"], "answer", answer)
    return True
  else:
    return False

# http://local-server/images/panel_view.jpg
# http://local-server/images/button_detail.jpg
def step_2_construct_prompt(state: QueryGraphState) -> str:

  original_query = state.get("original_query", "")
  rewritten_query = state.get("rewritten_query", "")

  question = rewritten_query if rewritten_query else original_query
  history = state.get("history", [])
  item_names = state.get("item_names", [])
  reranked_docs = state.get("reranked_docs") or []

  # ---------------------------------------------------------

  # ---------------------------------------------------------
  docs = []
  used = 0
  for i, doc in enumerate(reranked_docs, start=1):
    text = (doc.get("text") or "").strip()
    if not text:
      continue
    source = doc.get("source") or ""
    chunk_id = doc.get("chunk_id")
    url = (doc.get("url") or "").strip()
    title = (doc.get("title") or "").strip()
    score = doc.get("score")

    meta_parts = [f"[{i}]"]
    if source:
      meta_parts.append(f"[{source}]")
    if chunk_id:
      meta_parts.append(f"[chunk_id={chunk_id}]")
    if url:
      meta_parts.append(f"[url={url}]")
    if score is not None:

      meta_parts.append(f"[score={float(score):.4f}]")
    if title:
      meta_parts.append(f"[title={title}]")
    doc = " ".join(meta_parts) + "\n" + text
    if used + len(doc) > MAX_CONTEXT_CHARS:
      break
    docs.append(doc)

    used += len(doc) + 2
  context_str = "\n\n".join(docs) if docs else "No reference material"

  # ---------------------------------------------------------

  # ---------------------------------------------------------
  history_str = ""
  if history:
    for msg in history:

      role = msg.get("role")
      text = msg.get("text")
      if role == "user" and text:
        history_str += f"User: {text}\n"
      elif role == "assistant" and text:
        history_str += f"Assistant: {text}\n"

      used += len(history_str) + 2
      if used > MAX_CONTEXT_CHARS:
        break
  else:
    history_str = "No conversation history"

  item_names_str = ", ".join(item_names) if item_names else "No specified product"

  prompt = load_prompt("answer_out",
    context=context_str,
    history=history_str,
    item_names=item_names_str,
    question=question
  )

  logger.info(f"组装后的提示词为：{prompt}")

  return prompt

def step_3_generate_response(state: QueryGraphState, prompt: str) -> QueryGraphState:

  logger.info("---Step 3: 开始生成回答 (LLM Generation)---")
  logger.debug(f"最终Prompt内容: {prompt}")

  llm = get_llm_client()

  session_id = state.get("session_id")
  is_stream = state.get("is_stream")

  if is_stream:
    logger.info(f"模式: 流式输出 (Streaming), Session: {session_id}")
    final_text = ""
    try:

      for chunk in llm.stream(prompt):
        delta = getattr(chunk, "content", "") or ""
        if delta:
          final_text += delta

          push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})

      logger.info(f"流式输出完成，总长度: {len(final_text)}")

    except Exception as e:
      logger.error(f"流式生成出错: {e}", exc_info=True)

      push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})

    state["answer"] = final_text
  else:

    logger.info(f"模式: 非流式输出 (Blocking), Session: {session_id}")
    try:
      response = llm.invoke(prompt)
      content = response.content
      state["answer"] = content
      set_task_result(session_id, "answer", content)
      logger.info(f"生成回答完成，长度: {len(content)}")
    except Exception as e:
      logger.error(f"生成回答出错: {e}", exc_info=True)
      state["answer"] = "I could not generate an answer. Please try again."

  return state

def _extract_images_from_docs(docs):

    images = []
    seen = set()
    if not docs:
        return []
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

    logger.info(f"开始提取图片，待处理文档数: {len(docs)}")

    for i, doc in enumerate(docs):

        url = (doc.get("url") or "").strip()
        if url:

            if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                if url not in seen:
                    logger.debug(f"文档[{i}] 发现图片 URL (字段): {url}")
                    seen.add(url)
                    images.append(url)

        text = (doc.get("text") or "").strip()
        if text:

            matches = md_img_pattern.findall(text)
            for img_url in matches:
                img_url = img_url.strip()
                if img_url and img_url not in seen:
                    logger.debug(f"文档[{i}] 正文发现 Markdown 图片: {img_url}")
                    seen.add(img_url)
                    images.append(img_url)

    logger.info(f"图片提取完成，共找到 {len(images)} 张唯一图片: {images}")
    return images

def step_4_write_history(state: QueryGraphState, image_urls = None) -> QueryGraphState:

  session_id = state.get("session_id", "default")
  answer = (state.get("answer") or "").strip()
  item_names = state.get("item_names") or []

  try:
    if answer:
       save_chat_message(
        session_id=session_id,
        role="assistant",
        text=answer,
        rewritten_query="",
        item_names=item_names,
        image_urls=image_urls,
        message_id=None
      )
  except Exception as e:

    logger.error(f"写入Mongo历史记录失败: {e}")

  return state

def node_answer_output(state: QueryGraphState) -> QueryGraphState:

  logger.info("---node_answer_output (答案生成) 节点开始处理---")
  add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

  answer_exists = step_1_check_answer(state)

  if not answer_exists:
    prompt = step_2_construct_prompt(state)
    state["prompt"] = prompt

    step_3_generate_response(state, prompt)

  image_urls = _extract_images_from_docs(state.get("reranked_docs") or [])
  state["image_urls"] = image_urls

  if state.get("answer"):
    logger.info("---写入MongoDB历史记录---")
    step_4_write_history(state, image_urls=image_urls)

  add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

  logger.info(f"---发送 final 事件---图片为：{image_urls}")
  if state.get("is_stream"):
    push_to_session(
        state['session_id'],
        SSEEvent.FINAL,
        {
            "answer": state["answer"],
            "status": "completed",
            "image_urls": image_urls
        }
    )

  logger.info("---node_answer_output 节点处理结束---")
  return state
