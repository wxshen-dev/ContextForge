import sys
import json
import asyncio
from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import logger

async def mcp_call(query):
    from agents.mcp import MCPServerSse

    # ==================================================================================

    # ----------------------------------------------------------------------------------

    #

    # ==================================================================================
    search_mcp = MCPServerSse(
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": mcp_config.api_key},
            "timeout": 300,
            "sse_read_timeout": 300
        }
    )

    try:
        logger.info(f"[MCP] 正在连接百炼 WebSearch 服务: {mcp_config.mcp_base_url}")

        await search_mcp.connect()

        logger.info(f"[MCP] 连接成功，正在调用工具 'bailian_web_search' 查询: {query}")

        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 5}
        )
        logger.info("[MCP] 工具调用完成，已获取返回结果")
        return result

    except Exception as e:
        logger.error(f"[MCP] 调用过程中发生异常: {e}", exc_info=True)
        return None

    finally:

        await search_mcp.cleanup()

def node_web_search_mcp(state):

    logger.info("---node_web_search_mcp 开始处理---")

    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    if not mcp_config.mcp_base_url or not mcp_config.api_key:
        logger.info("Web search is not configured; continuing with local sources")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"web_search_docs": []}

    query = state.get("rewritten_query", "")
    if not query:

        query = state.get("original_query", "")

    docs = []

    if query:
        try:

            logger.info(f"启动异步 MCP 调用，Query: {query}")

            # ======================================================================

            # ----------------------------------------------------------------------

            #

            # result.content[0].text = """
            # {
            #   "pages": [
            #     {

            #       "url": "http://example.com/manual",

            #     },
            #     ...
            #   ]
            # }
            # """
            # ======================================================================
            result = asyncio.run(mcp_call(query))

            if result and not result.isError and result.content:

                raw_text = result.content[0].text
                try:
                    data = json.loads(raw_text)
                    pages = data.get("pages") or []

                    logger.info(f"MCP 返回原始页面数量: {len(pages)}")

                    for item in pages:
                        snippet = (item.get("snippet") or "").strip()
                        url = (item.get("url") or "").strip()
                        title = (item.get("title") or "").strip()

                        if not snippet:
                            continue

                        docs.append({"title": title, "url": url, "snippet": snippet})

                except json.JSONDecodeError:
                    logger.error(f"MCP 返回结果解析 JSON 失败: {raw_text[:100]}...")
            else:
                if result and result.isError:
                    logger.error(f"MCP 返回错误: {result}")
                else:
                    logger.warning("MCP 返回结果为空或无效")

            logger.info(f"结构化搜索结果数量: {len(docs)}")

        except Exception as e:
            logger.error(f"MCP 搜索节点执行异常: {e}", exc_info=True)
    else:
        logger.warning("查询词为空，跳过 MCP 搜索")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    logger.info("---node_web_search_mcp 处理结束---")

    if docs:
        return {"web_search_docs": docs}
    return {}
