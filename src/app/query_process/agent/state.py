import copy
from typing import Any
from typing_extensions import TypedDict


class QueryGraphState(TypedDict):
    session_id: str
    original_query: str
    embedding_chunks: list[Any]
    hyde_embedding_chunks: list[Any]
    web_search_docs: list[Any]
    rrf_chunks: list[Any]
    reranked_docs: list[Any]
    prompt: str
    answer: str
    image_urls: list[str]
    item_names: list[str]
    rewritten_query: str
    history: list[Any]
    is_stream: bool


query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "image_urls": [],
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
}


def create_query_default_state(**overrides: Any) -> QueryGraphState:
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


def get_query_default_state() -> QueryGraphState:
    return copy.deepcopy(query_graph_default_state)


def copy_query_state(state: QueryGraphState, **overrides: Any) -> QueryGraphState:
    result = copy.deepcopy(state)
    result.update(overrides)
    return result
