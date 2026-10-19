import os
from typing import Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.conf.milvus_config import milvus_config


from app.import_process.agent.state import ImportGraphState

from app.clients.milvus_utils import get_milvus_client

from app.lm.lm_utils import get_llm_client

from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings

from app.utils.normalize_sparse_vector import normalize_sparse_vector

from app.utils.task_utils import add_running_task, add_done_task

from app.core.logger import logger,node_log,step_log

from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string



DEFAULT_ITEM_NAME_CHUNK_K = 5

SINGLE_CHUNK_CONTENT_MAX_LEN = 800

CONTEXT_TOTAL_MAX_CHARS = 2500



@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """Process input and return the updated state."""

    add_running_task(state['task_id'], 'node_item_name_recognition')


    chunks, file_title = step_1_get_chunks_and_file_title(state)

    context = step_2_build_context(chunks)

    item_name = step_3_call_llm(context, file_title)

    step_4_update_chunks_and_state(state, item_name, chunks)

    dense_vector, sparse_vector = step_5_generate_embeddings(item_name)

    step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)

    add_done_task(state['task_id'], 'node_item_name_recognition')
    return state

@step_log("step_1_get_chunks_and_file_title")
def step_1_get_chunks_and_file_title(state) -> Tuple[str, str]:
    """Process input and return the updated state."""
    chunks = state.get('chunks')
    file_title = state.get('file_title')

    if not chunks:
        raise ValueError("chunks没有值，无法继续进行，抛出异常处理！")
    if not file_title:


        file_title = os.path.basename(state.get('md_path'))
        state['file_title'] = file_title
    return chunks, file_title

@step_log("step_2_build_context")
def step_2_build_context(chunks) -> str:
    """Process input and return the updated state."""

    parts = []
    total_chars = 0

    for index,chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K], start=1):
        chunk_title = chunk['title']
        chunk_content = chunk['content']

        # if len(chunk_content) + total_chars > SINGLE_CHUNK_CONTENT_MAX_LEN:
        #     chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN-total_chars]
        data = f"切片：{index}，标题:{chunk_title},内容：{chunk_content}"
        parts.append(data)
        total_chars += len(data)

        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            logger.info(f"已经达到最大字符数:{total_chars}，停止拼接！")
            break

    context = "\n\n".join(parts)

    final_context = context[:SINGLE_CHUNK_CONTENT_MAX_LEN]

    return final_context

@step_log("step_3_call_llm")
def step_3_call_llm(context, file_title) -> str:
    """Process input and return the updated state."""

    human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
    system_prompt = load_prompt("product_recognition_system")
    messages = [
        HumanMessage(content=human_prompt),
        SystemMessage(content=system_prompt)
    ]

    llm = get_llm_client(json_mode=False)

    chain = llm | StrOutputParser ()
    item_name = chain.invoke(messages)


    if not item_name:
        item_name = file_title

    return item_name

@step_log("step_4_update_chunks_and_state")
def step_4_update_chunks_and_state(state, item_name, chunks):
    """
    state[item_name] = item_name
    chunks -> {item_name:item_name}
    :param state:
    :param item_name:
    :param chunks:
    :return:
    """
    state['item_name'] = item_name

    for chunk in chunks:
        chunk['item_name'] = item_name
    state['chunks'] = chunks
    logger.info(f"完成了chunks和state[item_name]的赋值和修改！！")

@step_log("step_5_generate_embeddings")
def step_5_generate_embeddings(item_name):
    """Process input and return the updated state."""
    pass
    result = generate_embeddings([item_name])
    dense_vector,sparse_vector = result['dense'][0],result['sparse'][0]
    return dense_vector,sparse_vector

@step_log("step_6_save_to_vector_db")
def step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    """Process input and return the updated state."""

    milvus_client = get_milvus_client()

    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):


        schema = milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )

        # 3.2. Add fields to schema
        # pk file_title item_name dense_vector sparse_vector
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True,auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length =65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length =65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, # Maximum number of neighbors each node can connect to in the graph
                    "efConstruction": 200 },  # or "DAAT_WAND" or "TAAT_NAIVE"
        )
        pass

        index_params.add_index(
            field_name="sparse_vector",  # Name of the vector field to be indexed
            index_type="SPARSE_INVERTED_INDEX",  # Type of the index to create
            index_name="sparse_vector_index",  # Name of the index to create
            metric_type="IP",  # Metric type used to measure similarity

            params={"inverted_index_algo": "DAAT_MAXSCORE"},  # Algorithm used for building and querying the index
        )
        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )


    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    milvus_client.delete(collection_name=milvus_config.item_name_collection,
                         filter=f"item_name=='{item_name}'")

    item = {
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector
    }
    milvus_client.insert(collection_name=milvus_config.item_name_collection,
                         data=[item])
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    logger.info(f"保存了item_name:{item_name}的数据到向量数据库中！！")



def test_node_item_name_recognition():
    """Process input and return the updated state."""
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:

        mock_state = ImportGraphState({
            "task_id": "test_task_123456",
            "file_title": "华为Mate60 Pro手机使用说明书",
            "file_name": "华为Mate60Pro说明书.pdf",

            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })


        result_state = node_item_name_recognition(mock_state)


        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")


        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)

            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)



if __name__ == "__main__":

    test_node_item_name_recognition()
