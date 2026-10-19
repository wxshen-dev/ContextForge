
from pymilvus import DataType

from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger,node_log,step_log
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string


CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """Process input and return the updated state."""

    add_running_task(state['task_id'], "node_import_milvus")

    chunks = state.get('chunks')
    if not chunks:
        logger.error("node_import_milvus: chunks数据不存在")
        raise ValueError("node_import_milvus: chunks数据不存在")

    milvus_client = get_milvus_client()
    step_2_prepare_collection(milvus_client)

    step_3_delete_old_data(milvus_client, state['item_name'])

    with_id_chunks = step_4_insert_collections(milvus_client, chunks)
    state['chunks'] = with_id_chunks
    add_done_task(state['task_id'], 'node_import_milvus')
    return state


    chunks = state.get('chunks')
    if not chunks:
        logger.error("node_import_milvus: chunks数据不存在")
        raise ValueError("node_import_milvus: chunks数据不存在")

@step_log("step_2_prepare_collection")
def step_2_prepare_collection(milvus_client):
    """Process input and return the updated state."""

    if not milvus_client.has_collection(collection_name=milvus_config.chunks_collection):


        schema = milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )

        # 3.2. Add fields to schema
        # pk file_title item_name dense_vector sparse_vector
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 32,  # Maximum number of neighbors each node can connect to in the graph
                    "efConstruction": 300},  # or "DAAT_WAND" or "TAAT_NAIVE"
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
            collection_name=milvus_config.chunks_collection,
            schema=schema,
            index_params=index_params
        )
    return milvus_client

@step_log("step_3_delete_old_data")
def step_3_delete_old_data(milvus_client, item_name):
    """Process input and return the updated state."""
    milvus_client.delete(collection_name=CHUNKS_COLLECTION_NAME,
                         filter=f"item_name=='{item_name}'")

    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)


@step_log("step_4_insert_collections")
def step_4_insert_collections(milvus_client,chunks):
    """Process input and return the updated state."""
    insert_result = milvus_client.insert(collection_name=CHUNKS_COLLECTION_NAME, data=chunks)

    insert_count = insert_result.get("insert_count",0)
    logger.info(f"完成了数据插入，成功插入了 {insert_count} 条数据")


    ids = insert_result.get("ids",[])

    if ids and len(ids) == len(chunks):
        for index,chunk in enumerate(chunks):
            chunk['chunk_id'] = ids[index]

    return chunks

if __name__ == '__main__':


    import sys
    import os
    from dotenv import load_dotenv


    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))


    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "item_name":"测试项目_Milvus",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8}
            }
,
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus2",
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf2",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8}
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:

        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:

            result_state = node_import_milvus(test_state)


            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
