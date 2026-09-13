import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

# Global Milvus client instance for singleton reuse.
_milvus_client = None


def get_milvus_client():
    """
    Get the Milvus client singleton.
    Reuses the client connection to avoid repeated connection overhead.
    :return: MilvusClient instance, or None when connection fails.
    """
    try:
        global _milvus_client
        # Create a new connection only when the singleton is not initialized.
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            # Validate the Milvus connection URL.
            if not milvus_uri:
                logger.error("Milvus client connection failed: missing MILVUS_URL environment variable.")
                return None
            # Initialize the Milvus client.
            _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus client connected successfully.")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus client connection error: {str(e)}", exc_info=True)
        return None


def _coerce_int64_ids(ids):
    """
    Convert chunk_id values to the INT64 type required by Milvus.
    Filters invalid IDs and separates convertible IDs from invalid ones.
    :param ids: chunk_id list to convert.
    :return: Tuple of (ok_ids, bad_ids).
    """
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        try:
            ok.append(int(x))
        except Exception:
            bad.append(x)
    return ok, bad


def fetch_chunks_by_chunk_ids(
        client,
        collection_name: str,
        chunk_ids,
        *,
        output_fields=None,
        batch_size: int = 100,
):
    """
    Batch-query chunk data in Milvus by chunk_id primary keys.
    Used to complete chunk information when only chunk_id values are available.
    Prefer primary-key get for performance, then fall back to query filtering.
    :param client: MilvusClient instance.
    :param collection_name: Collection name.
    :param chunk_ids: chunk_id list to query.
    :param output_fields: Fields to return. Defaults to core chunk fields.
    :param batch_size: Batch size to avoid overly large single queries.
    :return: List of Milvus entity dictionaries. Returns an empty list on failure.
    """
    # Return early when the client or collection name is invalid.
    if client is None:
        return []
    if not collection_name:
        return []
    # Default fields: core chunk identifiers and content fields.
    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]

    # Convert IDs to INT64 and separate valid/invalid values.
    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)
    if bad_ids:
        # Log invalid IDs and skip them.
        logger.warning(f"Found chunk_id values that cannot be converted to INT64; skipping them: {bad_ids}")

    # Return early when no valid IDs remain.
    if not ok_ids:
        return []

    results = []
    # Query valid IDs in batches.
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        # Option 1: prefer primary-key get for best performance.
        if hasattr(client, "get"):
            try:
                got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)
                if got:
                    results.extend(got)
                continue
            except Exception as e:
                logger.warning(f"Milvus get query failed; falling back to query method: {str(e)}")

        # Option 2: fall back to filter-based query.
        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            q = client.query(collection_name=collection_name, filter=expr, output_fields=output_fields)
            if q:
                results.extend(q)
        except Exception as e:
            logger.error(f"Milvus batch query by chunk_id failed: {str(e)}", exc_info=True)

    return results


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """
    Build Milvus hybrid-search request objects.
    Creates separate dense and sparse vector search requests for later fusion.
    :param dense_vector: Dense vector generated from text.
    :param sparse_vector: Sparse vector generated from text.
    :param dense_params: Dense-vector search params. Defaults to cosine similarity.
    :param sparse_params: Sparse-vector search params. Defaults to inner product.
    :param expr: Search filter expression for precise filtering.
    :param limit: Result count for each vector search. Defaults to 5.
    :return: Search request list containing [dense_req, sparse_req].
    """
    # Default dense-vector params: cosine similarity for BGE-M3 dense vectors.
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # Default sparse-vector params: inner product for BGE-M3 sparse vectors.
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # Build the dense-vector ANN search request for the dense_vector field.
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    # Build the sparse-vector search request for the sparse_vector field.
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """
    Execute Milvus dense + sparse hybrid vector search.
    Uses WeightedRanker to fuse both search result sets and improve retrieval quality.
    :param client: MilvusClient instance.
    :param collection_name: Collection name.
    :param reqs: Search request list, expected to be [dense_req, sparse_req].
    :param ranker_weights: Fusion weights for dense and sparse vectors.
    :param norm_score: Whether to normalize scores before fusion.
    :param limit: Final hybrid-search result count. Defaults to 5.
    :param output_fields: Fields to return. Defaults to item_name.
    :param search_params: Search params, such as ef/topk. Defaults to None.
    :return: Hybrid-search results, or None on failure.
    """
    try:
        # Initialize the weighted ranker for dense/sparse fusion.
        # norm_score=True normalizes scores to 0-1 before applying weights.
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        # Default returned field: document identifier.
        if output_fields is None:
            output_fields = ["item_name"]

        # Run hybrid search and rerank fused dense/sparse results by weight.
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus hybrid search completed for collection [{collection_name}]; retrieved {len(res[0])} results.")
        return res
    except Exception as e:
        logger.error(f"Milvus hybrid search failed for collection [{collection_name}]: {str(e)}", exc_info=True)
        return None
