import os

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# Model singleton to avoid repeated initialization.
_bge_m3_ef = None

def get_bge_m3_ef():
    """
    Get the BGE-M3 model singleton and load configuration automatically.
    :return: Initialized BGEM3EmbeddingFunction instance.
    """
    global _bge_m3_ef
    # Return the initialized singleton to avoid reloading the model.
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3 model singleton already exists; returning the existing instance.")
        return _bge_m3_ef

    # Load configuration from the environment, falling back to the default model.
    # A local path can be used when available; otherwise "BAAI/bge-m3" will download automatically.
    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
    logger.info(f"BGE model_name = {model_name}")
    logger.info(f"BGE path exists = {os.path.exists(model_name)}")
    logger.info(f"BGE path isdir = {os.path.isdir(model_name)}")

    device = embedding_config.bge_device or "cpu"
    use_fp16 = embedding_config.bge_fp16 or False

    # Log model initialization settings for troubleshooting.
    logger.info(
        "Initializing BGE-M3 model.",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True
        }
    )

    try:
        # Initialize BGE-M3 with built-in L2 normalization for Milvus IP retrieval.
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True  # Built-in L2 normalization for dense and sparse vectors.
        )
        logger.success("BGE-M3 model initialized successfully with built-in L2 normalization enabled.")
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3 model initialization failed: {str(e)}", exc_info=True)
        raise  # Re-raise for the caller to handle.


def generate_embeddings(texts):
    """
    Generate dense + sparse hybrid embeddings for a list of texts.
    The model applies built-in L2 normalization.
    :param texts: Text list to embed. A single text must also be wrapped in a list.
    :return: Dict with dense and sparse vector results.
    :raise: Exceptions from embedding generation are propagated to the caller.
    """
    # Validate input.
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("Invalid embedding input: texts must be a non-empty list.")
        raise ValueError("Parameter 'texts' must be a non-empty list containing text.")

    logger.info(f"Generating hybrid embeddings for {len(texts)} texts.")
    try:
        # Load the BGE-M3 model singleton.
        model = get_bge_m3_ef()
        # Encode texts into dense vectors and CSR-format sparse vectors.
        embeddings = model.encode_documents(texts)
        logger.debug(f"Model encoding completed; parsing sparse vector format for {len(texts)} texts.")

        # Parse sparse vectors into dictionaries for serialization and storage.
        processed_sparse = []
        for i in range(len(texts)):
            # Convert sparse indices from np.int64 to Python int for hashable dict keys.
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # Convert sparse weights from np.float32 to Python float for JSON serialization.
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # Build a sparse-vector dict of {feature_index: normalized_weight}.
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # Convert dense vectors to lists because numpy arrays are not JSON-serializable.
        result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # Nested list aligned with the input texts.
            "sparse": processed_sparse  # Dict list; the model has already applied L2 normalization.
        }
        logger.success(f"Generated embeddings for {len(texts)} texts; output format is ready for production use.")
        return result

    except Exception as e:
        logger.error(f"Text embedding generation failed: {str(e)}", exc_info=True)
        raise  # Propagate the exception so the caller can retry or degrade gracefully.


"""
Core design notes:
1. Built-in model normalization: normalize_embeddings=True applies L2 normalization
   to dense and sparse vectors, which fits Milvus IP retrieval well.
2. NumPy key safety: sparse_indices.tolist() converts np.int64 to native Python int,
   making dictionary keys hashable.
3. Sparse-value serialization: sparse_data.tolist() converts np.float32 to native
   Python float for JSON responses, API payloads, and Milvus writes.
4. Singleton optimization: the model is initialized once to avoid repeated load cost.
5. Business-call compatibility: returns nested dense-vector lists and sparse dict lists.
6. Layered logging: model initialization, embedding generation, and errors are all logged.
7. Input validation: rejects empty or non-list inputs before model internals fail.
"""
