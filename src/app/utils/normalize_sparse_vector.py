import numpy as np
def normalize_sparse_vector(sparse_vec):
    """
    Apply L2 normalization to a sparse vector.
    Only non-zero dimensions are processed; zero dimensions are unaffected.
    :param sparse_vec: Original sparse vector in dict format: {dimension: value}
    :return: Normalized sparse vector
    """
    if not sparse_vec:  # Return empty vectors as-is.
        return sparse_vec

    # Extract values from non-zero dimensions.
    values = np.array(list(sparse_vec.values()), dtype=np.float64)
    # Compute the L2 norm and guard against division by zero.
    l2_norm = np.linalg.norm(values)
    if l2_norm < 1e-9:  # Return near-zero vectors as-is.
        return sparse_vec

    # Normalize each value by the L2 norm.
    normalized_values = values / l2_norm
    # normalized_values = (values / l2_norm).astype(np.float32)  # Convert uniformly to float32.
    # Rebuild the sparse vector dict.
    return dict(zip(sparse_vec.keys(), normalized_values))
