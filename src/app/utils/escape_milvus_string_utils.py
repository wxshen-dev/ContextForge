# ===================== Core Helper Functions =====================
def escape_milvus_string(value: str) -> str:
    """
    Escape strings for Milvus filter expressions.
    This prevents special characters in raw values from breaking filter_expr
    parsing and keeps CRUD operations reliable.

    Escaping rules:
        1. Backslash (\\) -> double backslash (\\\\)
        2. Double quote (") -> escaped double quote (\\")
        3. Newlines, carriage returns, and tabs -> spaces

    Args:
        value: Raw string to escape, such as an item name or file title.

    Returns:
        Safe string that can be used directly in a Milvus filter_expr.
    """
    if value is None:
        return ""
    # Convert input to string to avoid errors from non-string values.
    s = str(value)
    # Escape special characters according to Milvus expression rules.
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    # Keep the expression on a single valid line.
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return s
