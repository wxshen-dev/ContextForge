# Load environment variables from .env, such as Milvus URL, KG service URL, and BGE model path.
from dotenv import load_dotenv
# Import LangGraph core dependencies: StateGraph and built-in START/END constants.
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
# Import the custom state class that stores all data shared or modified by workflow nodes.
from app.import_process.agent.state import ImportGraphState, create_default_state
# Import custom business nodes for each knowledge-base import step.
from app.import_process.agent.nodes.node_entry import node_entry  # Entry node: initialize parameters and validate input.
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md  # PDF to Markdown conversion.
from app.import_process.agent.nodes.node_md_img import node_md_img  # Markdown image processing.
from app.import_process.agent.nodes.node_document_split import node_document_split  # Split long documents into chunks.
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  # Identify core item names.
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # Convert text chunks into BGE vectors.
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # Write vector data into Milvus.


# Initialize environment variables before reading configuration.
load_dotenv()

# ===================== 1. Initialize the LangGraph State Graph =====================
# StateGraph is the LangGraph core class for building stateful workflows.
# ImportGraphState is a custom TypedDict that defines all workflow state fields.
# Each node receives this state object, and returned key-value pairs are merged back into it.
workflow = StateGraph(ImportGraphState)

# ===================== 2. Register Business Nodes =====================
# Syntax: add_node("unique_node_id", node_function).
# Node functions must receive the state object and return a dict used to update state.
# Nodes are registered in knowledge-base import order for easier maintenance.
workflow.add_node("node_entry", node_entry)  # Flow entry: parameter initialization and input validation.
workflow.add_node("node_pdf_to_md", node_pdf_to_md)  # PDF to Markdown preprocessing.
workflow.add_node("node_md_img", node_md_img)  # Ensure images in Markdown are accessible.
workflow.add_node("node_document_split", node_document_split)  # Split large text for embedding and reasoning.
workflow.add_node("node_item_name_recognition", node_item_name_recognition)  # Extract the core business identifier.
workflow.add_node("node_bge_embedding", node_bge_embedding)  # Convert text to vectors for Milvus storage.
workflow.add_node("node_import_milvus", node_import_milvus)  # Persist vector data into Milvus.

# ===================== 3. Set the Workflow Entry Node =====================
# set_entry_point("node_id") directly specifies the first node in the flow.
# Equivalent to workflow.add_edge(START, "node_entry"), but more concise.
workflow.set_entry_point("node_entry")

# ===================== 4. Define Conditional Routing After Entry =====================
# Dynamically choose the next path from state flags for PDF import or direct Markdown import.
# The function receives the state object and returns a target node ID or END.
def route_after_entry(state: ImportGraphState) -> str:
    """
    Conditional routing logic after the entry node.
    :param state: Full workflow state containing configuration and intermediate results.
    :return: Target node ID or END. LangGraph will jump to the returned node.
    """
    # Branch 1: direct Markdown import; skip PDF conversion and process Markdown images.
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    # Branch 2: PDF import; convert PDF to Markdown before continuing.
    elif state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    # Branch 3: no import path enabled; terminate the workflow.
    else:
        return END

# Register conditional edges by binding the entry node to the routing function.
# After the source node finishes, LangGraph calls the router and jumps to the returned target.
workflow.add_conditional_edges(
    "node_entry",
    route_after_entry,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END
    }
)

# ===================== 5. Register Static Sequential Edges =====================
# All branches merge into a fixed sequence from Markdown image processing through Milvus import.
# add_edge("source_node_id", "target_node_id_or_END") defines a static route without branching.
workflow.add_edge("node_pdf_to_md", "node_md_img")  # PDF conversion complete -> Markdown image processing.
workflow.add_edge("node_md_img", "node_document_split")  # Markdown processing complete -> document splitting.
workflow.add_edge("node_document_split", "node_item_name_recognition")  # Splitting complete -> item-name recognition.
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")  # Item-name recognition complete -> BGE embedding.
workflow.add_edge("node_bge_embedding", "node_import_milvus")  # Embedding complete -> import into Milvus.
workflow.add_edge("node_import_milvus", END)  # Milvus import complete -> workflow ends.

# ===================== 6. Compile the Workflow =====================
# compile() turns the StateGraph flow into an executable LangGraph application.
# kb_import_app can be invoked repeatedly with different initial states.
kb_import_app = workflow.compile()
