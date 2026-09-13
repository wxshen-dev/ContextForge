# Core dependencies shared with other configuration classes.
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load .env once at module import time.
load_dotenv()

# ===================== Other Config Classes Can Be Added Above =====================

# Milvus vector database configuration.
@dataclass
class MilvusConfig:
    milvus_url: str          # Milvus server connection URL.
    chunks_collection: str   # Collection name for document chunks.
    entity_name_collection: str  # Reserved collection for entity names.
    item_name_collection: str    # Collection name for document item names.

# Instantiate the Milvus config object.
milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL"),
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),
    entity_name_collection=os.getenv("ENTITY_NAME_COLLECTION"),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION")
)
