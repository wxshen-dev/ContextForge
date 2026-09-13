# System imports for reading environment variables.
import os
# Logging imports for runtime success, failure, and error messages.
import logging
# Type hints for function parameters and return values.
from typing import List, Dict, Any, Optional
# Datetime utilities for message timestamps.
from datetime import datetime
# PyMongo driver imports for database connections and query sorting.
from pymongo import MongoClient, ASCENDING
# MongoDB primary-key type.
from bson import ObjectId
# Load environment variables from .env to avoid hardcoding sensitive config.
from dotenv import load_dotenv

# Load environment variables from .env so os.getenv can read configuration.
load_dotenv()


class HistoryMongoTool:
    """
    MongoDB read/write utility for conversation history, implemented with PyMongo.
    Encapsulates MongoDB connection setup, collection initialization, and index creation.
    """

    def __init__(self):
        """
        Initialize MongoDB connection, database, collection, and indexes.
        Initialization failures are logged and re-raised so callers can detect connection issues.
        """
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.chat_message = self.db["chat_message"]

            # Create an idempotent compound index for "latest records by session" queries.
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            logging.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise


def clear_history(session_id: str) -> int:
    """
    Clear all conversation history records for a session.
    :param session_id: Unique session identifier used to filter records.
    :return: Number of deleted documents, or 0 on failure.
    """
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logging.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logging.error(f"Error clearing history for session {session_id}: {e}")
        return 0


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        message_id: str = None
) -> str:
    """
    Insert or update a single conversation record in MongoDB.
    Inserts a new record when message_id is absent, and updates an existing record when present.
    :param session_id: Unique session identifier.
    :param role: Message role, such as user or assistant.
    :param text: Core message content.
    :param rewritten_query: Optional rewritten query for retrieval-augmented scenarios.
    :param item_names: Optional related item-name list.
    :param message_id: Optional document primary key ID. Updates when provided.
    :return: Inserted or updated record ID.
    """
    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "item_names": item_names,
        "ts": ts
    }

    mongo_tool = get_history_mongo_tool()
    if message_id:
        mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document}
        )
        return message_id

    result = mongo_tool.chat_message.insert_one(document)
    return str(result.inserted_id)


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """
    Batch-update related item names for conversation history records.
    Only updates records whose primary key is in the given list and whose item_names field is empty, missing, or None.
    :param ids: Primary key ID list as strings.
    :param item_names: New item-name list to set.
    :return: Number of updated documents, or 0 on failure.
    """
    mongo_tool = get_history_mongo_tool()
    try:
        object_ids = [ObjectId(i) for i in ids]
        result = mongo_tool.chat_message.update_many(
            {
                "_id": {"$in": object_ids},
                "$or": [
                    {"item_names": {"$exists": False}},
                    {"item_names": []},
                    {"item_names": None}
                ]
            },
            {"$set": {"item_names": item_names}}
        )
        logging.info(f"Updated {result.modified_count} records to item_names: {item_names}")
        return result.modified_count
    except Exception as e:
        logging.error(f"Error updating history item_names: {e}")
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Query the most recent N conversation records for a session.
    Results are returned in chronological order and can be used directly as LLM context.
    :param session_id: Unique session identifier used to filter records.
    :param limit: Maximum number of records to return. Defaults to 10.
    :return: Conversation record list as dictionaries, or an empty list on failure.
    """
    mongo_tool = get_history_mongo_tool()
    try:
        query = {"session_id": session_id}
        cursor = mongo_tool.chat_message.find(query).sort("ts", ASCENDING).limit(limit)
        return list(cursor)
    except Exception as e:
        logging.error(f"Error getting recent messages: {e}")
        return []


_history_mongo_tool = None


def get_history_mongo_tool() -> HistoryMongoTool:
    """
    Get the HistoryMongoTool singleton with lazy loading.
    :return: HistoryMongoTool singleton instance.
    """
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool


try:
    _history_mongo_tool = HistoryMongoTool()
except Exception as e:
    logging.warning(f"Could not initialize HistoryMongoTool on module load: {e}")


if __name__ == "__main__":
    sid = "000015_hybrid"
    save_chat_message(sid, "user", "Hello (Hybrid)")
    save_chat_message(sid, "assistant", "Hello! I am an assistant based on native Mongo and LangChain objects.")
    save_chat_message(sid, "user", "How do I replace the battery in this multimeter?", item_names=["Hybrid multimeter"])

    print("--- Query LangChain Object Records ---")
    messages = get_recent_messages(sid, limit=5)
    print(f"Records found: {len(messages)}")
    for m in messages:
        print(f" {m}  ")
