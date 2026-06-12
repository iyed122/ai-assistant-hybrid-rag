import os
from pymongo import MongoClient
from bson import json_util

# --- CONFIGURATION ---
USER_PROFILE = os.environ.get("USERPROFILE")
FILE_PATH = os.path.join(USER_PROFILE, "Desktop", "knowledge_base.chat_history_200.json")

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "knowledge_base"
COLLECTION_NAME = "chat_history"

def undo_prepend():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # 1. Verify the file exists to identify what to delete
    if not os.path.exists(FILE_PATH):
        print(f"ERROR: Cannot undo. File not found at {FILE_PATH}")
        return

    # 2. Load the JSON data to get the specific IDs
    print(f"Identifying documents to remove from: {FILE_PATH}")
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        # json_util ensures we handle ObjectIDs correctly
        data_to_remove = json_util.loads(f.read())

    # 3. Extract the list of IDs
    ids_to_remove = [doc['_id'] for doc in data_to_remove if '_id' in doc]

    if not ids_to_remove:
        print("No valid IDs found in the file to perform an undo.")
        return

    # 4. Delete only those specific documents
    print(f"Removing {len(ids_to_remove)} prepended documents...")
    result = collection.delete_many({"_id": {"$in": ids_to_remove}})

    print(f"SUCCESS: Removed {result.deleted_count} documents.")
    print("Your database now only contains your initial queries.")

if __name__ == "__main__":
    undo_prepend()