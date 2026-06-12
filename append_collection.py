import json
import os
from pymongo import MongoClient
from bson import json_util

# --- CONFIGURATION ---
USER_PROFILE = os.environ.get("USERPROFILE")
FILE_PATH_245 = os.path.join(USER_PROFILE, "Desktop", "knowledge_base.chat_history_245.json")
FILE_PATH_EVAL = os.path.join(USER_PROFILE, "Desktop", "knowledge_base.chat_history_eval.json")

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "knowledge_base"
COLLECTION_NAME = "chat_history"

def prepend_and_fix():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # 1. Check if both files exist
    for file_path in [FILE_PATH_245, FILE_PATH_EVAL]:
        if not os.path.exists(file_path):
            print(f"ERROR: File not found at {file_path}")
            return

    # 2. Load data and remove _id to prevent DuplicateKeyError collisions
    print(f"Reading: {FILE_PATH_245}")
    with open(FILE_PATH_245, 'r', encoding='utf-8') as f:
        data_245 = json_util.loads(f.read())
        for doc in data_245:
            doc.pop('_id', None)

    print(f"Reading: {FILE_PATH_EVAL}")
    with open(FILE_PATH_EVAL, 'r', encoding='utf-8') as f:
        data_eval = json_util.loads(f.read())
        for doc in data_eval:
            doc.pop('_id', None)

    # 3. Fetch existing data currently in Mongo
    print("Backing up current database entries...")
    existing_data = list(collection.find())

    # 4. Wipe the collection
    print("Clearing collection to reorder...")
    collection.delete_many({})

    # 5. Merge: [245] + [Eval] + [Existing]
    combined_data = data_245 + data_eval + existing_data

    # 6. Insert back into Mongo
    if combined_data:
        print(f"Inserting {len(combined_data)} total documents...")
        collection.insert_many(combined_data)
        
        # 7. Final Clean-up
        print("Removing evaluation fields from entries...")
        collection.update_many({}, {"$unset": {"evaluation": ""}})
        
        print("\nSUCCESS: Both files prepended safely. Database is ready.")
    else:
        print("No data found to insert.")

if __name__ == "__main__":
    prepend_and_fix()