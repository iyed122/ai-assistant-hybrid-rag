from pymongo import MongoClient
db = MongoClient('mongodb://localhost:27017/')['knowledge_base']
print('document_chunks total:  ', db.document_chunks.count_documents({}))
print('with embeddings:         ', db.document_chunks.count_documents({'embedding': {'$exists': True}}))
print('normalized_documents:    ', db.normalized_documents.count_documents({}))
