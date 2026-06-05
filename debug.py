# debug.py
import chromadb

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection("maternity_rag_local_v1")

print(f"チャンク数: {collection.count()}")

# 実際に保存されているテキストを3件表示
results = collection.get(limit=3, include=["documents", "metadatas"])
for i, doc in enumerate(results["documents"]):
    print(f"\n--- チャンク {i+1} ---")
    print(doc[:300])  # 最初の300文字