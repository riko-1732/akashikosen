# build_db.py
import os
import chromadb
import fitz  # PyMuPDF
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

def extract_text_from_pdfs(data_dir: str) -> list[Document]:
    """PyMuPDFで全PDFからテキストを抽出"""
    documents = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".pdf"):
            continue
        filepath = os.path.join(data_dir, filename)
        print(f"読み込み中: {filename}")
        
        pdf = fitz.open(filepath)
        for page_num, page in enumerate(pdf):
            text = page.get_text()
            if text.strip():  # 空ページは無視
                documents.append(Document(
                    text=text,
                    metadata={
                        "file_name": filename,
                        "page": page_num + 1
                    }
                ))
        
        print(f"  → {len(pdf)}ページ読み込みました")
        pdf.close()
    
    print(f"\n合計 {len(documents)} ページ分のテキストを取得")
    return documents

def main():
    print("Embeddingモデル準備中...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.embed_batch_size = 100

    # 古いDBを削除して作り直す
    db = chromadb.PersistentClient(path="./chroma_db")
    
    # 既存コレクションを削除（バイナリデータが混じっているため）
    try:
        db.delete_collection("maternity_rag_local_v1")
        print("古いDBを削除しました")
    except:
        pass
    
    chroma_collection = db.get_or_create_collection("maternity_rag_local_v1")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # PyMuPDFでテキスト抽出
    documents = extract_text_from_pdfs("./data")
    if not documents:
        print("エラー: dataフォルダにPDFがありません")
        return

    print("\nベクトル化中...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("✅ DB作成完了！")

if __name__ == "__main__":
    main()