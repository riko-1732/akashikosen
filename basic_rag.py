import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 日本語用プロンプト
JP_QA_PROMPT = PromptTemplate(
    "以下の情報を参考にして、質問に日本語で答えてください。\n"
    "情報に記載がない場合は「その情報は手元の資料にありません。」と答えてください。\n\n"
    "参考情報:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "質問: {query_str}\n"
    "回答:"
)

def main():
    print("ローカルEmbeddingモデルを準備中...")
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.embed_batch_size = 100

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("maternity_rag_local_v1")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() > 0:
        print(">>> 既存のデータベースを読み込みました。")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
    else:
        print(">>> PDFを読み込んで新規作成します...")
        if not os.path.exists("./data") or not os.listdir("./data"):
            print("エラー: dataフォルダにPDFを入れてください。")
            return
        documents = SimpleDirectoryReader("./data").load_data()
        print("ベクトル化中...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        print(">>> 完了しました。")

    # 日本語プロンプトをセット
    query_engine = index.as_query_engine(
        text_qa_template=JP_QA_PROMPT,
        similarity_top_k=3  # 参照するチャンク数
    )

    print("\n" + "="*40)
    print("RAGチャットボット起動")
    print("終了: 'q' を入力")
    print("="*40)

    while True:
        question = input("\nあなた: ")
        if question.lower() in ['q', 'quit', 'exit']:
            print("終了します。")
            break
        if not question.strip():
            continue

        print("思考中...")
        response = query_engine.query(question)
        print(f"\nAI: {response}")

        print("\n[参照元]")
        for node in response.source_nodes:
            score = node.score if node.score is not None else 0.0
            print(f"  - {node.metadata.get('file_name')} (Score: {score:.4f})")

if __name__ == "__main__":
    main()