import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import time

load_dotenv()

# ===== 設定 =====
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY が .env に設定されていません")

# 日本語プロンプトテンプレート（指示に合わせて修正）
JP_QA_PROMPT = PromptTemplate(
    "以下の情報をもとに、質問に日本語で答えてください。\n"
    "【必須ルール】\n"
    "・必ず「です・ます」調（敬体）で答えてください。\n"
    "・回答は簡潔にまとめ、3〜5項目以内に収めてください。\n"
    "・中学生とその保護者にもわかりやすい言葉を使ってください。\n"
    "・質問に直接関係のない情報は回答に含めないでください。\n"
    "・参考情報に書かれている内容だけを根拠に回答してください。\n"
    "・推測や一般知識は使用しないでください。\n"
    "・回答は2〜5文程度で説明してください。\n"
    "・1文だけで終わらせないでください。\n"
    "・入試日程・出願期間・募集要項など年度に依存する情報を回答する場合は、末尾に「※この情報は令和8年度のものです。最新情報は明石高専公式サイトをご確認ください。」と必ず添えてください。\n"
    "・情報に記載がない場合は「その情報は手元の資料にありません。明石高専の入試窓口にお問い合わせください。」と答えてください。\n\n"
    "参考情報:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "質問: {query_str}\n"
    "回答:"
)

# ===== データモデル =====
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# ===== FastAPI アプリ初期化 =====
app = FastAPI(
    title="明石高専 案内ボット",
    description="学校紹介・入試案内に特化したRAGチャットボット",
    version="1.0.0",
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== グローバル状態 =====
query_engine = None
response_cache = {}


def init_rag():
    """RAG エンジンの初期化"""
    global query_engine
    
    print("🚀 Embedding モデルを準備中...")
    Settings.llm = OpenAI(model="gpt-5.4", api_key=API_KEY, temperature=0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.embed_batch_size = 100

    print("📦 ChromaDB から既存ベクトルストアを読み込み中...")
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("akashi_kosen_v1")
    
    if chroma_collection.count() == 0:
        raise RuntimeError(
            "❌ ベクトルDB が空です。先に `uv run python build_db.py` を実行してください。"
        )

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print(f"✅ {chroma_collection.count()} チャンク読み込み完了")
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )

    query_engine = index.as_query_engine(
        text_qa_template=JP_QA_PROMPT,
        similarity_top_k=5,
        response_mode="compact"
    )
    print("✅ RAG エンジン初期化完了\n")


@app.on_event("startup")
async def startup_event():
    """アプリ起動時の初期化"""
    init_rag()


# ===== API エンドポイント =====

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "ok"}

# /chat エンドポイントを修正
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="メッセージが空です")
    if query_engine is None:
        raise HTTPException(status_code=500, detail="RAG エンジンが初期化されていません")

    cache_key = request.message.strip()
    if cache_key in response_cache:
        print(f"✅ キャッシュヒット: {cache_key}")
        return response_cache[cache_key]

    try:
        import time
        t1 = time.time()
        response = await query_engine.aquery(request.message)
        print(f"[RAG処理] {time.time() - t1:.2f}秒")

        sources = []
        seen = set()
        for node in response.source_nodes:
            file_name = node.metadata.get("file_name")
            if file_name and file_name not in seen:
                sources.append(file_name)
                seen.add(file_name)

        result = ChatResponse(answer=str(response), sources=sources)
        response_cache[cache_key] = result
        return result

    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        raise HTTPException(status_code=500, detail="回答生成中にエラーが発生しました")

# ===== 静的ファイル（フロントエンド）- API定義の後ろに配置 =====
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


# ===== 起動スクリプト =====
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("明石高専 RAG チャットボット")
    print("=" * 50)
    print("🌍 http://localhost:8000 で起動します\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)