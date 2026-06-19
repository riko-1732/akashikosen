# 明石工業高等専門学校 RAG チャットボット

学校紹介・入試案内に特化した**生成AIチャットボット**です。
合同発表会のブース展示用デモとして設計されています。

## 🎯 特徴

- **RAGベース**：ローカルに保管した公式資料から回答を生成
- **ローカル処理**：Embeddingはすべてローカルで実行（HuggingFace）
- **中学生向け**：難しい説明を避け、わかりやすく回答
- **参照元表示**：回答時に参照したファイルを明記
- **シンプルUI**：端末は横画面ノートPC向けの2カラムレイアウト

## 🛠️ 技術スタック

| 項目               | 選択                           |
| ------------------ | ------------------------------ |
| バックエンド       | Python / FastAPI               |
| RAG フレームワーク | LlamaIndex                     |
| LLM API            | Google Gemini 2.5 Flash        |
| ベクトルDB         | ChromaDB (ローカル・永続化)    |
| Embedding          | intfloat/multilingual-e5-small |
| PDF テキスト抽出   | PyMuPDF (fitz)                 |
| フロントエンド     | HTML / CSS / JavaScript        |
| 環境管理           | uv (Python)                    |

## 📁 ディレクトリ構成

```
akashi-chatbot/
├── data/                  # RAG参照用PDF/テキスト
├── chroma_db/             # ChromaDB永続化ディレクトリ（自動生成）
├── frontend/              # フロントエンド
│   ├── index.html
│   └── style.css
├── build_db.py            # DB構築スクリプト
├── main.py                # FastAPI バックエンド
├── .env                   # GOOGLE_API_KEY
├── pyproject.toml         # 依存管理（uv）
└── README.md              # このファイル
```

## 🚀 起動手順

### 1. 依存インストール

```bash
uv sync
```

### 2. DB構築（初回または更新時）

`data/` ディレクトリに PDF / テキストファイルを配置してから実行：

```bash
uv run python build_db.py
```

**入力ファイル形式：**

- PDF: `data/*.pdf`
- テキスト: `data/*.txt`, `data/*.md`

**出力:**

- `chroma_db/akashi_kosen_v1` コレクション作成

### 3. サーバー起動

```bash
uv run uvicorn main:app --reload
```

### 4. ブラウザで開く

```
http://localhost:8000
```

左パネルのサンプル質問をクリックするか、チャット欄に直接質問入力してください。

## 🔧 API エンドポイント

### `POST /chat`

**リクエスト:**

```json
{
  "message": "ユーザーの質問"
}
```

**レスポンス:**

```json
{
  "answer": "回答テキスト",
  "sources": ["参照ファイル名1", "参照ファイル名2"]
}
```

### `GET /health`

ヘルスチェック

```json
{
  "status": "ok"
}
```

## ⚙️ 設定項目

### プロンプトテンプレート（main.py）

中学生向けの日本語プロンプトがハードコードされています：

```python
JP_QA_PROMPT = PromptTemplate(
    "以下の情報をもとに、質問に日本語でわかりやすく答えてください。\n"
    "回答は中学生でも理解できる言葉を使ってください。\n"
    ...
)
```

### RAG パラメータ

| 項目               | 値                               | 備考                   |
| ------------------ | -------------------------------- | ---------------------- |
| `chunk_size`       | 512                              | テキストチャンクサイズ |
| `chunk_overlap`    | 50                               | チャンク間の重複       |
| `similarity_top_k` | 3                                | 検索結果の上位件数     |
| Embedding モデル   | `intfloat/multilingual-e5-small` | HuggingFace            |

## 🔐 環境設定

### `.env` ファイル

```
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE
```

Google AI Studio から API キーを取得してください：
https://aistudio.google.com/app/apikey

## 💡 使用例

**中学生の質問例：**

- 「普通高校と何が違うの？」
- 「入試ではどんな科目があるの？」
- 「卒業後の進路を教えて」
- 「どんな学科があるの？」
- 「寮はありますか？」

## 📝 よくある質問

### Q: ブラウザを開いてもページが真っ白です

**A:** サーバーログを確認してください。`build_db.py` を実行しているか、GOOGLE_API_KEY が設定されているか確認してください。

### Q: API から「ベクトルDB が空です」というエラーが出ます

**A:** 以下の手順で DB を構築してください：

```bash
# 1. data/ に PDF を配置
# 2. DB 構築
uv run python build_db.py
# 3. サーバー再起動
uv run uvicorn main:app --reload
```

### Q: 回答が「その情報は手元の資料にありません」ばかりです

**A:** PDF の品質やフォーマット、検索ロジックを確認してください。例えば：

- テキスト層のない画像スキャンPDF → OCR 必要
- スキャン品質が低い → テキスト認識率低下
- 検索キーワードと資料内容の言葉遣いが違う

## 🛠️ トラブルシューティング

### リセットボタンで履歴削除後も応答が古い場合

ブラウザキャッシュをクリアしてください。

### サーバーが起動しない場合

```bash
# ポート8000が既に使われていないか確認
netstat -ano | findstr :8000

# 別のポートで起動（例：8001）
uv run uvicorn main:app --reload --port 8001
```

## 📚 参考資料

- [LlamaIndex ドキュメント](https://docs.llamaindex.ai/)
- [ChromaDB ドキュメント](https://docs.trychroma.com/)
- [FastAPI ドキュメント](https://fastapi.tiangolo.com/)

## 📄 ライセンス

このプロジェクトは明石工業高等専門学校の内部用です。

---

**最終更新:** 2026年6月19日
