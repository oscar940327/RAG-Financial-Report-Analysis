# RAG Financial Report Analysis 
**企業財報與供應鏈風險問答系統 (Two-Stage RAG)**

這是一個基於大型語言模型 (LLM) 與檢索增強生成 (RAG) 技術建置的財務問答系統。本專案專注於解析大型企業（如 Tesla、Palantir）的財務報告，透過導入 **Cross-Encoder 重排序 (Reranking)** 與 **獨立問題生成 (Query Rewriting)** 機制，解決傳統向量檢索在處理密集財務數據與跨語系提問時的精準度瓶頸。

## 核心技術特徵 (Key Features)

- **兩階段檢索架構 (Two-Stage Retrieval)**
    - **初階召回 (Recall)**：使用 OpenAI `text-embedding-3-small` 將文本轉為密集向量 (Dense Vector)，並透過 Qdrant 進行餘弦相似度 (Cosine Similarity) 快速初篩 (Top-15)。
    - **精確重排序 (Precision Reranking)**：導入 `sentence-transformers` 的 Cross-Encoder 模型 (`ms-marco-MiniLM-L-6-v2`)，對初篩結果與使用者提問進行語意關聯性二次評分，精準抓取 Top-3 段落，有效排除財報中字面相似的數字雜訊。
- **多輪對話上下文記憶 (Contextual Memory)**
    - 實作 Standalone Query Generation（獨立問題生成）機制。在進入向量檢索前，由 LLM 自動讀取歷史對話，替換代名詞並將中文提問強制改寫為無雜訊的英文檢索句，解決語言維度不對齊的問題。
- **多資料集狀態管理 (Multi-Dataset State Management)**
    - 支援多間企業財報（Tesla、Palantir）即時切換。實作 Streamlit Callback 機制，在切換資料庫目標時自動清空 `session_state`，防止跨財報的上下文記憶錯亂。
- **嚴格防幻覺機制 (Anti-Hallucination)**
    - 採用封閉領域限制 (Closed-Book Constraint)，強制 LLM 僅能依據 Qdrant 檢索出的 `context_text` 進行全英文回答。若無相關資料，系統會嚴格輸出無法作答，拒絕編造財務數據。

## 技術堆疊 (Tech Stack)

* **Frontend**: Streamlit
* **LLM & Embeddings**: OpenAI API (`gpt-5.4`, `text-embedding-3-small`)
* **Vector Database**: Qdrant (Local deployment)
* **Reranker**: `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
* **Document Parsing**: LangChain (`PyPDFLoader`, `RecursiveCharacterTextSplitter`)

## 本機安裝與執行 (Local Setup)

1. **Clone 專案與建立虛擬環境**
    ```bash
    git clone <your-github-repo-url>
    cd <your-repo-folder>
    python -m venv .venv
    .\.venv\Scripts\activate  # Windows 環境
    ```

2. **安裝依賴套件**
    ```bash
    pip install -r requirements.txt
    ```

3. **設定環境變數**
    在專案根目錄建立 `.env` 檔案，並寫入你的 OpenAI API Key：
    ```bash
    OPENAI_API_KEY="sk-..."
    ```

4. **執行資料寫入管線 (ETL Pipeline)**
    將財報 PDF 放入 `docs/` 資料夾，並執行資料清洗、切塊與向量化寫入：
    ```bash
    python src/ingest.py
    ```

5. **啟動前端應用程式**
    ```bash
    streamlit run src/app.py
    ```
    初次啟動時，系統會在背景自動下載 Hugging Face 的 Cross-Encoder 模型權重，需稍候片刻。

## 雲端部署準備 (Streamlit Cloud Deployment)

本專案支援直接部署至 Streamlit Community Cloud：

1. 將專案推至 GitHub（請確保 `.gitignore` 已排除 `.env` 與 `.venv/`）。
2. 於 Streamlit 後台連接 GitHub Repo，並指定 Main file path 為 `src/app.py`。
3. 進入 Advanced Settings，在 Secrets 區塊設定 `OPENAI_API_KEY` 後即可一鍵部署。

## Disclaimer

本系統僅作學術與技術實作探討，檢索與生成之內容不構成任何投資建議。