import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()
client = OpenAI()
qdrant_db = QdrantClient(path="local_qdrant")

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)

    pages = loader.load()

    full_text = ""

    for p in pages:
        full_text += p.page_content + "\n"

    return full_text

def ingest_data(text):

    """
    負責將原始長文處理好然後存到資料庫
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50
    )

    chunks = text_splitter.split_text(text)
    print(f"總共切出 {len(chunks)} 個字塊，準備分批轉換向量...")

    all_embeddings = []
    batch_size = 1000

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch_chunks
        )

        all_embeddings.extend([data.embedding for data in response.data])
        print(f"已完成 {min(i + batch_size, len(chunks))}/{len(chunks)} 筆向量轉換...")    

    collection_name = "Palantir_collection"

    if not qdrant_db.collection_exists(collection_name):
        qdrant_db.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    points_list = []
    for index, chunk_text in enumerate(chunks):
        points_list.append(PointStruct(
            id=index, 
            vector=all_embeddings[index], 
            payload={"text": chunk_text}))

    print("開始將向量寫入 Qdrant 資料庫...")
    qdrant_db.upload_points(
        collection_name=collection_name,
        points=points_list,
        batch_size=500
    )    

if __name__ == "__main__":
    pdf_path = "docs/2025 FY PLTR 10-K.pdf"
    print("開始讀取與切割 PDf...")
    document_text = load_pdf(pdf_path)

    print("開始將資料向量化並寫入 Qdrant...")
    ingest_data(document_text)

    qdrant_db.close()
    print("資料庫建置完成")