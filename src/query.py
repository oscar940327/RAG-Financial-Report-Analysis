import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

load_dotenv()
client = OpenAI()
qdrant_db = QdrantClient(path="local_qdrant")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_standalone_query(query, history):

    system_prompt = "" \
    "You are an AI assistant. Your task is to rephrase the user's input into a clear, standalone ENGLISH question for semantic vector database search. " \
    "1. Translate to English if necessary. " \
    "2. Resolve any pronouns using the conversation history. " \
    "3. Use natural language ONLY. DO NOT use web search operators (like site:, AND, OR) or SEO keywords. " \
    "4. Just return the rephrased question."

    system_message = {"role": "system", "content": system_prompt}

    user_message = {"role": "user", "content": query}

    message_payload = [system_message] + history + [user_message]

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=message_payload
    )

    return response.choices[0].message.content

def ask_question(query, collection_name, history):

    """
    負責處理使用者的提問並給出精準答案
    """

    history = history or []

    standalone_query = get_standalone_query(query, history)
    print(f"後端檢索使用的英文 Query: {standalone_query}")

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=standalone_query
    )

    search_results = qdrant_db.query_points(collection_name=collection_name, query=query_embedding.data[0].embedding, limit=15)

    print(len(search_results.points))
    
    retrieved_texts = []

    for point in search_results.points:
        retrieved_texts.append(point.payload["text"])

    sentence_pairs = [[standalone_query, text] for text in retrieved_texts]

    scores = reranker.predict(sentence_pairs) 

    scored_texts = list(zip(retrieved_texts, scores))

    scored_texts.sort(key=lambda x: x[1], reverse=True)

    top_texts = [text for text, _ in scored_texts[:3]]

    context_text = "\n".join(top_texts)

    system_prompt = "" \
    "你是一位專業的分析助理。請「僅根據」以下提供的參考資料c來回答問題。" \
    "如果你無法從參考資料中找到答案，請直接回答「Based on the provided context, I cannot answer this question.」，嚴禁自行編造資訊。" \
    "【重要指令】：無論使用者使用中文或英文提問，你都必須「一律使用全英文 (English)」撰寫最終的分析與回答。" \
    f"\n參考資料：\n{context_text}" 

    system_message = {"role": "system", "content": system_prompt}

    user_message = {"role": "user", "content": query}

    message_payload = [system_message] + history + [user_message]

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=message_payload
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    try:
        test_query = "根據財報摘要，自由現金流(Free cash flow)是如何計算的？"
        target_collection = "Palantir_collection"

        print(f"你的問題: {test_query}")
        print("正在檢索資料並生成答案...")

        answer = ask_question(test_query, target_collection, [])
        print(f"\nAI 回答:\n{answer}")
    finally:
        qdrant_db.close()