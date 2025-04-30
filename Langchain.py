import json
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import os

# 載入 .env 檔案
load_dotenv()

# 從環境變數中獲取 OpenAI API 金鑰
openai_api_key = os.getenv("OPENAI_API_KEY")

# 檢查 API 金鑰是否存在
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 環境變數未設定。請檢查你的 .env 檔案。")

# 步驟 1: 載入產品資料
with open("products.json", "r", encoding="utf-8") as f:
    products_data = json.load(f)

product_texts = [p["description"] for p in products_data]
product_names = [p["name"] for p in products_data]

# 步驟 2: 初始化 OpenAI client
client = OpenAI(api_key=openai_api_key)

# 步驟 3: 創建產品描述的向量嵌入
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

product_embeddings = [get_embedding(text) for text in product_texts]

# 步驟 4: 回答問題的函數
def answer_question(question, product_embeddings, product_texts, client, top_n=2):
    question_embedding = get_embedding(question)

    # 計算問題嵌入與產品嵌入之間的餘弦相似度
    similarities = cosine_similarity([question_embedding], product_embeddings)[0]

    # 獲取最相似的產品描述的索引
    most_similar_indices = np.argsort(similarities)[::-1][:top_n]

    # 獲取相關的產品描述
    relevant_products = [product_texts[i] for i in most_similar_indices]

    # 創建 Prompt
    prompt = f"根據以下產品描述回答問題：\n\n"
    for i, product in enumerate(relevant_products):
        prompt += f"{i+1}. {product}\n"
    prompt += f"\n問題：{question}\n答案："

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 你可以選擇其他模型
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # 調整以獲得更確定或更多樣的回應
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"回答問題時發生錯誤：{e}"

# 步驟 5: 測試問答機器人
queries = [
    "哪款筆記型電腦比較輕？",
    "桌上型電腦的處理器是什麼？",
    "有沒有續航很久的耳機？"
]

for query in queries:
    answer = answer_question(query, product_embeddings, product_texts, client)
    print(f"問題：{query}")