# 步驟 1: 匯入必要的函式庫
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI  # 或其他 LLM
from langchain.chains import RetrievalQA
import json

# 步驟 2: 載入產品資料
with open("products.json", "r", encoding="utf-8") as f:
    products_data = json.load(f)

product_texts = [p["description"] for p in products_data]
product_names = [p["name"] for p in products_data]

# 步驟 3: 建立向量儲存庫
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_texts(product_texts, embeddings)
retriever = db.as_retriever()

# 步驟 4: 載入 LLM (需要 OpenAI API 金鑰)
llm = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# 步驟 5: 建立檢索問答鏈
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 步驟 6: 測試問答機器人
query = "哪款筆記型電腦比較輕？"
result = qa.run(query)
print(f"問題：{query}")
print(f"答案：{result}")

query = "桌上型電腦的處理器是什麼？"
result = qa.run(query)
print(f"問題：{query}")
print(f"答案：{result}")