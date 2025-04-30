import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

load_dotenv()

# 設定Embedding
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 讀取向量資料庫
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vectordb.as_retriever()

# 設定LLM
llm = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=os.getenv("OPENAI_API_KEY")  # 或直接填"你的API KEY"
)

# 問答系統
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = input("請問你要查什麼？")
result = qa.run(query)
print(result)
