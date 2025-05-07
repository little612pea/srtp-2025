import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 设置路径
DATA_DIR = "/home/jovyan/2024-srtp/srtp-final/4K50FPS/srts"  # 存放 txt 文件的文件夹
DB_DIR = "chroma_db"  # 向量数据库持久化目录

# 1. 加载所有 .txt 文件
def load_all_txt_files(data_dir):
    docs = []
    for file in os.listdir(data_dir):
        if file.endswith(".srt"):
            print(f"Loading {file}...")
            loader = TextLoader(os.path.join(data_dir, file), encoding="utf-8")
            docs.extend(loader.load())
    return docs

# 2. 切分文本
def split_text(docs):
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)

# 3. 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. 创建或加载向量数据库
if os.path.exists(DB_DIR):
    print("Loading existing vector database...")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    print("Creating new vector database...")
    docs = load_all_txt_files(DATA_DIR)
    chunks = split_text(docs)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

# 5. 设置 Qwen LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    model="qwen/qwen3-235b-a22b:free"
)

# 6. 创建检索器和 QA 链
retriever = db.as_retriever(search_kwargs={"k": 2})
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 7. 进行问答
response = rag_chain.invoke({"query": "请生成一段羽毛球解说词"})
print(response['result'])