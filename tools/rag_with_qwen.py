import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


# 全局配置
DATA_DIR = "/home/jovyan/2024-srtp/srtp-final/4K50FPS/srts"  # 存放 .srt 文件的文件夹
DB_DIR = "chroma_db"  # 向量数据库持久化目录


# 1. 加载所有 .srt 文件
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
def init_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# 4. 创建或加载向量数据库
def init_vectorstore(embeddings):
    if os.path.exists(DB_DIR):
        print("Loading existing vector database...")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        print("Creating new vector database...")
        docs = load_all_txt_files(DATA_DIR)
        chunks = split_text(docs)
        return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


# 5. 设置 Qwen LLM
def init_llm():
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
        model="deepseek/deepseek-chat-v3-0324:free"
    )


# 6. 创建检索器和 QA 链
def init_rag_chain(llm, retriever):
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# 7. 封装查询函数供 Gradio 调用
class RAGService:
    def __init__(self):
        self.embeddings = init_embeddings()
        self.db = init_vectorstore(self.embeddings)
        self.retriever = self.db.as_retriever(search_kwargs={"k": 2})
        self.llm = init_llm()
        self.rag_chain = init_rag_chain(self.llm, self.retriever)

    def query(self, question: str) -> str:
        result = self.rag_chain.invoke({"query": question})
        return result.get("result", "抱歉，我没有找到相关信息。")


# 实例化服务对象
rag_service = RAGService()


# 供 Gradio 调用的接口函数
def query_rag(question: str) -> str:
    return rag_service.query(question)


# 示例调用
if __name__ == "__main__":
    response = query_rag("请生成一段羽毛球解说词")
    print(response)