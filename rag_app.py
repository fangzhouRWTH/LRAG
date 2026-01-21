import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp

# === 1. 加载并分块知识库 ===
with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.create_documents([text])

# === 2. 初始化嵌入模型 + 构建向量库 ===
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vector_store = FAISS.from_documents(chunks, embeddings)

# === 3. 初始化本地 LLM ===
llm = LlamaCpp(
    model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers = 40,
    verbose=False
)

# === 4. 手动拆分 RAG 步骤 ===
def rag_step_by_step(query: str, k: int = 2):
    # 🔍 步骤 1: 检索知识库 → 获取 context
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    
    # 提取文本内容（可选：保留元数据）
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"🔍 检索到 {len(docs)} 段相关文档:\n{context}\n{'-'*50}")
    
    # 🧠 步骤 2: 手动构造 prompt
    prompt = f"""[INST]
Use only the following context to answer the question.
If you don't know, say "I don't know based on the provided information."

Context:
{context}

Question: {query}
[/INST]
Answer:
"""
    
    # 🤖 步骤 3: 调用大模型
    response = llm.invoke(prompt)
    #response = prompt
    return response.strip(), docs

# === 5. 交互式问答 ===
if __name__ == "__main__":
    print("✅ 拆分版 RAG 已启动！输入问题（'exit' 退出）:")
    while True:
        query = input("\n> ")
        if query.lower() == "exit":
            break
        
        answer, sources = rag_step_by_step(query)
        print(f"\n🤖 最终答案:\n{answer}\n")
        print("📚 引用来源:")
        for i, doc in enumerate(sources, 1):
            print(f"  [{i}] {doc.page_content[:120]}...")