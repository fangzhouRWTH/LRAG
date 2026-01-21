import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# === 1. 加载并分块知识库 ===
with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.create_documents([text])

# === 2. 初始化嵌入模型（本地）===
model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},  # 可改为 "cuda" 如果有 GPU
    encode_kwargs={"normalize_embeddings": True}
)

# === 3. 构建 FAISS 向量库 ===
vector_store = FAISS.from_documents(chunks, embeddings)

# === 4. 加载本地 LLM（Mistral 7B 4-bit 量化）===
# 下载模型: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# 放到 ./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
llm = LlamaCpp(
    model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,        # 根据 CPU 核心数调整
    n_gpu_layers=0,     # 设为 >0 若使用 GPU（需 llama-cpp 支持 CUDA）
    verbose=False
)

# === 5. 构建 RAG 链 ===
prompt_template = """
[INST]
Use only the following context to answer the question.
If you don't know, say "I don't know based on the provided information."

Context:
{context}

Question: {question}
[/INST]
Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# === 6. 交互式问答 ===
if __name__ == "__main__":
    print("✅ Local RAG is ready! Ask any question (type 'exit' to quit):")
    while True:
        query = input("\n> ")
        if query.lower() == "exit":
            break
        result = qa_chain({"query": query})
        print("\n🤖 Answer:", result["result"].strip())
        print("\n📚 Sources:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"  [{i}] {doc.page_content[:100]}...")