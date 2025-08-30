import json
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from llama_cpp import Llama
from langchain_ollama import OllamaLLM


with open("cag_data/hackbattle_doc.txt", "r") as f:
    raw_text = f.read()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = text_splitter.split_text(raw_text)

documents = [
    Document(
        page_content=chunk,
        metadata={"source": "hackbattle_doc"}
    )
    for chunk in chunks
]


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


INDEX_PATH = "faq_index"

if os.path.exists(INDEX_PATH):
    vector_store = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(INDEX_PATH)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})


# llm = Llama(
#     model_path="/Users/Harsha/Library/Caches/llama.cpp/unsloth_gemma-3-270m-it-GGUF_gemma-3-270m-it-Q4_K_M.gguf",
#     temperature=0,
#     n_ctx=2048,
#     n_threads=8,
#     n_batch=256,
# )

llm = OllamaLLM(model="gemma3:270m", temperature=0)


def get_llm_response(user_query, relevant_docs=None):
    if relevant_docs is None:
        relevant_docs = retriever.invoke(user_query)  # ✅ use invoke instead of deprecated get_relevant_documents
        print(f"Retrieved {len(relevant_docs)} documents.")

    # Format context clearly
    context = ""
    for doc in relevant_docs:
        context += f"Q: {doc.metadata.get('question', '')}\nA: {doc.page_content}\n\n"

    prompt = f"""
You are a helpful assistant for IEEE-CS VIT. 
Below are some frequently asked questions and their answers. 
Answer the user's question using the context obtained from the document. 
- If anything related to query asked is present, use it to construct your answer.
- If the answer is not present, reply exactly: "I don't know."

Question context from document Context:
{context}

User's Question: {user_query}

Answer the users query accordingly:
"""

    response = llm.invoke(prompt)
    return response.strip()


answer_cache = {}

def check_cache(query):
    normalized_query = query.lower().strip()
    return answer_cache.get(normalized_query, None)

def update_cache(query, answer):
    normalized_query = query.lower().strip()
    answer_cache[normalized_query] = answer



def chatbot(user_query):
    cached_answer = check_cache(user_query)
    if cached_answer:
        return f"[CACHED] {cached_answer}"

    relevant_docs = retriever.get_relevant_documents(user_query)
    # print(relevant_docs)
    answer = get_llm_response(user_query, relevant_docs)
    update_cache(user_query, answer)
    return f"[FAQ] {answer}"



if __name__ == "__main__":
    print("FAQ RAG Chatbot (type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = chatbot(query)
        print(f"Bot: {response}\n")