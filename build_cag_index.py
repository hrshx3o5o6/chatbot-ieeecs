import json
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from llama_cpp import Llama



with open("cag_data/faq_raw.json", "r") as f:
    faq_data = json.load(f)


documents = [
    Document(
        page_content=f"Q: {item['title']}\nA: {item['text']}",
        metadata={"source": "faq", "question": item["title"]}
    )
    for item in faq_data
]


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


INDEX_PATH = "faq_index"

if os.path.exists(INDEX_PATH):
    vector_store = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(INDEX_PATH)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


llm = Llama(
    model_path="/Users/Harsha/Library/Caches/llama.cpp/unsloth_gemma-3-270m-it-GGUF_gemma-3-270m-it-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    n_batch=256,
)



def get_llm_response(user_query, relevant_docs=None):
    if relevant_docs is None:
        relevant_docs = retriever.get_relevant_documents(user_query)

    context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are a helpful assistant for IEEE-CS VIT.
Use ONLY the following context to answer the user's question.
If the answer is not in the context, reply exactly with: "I don't know."
STRICTLY avoid any additional information not present in the context. when you don't know something
then STRICTLY say "I don't know."

Context:
{context}

User: {user_query}
Assistant:"""

    output = llm(prompt, max_tokens=200, stop=["User:", "You:"], echo=False)
    return output["choices"][0]["text"].strip()


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
    if relevant_docs:
        answer = get_llm_response(user_query, relevant_docs)
        update_cache(user_query, answer)
        return f"[FAQ] {answer}"

    # fallback (unlikely, since retriever always returns something)
    llm_answer = get_llm_response(user_query, [])
    update_cache(user_query, llm_answer)
    return f"[LLM] {llm_answer}"



if __name__ == "__main__":
    print("FAQ RAG Chatbot (type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = chatbot(query)
        print(f"Bot: {response}\n")