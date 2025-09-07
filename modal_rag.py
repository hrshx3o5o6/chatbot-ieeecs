import os
from pathlib import Path
import subprocess
import time
import modal
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# ---------------- Modal Volume ----------------
volume = modal.Volume.from_name("embedding-model-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

# ---------------- Modal Image ----------------
image = (
    modal.Image.debian_slim()
    .apt_install("curl", "git", "procps")  # procps needed for `ps` commands
    .run_commands([
        "curl -fsSL https://ollama.com/install.sh | bash"
    ])
    .pip_install(
        "langchain",
        "langchain-community",
        "langchain-ollama",
        "langchain-huggingface",
        "sentence-transformers",
        "fastapi[all]",
        "faiss-cpu"
    )
)

app = modal.App("chatbot-ollama-cpu-serve", image=image, volumes={MODEL_DIR: volume})

@app.cls(image=image, volumes={MODEL_DIR: volume})
class ChatbotAPIcpu:
    @modal.enter()
    def load_models(self):
        print("🔄 Loading embeddings and FAISS index...")

        # 1️⃣ Load embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5"
        )

        # 2️⃣ Load FAISS index
        index_path = MODEL_DIR / "faiss_index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.vector_store = FAISS.load_local(index_path, self.embedding_model, allow_dangerous_deserialization=True)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 7})
        print("✅ FAISS index loaded")

        # 3️⃣ Start Ollama server first
        print("🔄 Starting Ollama server...")
        self.serve_proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "OLLAMA_MODELS": str(MODEL_DIR)}
        )

        # wait a few seconds for the server to start
        time.sleep(5)
        print("✅ Ollama server started")

        # 4️⃣ Pull model if not exists
        model_path = MODEL_DIR / "llama3.2:3b"
        if not model_path.exists():
            print("🔄 Pulling gemma3:1b into volume...")
            subprocess.run(["ollama", "pull", "llama3.2:3b"], check=True)
            print("✅ Model pulled")

        # 5️⃣ Connect LangChain OllamaLLM to local server
        self.llm = OllamaLLM(model="llama3.2:3b", temperature=0)
        print("✅ Ollama LLM ready")

        # Cache
        self.answer_cache = {}

    # ---------------- Cache Helpers ----------------
    def check_cache(self, query):
        return self.answer_cache.get(query.lower().strip(), None)

    def update_cache(self, query, answer):
        if answer.strip() != "I don't know":
            self.answer_cache[query.lower().strip()] = answer

    # ---------------- Generate Response ----------------
    def get_llm_response(self, user_query, relevant_docs):
        context = ""
        for doc in relevant_docs:
            context += f"A: {doc.page_content}\n\n"

        prompt = f"""
You are a helpful assistant for IEEE-CS VIT HackBattle FAQ.
Answer the user's question using ONLY the context below.
Do NOT include any information outside this context.
Always respond in complete sentences using proper English.
You may rephrase or simplify the answer to make it clear and user-friendly.
Do NOT guess, infer, or add information that is not explicitly in the context.
If the answer is not in the context, reply exactly: "I don't know".

Context from documents:
{context}

User's Question: {user_query}

Answer:
"""
        response = self.llm.invoke(prompt)
        return response.strip()

    # ---------------- FastAPI Endpoint ----------------
    class QueryRequest(BaseModel):
        question: str

    @modal.fastapi_endpoint(method="POST", docs=True, requires_proxy_auth=False)
    async def query(self, request: QueryRequest):
        question = request.question
        cached = self.check_cache(question)
        if cached:
            return JSONResponse({"answer": cached, "cached": True})

        docs_and_scores = self.vector_store.similarity_search_with_score(question, k=7)
        filtered_docs = [doc for doc, score in docs_and_scores if score >= 0.30]

        if not filtered_docs:
            answer = "I don't know"
        else:
            answer = self.get_llm_response(question, filtered_docs)

        self.update_cache(question, answer)
        return JSONResponse({"answer": answer, "cached": False})