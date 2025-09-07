# build-index.py
from pathlib import Path
import modal
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

# Volume (persistent storage for the FAISS index)
MODEL_DIR = Path("/models")
volume = modal.Volume.from_name("embedding-model-vol", create_if_missing=True)

# Define container image
image = (
    modal.Image.debian_slim()
    .pip_install("langchain", "langchain-community", "langchain-huggingface", "sentence-transformers", "faiss-cpu")
)

# Modal app
app = modal.App("build-faiss-index", image=image, volumes={MODEL_DIR: volume})

@app.function()
def build():
    print("🔄 Loading document...")
    loader = TextLoader("/models/models/hackbattle_doc_cleaned.txt")
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents")

    print("🔄 Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    print("🔄 Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    index_path = MODEL_DIR / "faiss_index"
    vectorstore.save_local(str(index_path))
    print(f"✅ FAISS index saved to {index_path}")