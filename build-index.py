# build-index.py
from pathlib import Path
import os
import modal
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Note: every time you make changes to the txt files, make sure to change the references for the same in new_chunk_modal.py.

# ---------------- Directories ----------------
MODEL_DIR = Path("/models")  # persistent volume for FAISS index
     # mounted path inside container

volume = modal.Volume.from_name("embedding-model-vol", create_if_missing=True)
DOC_DIR = MODEL_DIR / "hackbattle_docs_new_4" # change this every time you update the txt files

# ---------------- Modal Image ----------------
image = (
    modal.Image.debian_slim()
    .pip_install(
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "sentence-transformers",
        "faiss-cpu"
    )
)

# ---------------- Modal App ----------------
app = modal.App("build-faiss-index", image=image, volumes={MODEL_DIR: volume})

# ---------------- Example Questions Mapping ----------------
questions = {
    "description.txt": [
    "What is HackBattle?",
    "How long does HackBattle last?",
    "Who organizes HackBattle?",
    "Why is HackBattle considered fresher-friendly?",
    "Can we leave for quiz?",
    "Can we leave at night and go to hostels?",
    "What if I don't have a team?",
    "When is hackbattle?",
    "What date is the event?",
    "How do I register?",
    "Will OD be provided?",
    "Is there OD for the hackathon?"
],
    "eventDetails.txt": [
    "Where is Hackbattle",
    "How much should I pay to enter?",
    "What is the entry fee for HackBattle?",
    "What is the team size for HackBattle?",
    "What date is the event?"
    
],
    "faq.txt": [
        "What is HackBattle?",
        "How do I register?",
        "Is HackBattle fresher-friendly?",
        "What should I bring?",
        "Is there a team size limit?"
    ],
    "ieeeCompSoc.txt": [
    "What does IEEE-CS VIT focus on?",
    "Is IEEE-CS VIT recognized globally?",
    "What is ARCS?",
],
    "judgeCrit.txt": [
    "What are the main stages of HackBattle judging?",
    "What does the first review evaluate?",
    "What does the second review focus on?",
    "What does the final pitch criteria evaluate?",
],
    "rulesAndReg.txt": [
    "Do participants need to officially register their team?",
    "Can participants be from any branch",
    "Can I eat snacks inside the venue if I get hungry?",
    "When must the project work be completed?",
    "Are pre-built projects or existing codebases allowed?",
    "What rules must participants follow?",
    "Are participants required to stay within the venue?",
    "What items must participants bring to the event?",
    "Will internet be provided?",
    "What must the final submission include?"
],
    "uniqueAspects.txt": [
    "Why is HackBattle beginner friendly?",
    "What do i need to know for me to participate"
    "How does HackBattle add career value for participants?",
    "Can HackBattle projects be showcased on LinkedIn or portfolios?",
    "Does HackBattle help with internships and placements?",
    "How many years has HackBattle been running?",

]
}


@app.function(
    volumes={MODEL_DIR: volume},
)
def build():
    print("🔄 Loading documents from:", DOC_DIR)

    docs = {}
    for filename in os.listdir(DOC_DIR):
        if filename.endswith(".txt"):
            with open(DOC_DIR / filename, "r", encoding="utf-8") as f:
                docs[filename] = f.read()

    print(f"✅ Loaded {len(docs)} documents")

    # ---------------- Chunking ----------------
    chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []

    for doc_name, content in docs.items():
        doc_chunks = chunker.split_text(content)
        doc_questions = questions.get(doc_name, [])
        for i, chunk in enumerate(doc_chunks):
            metadata = {
                "source": doc_name,
                "chunk_id": i,
                "questions": doc_questions,
            }
            all_chunks.append({"text": chunk, "metadata": metadata})

    print(f"✅ Created {len(all_chunks)} chunks with metadata")

    # ---------------- Embeddings ----------------
    print("🔄 Initializing embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    # ---------------- Create FAISS Index ----------------
    print("🔄 Building FAISS index...")
    faiss_index = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)

    index_path = MODEL_DIR / "faiss_bge_index_newdocs_4" # change this every time you update the txt files, this is where the FAISS index is stored.
    faiss_index.save_local(str(index_path))

    print(f"✅ FAISS index saved to {index_path}")