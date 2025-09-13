import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# ---------------- Directory ----------------
DOC_DIR = "/Users/Harsha/Downloads/Harsha_Stuff/Internship_Clubs_papers_hacks/IEEE/chatBot/hackbattle_docs"  # path to your documents
# Example questions mapping (per doc)
questions = {
    "description.txt": [
    "What is HackBattle?",
    "How long does HackBattle last?",
    "Who organizes HackBattle?",
    "Why is HackBattle considered fresher-friendly?",
    "What makes HackBattle unique?",
    "How many editions of HackBattle have been held?",
],
    "eventDetails.txt": [
    "How many participants can join HackBattle?",
    "What is the entry fee for HackBattle?",
    "What is the team size for HackBattle?",
    "When is HackBattle conducted?",
    "How many participants are there in total?"
],
    "faq.txt": [
        "What is HackBattle?",
        "Do I need Minecraft to join?",
        "How do I register?",
        "Is HackBattle fresher-friendly?",
        "What should I bring?",
        "Is there a team size limit?",
        "Can I eat snacks inside the venue if I get hungry?",
        "Can we leave for quiz?", 
        "Can we leave at night and go to hostels?", 
        "What if i don't have a team?", 
        "Will OD be provided?", 
        "Is there OD for the hackathon?"
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
    "Can participants be from any background or discipline?",
    "When must the project work be completed?",
    "Are pre-built projects or existing codebases allowed?",
    "What code of conduct must participants follow?",
    "Are participants required to stay within the venue?",
    "What items must participants bring to the event?",
    "Will internet and infrastructure be provided?",
    "What must the final submission include?",
    "Can organizers modify rules and schedules?",
    "Do teams allow organizers to showcase their projects by participating?"
],
    "uniqueAspects.txt": [
    "Why is HackBattle beginner friendly?",
    "What technologies can freshers learn at HackBattle?",
    "How does HackBattle add career value for participants?",
    "Can HackBattle projects be showcased on LinkedIn or portfolios?",
    "Does HackBattle help with internships and placements?",
    "How many years has HackBattle been running?",

]
}

# ---------------- Load Documents ----------------
docs = {}
for filename in os.listdir(DOC_DIR):
    if filename.endswith(".txt"):  # or any other file type
        with open(os.path.join(DOC_DIR, filename), "r", encoding="utf-8") as f:
            docs[filename] = f.read()

# ---------------- Chunking ----------------
chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_chunks = []

for doc_name, content in docs.items():
    doc_chunks = chunker.split_text(content)
    doc_questions = questions.get(doc_name, [])
    for i, chunk in enumerate(doc_chunks):

        text_for_embedding = f"{chunk}\nQuestions: {'; '.join(doc_questions)}"
        metadata = {
            "source": doc_name,
            "chunk_id": i,
            "questions": questions.get(doc_name, [])
        }
        all_chunks.append({"text": chunk, "metadata": metadata})

# ---------------- Embeddings ----------------
embedding_model = HuggingFaceEmbeddings(model_name="BAAI_bge-large-en-v1.5")

texts = [c["text"] for c in all_chunks]
metadatas = [c["metadata"] for c in all_chunks]

# ---------------- Create FAISS Index ----------------
faiss_index = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)

# ---------------- Save Locally ----------------
faiss_index.save_local("faiss_bge_index")

print("FAISS index created and saved locally from directory!")