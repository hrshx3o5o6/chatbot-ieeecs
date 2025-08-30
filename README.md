# FAQ RAG Chatbot

This project is a chatbot that answers FAQs using a hybrid of Cache-Augmented Generation (CAG) and Retrieval-Augmented Generation (RAG). It is designed for IEEE-CS VIT FAQs but can be adapted to other contexts.

---

## How It Works

### 1. Data Preparation
- The raw text from `hackbattle_doc.txt` is read into memory.
- The text is split into chunks using `RecursiveCharacterTextSplitter` (chunk size = 200, overlap = 50).
- Each chunk is converted into a `Document` with metadata.

### 2. Embedding and Indexing
- HuggingFace sentence transformer (`all-MiniLM-L6-v2`) is used to generate embeddings for each chunk.
- These embeddings are stored in a FAISS vector database.
- On subsequent runs, the FAISS index is loaded from disk instead of recomputing.

### 3. Retrieval
- A retriever is created from the FAISS vector store.
- For each user query, the retriever fetches the top-k most relevant chunks (k = 10).

### 4. Answer Generation
- If the query has been seen before, the cached answer is returned.
- Otherwise:
  - The retrieved chunks are formatted into a context prompt.
  - The user query and context are passed to a local LLM (Gemma via Ollama).
  - The LLM generates an answer using only the provided context.
  - If no relevant context exists, the model is instructed to reply: `"I don't know."`
- The new answer is cached for future queries.

### 5. CLI Chat
- The chatbot runs in a terminal loop.
- Type a question to receive an answer.
- Type `exit` or `quit` to end the session.

---

## Key Features
- **RAG-based answering**: Uses document context for accuracy.
- **Cache-Augmented Generation (CAG)**: Previously asked queries are stored and served instantly.
- **Local inference**: Runs with a local Gemma model using Ollama.
- **Reusable FAISS index**: Saves embeddings for fast reloading.

---

## File Overview
- `build_cag_index.py`: Main script for building the index and running the chatbot.
- `hackbattle_doc.txt`: Source text file containing FAQ-related content.
- `faq_index/`: Directory where the FAISS vector store is saved.

---

## Running the Chatbot
1. Ensure Ollama is installed and the Gemma model is available locally.
2. Run the script:
   ```python3 build_cag_index.py ```
3. ask your question in the terminal

---
### Architecture flow
<img width="463" height="628" alt="image" src="https://github.com/user-attachments/assets/b7492c08-8661-4609-ba1d-e0096c9cf769" />
   
