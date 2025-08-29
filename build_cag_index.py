import json
from rapidfuzz import process, fuzz
from llama_cpp import Llama

# -------------------------------
# Load FAQ Data
# -------------------------------
with open("cag_data/faq_raw.json", "r") as f:
    faq_data = json.load(f)

faq_questions = [item["title"] for item in faq_data]

def get_faq_answer(user_query, threshold=60):
    """Check FAQ cache first."""
    match, score, idx = process.extractOne(
        user_query, faq_questions, scorer=fuzz.QRatio
    )
    if score >= threshold:
        return faq_data[idx]["text"]
    return None

# -------------------------------
# Load Local Gemma Model
# -------------------------------
llm = Llama(
    model_path="/Users/Harsha/Library/Caches/llama.cpp/unsloth_gemma-3-270m-it-GGUF_gemma-3-270m-it-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,   # adjust based on your CPU
    n_batch=256,   # adjust for performance
)

def get_llm_response(user_query):
    """Query Gemma model for a response."""
    prompt = f"You are a helpful assistant for a chapter at VIT Vellore called IEEE-CS that answers FAQs and engages in conversation with the user. The user asked: {user_query}\nAnswer:"
    output = llm(prompt, max_tokens=256, stop=["User:", "You:"], echo=False)
    return output["choices"][0]["text"].strip()

# -------------------------------
# Hybrid CAG Chatbot
# -------------------------------
def chatbot(user_query):
    faq_answer = get_faq_answer(user_query)
    if faq_answer:
        return faq_answer  # Cache hit (FAQ)
    else:
        return get_llm_response(user_query)  # Fall back to LLM

# -------------------------------
# CLI Loop
# -------------------------------
if __name__ == "__main__":
    print("FAQ + Conversational Chatbot (type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = chatbot(query)
        print(f"Bot: {response}\n")