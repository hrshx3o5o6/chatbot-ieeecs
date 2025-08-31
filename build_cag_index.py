import json
from rapidfuzz import process, fuzz
# from llama_cpp import Llama
import spacy 
from langchain_ollama import OllamaLLM


nlp = spacy.load("en_core_web_sm")

with open("cag_data/faq_raw.json", "r") as f:
    faq_data = json.load(f)

faq_questions = [item["title"] for item in faq_data]

answer_cache = {}

def extract_entities(query):
    """using spacy doing NER ( named entity recognition)"""
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]
    print("[DEBUG] Extracted entities:", entities)
    return entities

def get_faq_answer(user_query, threshold=60):
    """Check all FAQ with NER + fuzzy matching."""
    entities = extract_entities(user_query)
    filtered_questions = faq_questions
    print("[DEBUG] Using filtered_questions (before entity filtering):", filtered_questions)

    # If entities exist, filter FAQs first
    if entities:
        filtered_questions = [
            item["title"] for item in faq_data
            if any(ent.lower() in item["title"].lower() or ent.lower() in item["text"].lower()
                   for ent in entities)
        ]
        print("[DEBUG] Using filtered_questions (after entity filtering):", filtered_questions)
        # If nothing matched entity filter, fallback to full FAQ
        if not filtered_questions:
            filtered_questions = faq_questions
            print("[DEBUG] No entity matches found, falling back to full FAQ list.")

    print("[DEBUG] Performing fuzzy matches...")
    matches = process.extract(
        user_query, filtered_questions, scorer=fuzz.token_set_ratio, limit=5, score_cutoff=threshold
    )

    print("[DEBUG] Matches found:", matches)
    results = []
    for match, score, idx in matches:
        results.append({
            "question": faq_data[idx]["title"],
            "text": faq_data[idx]["text"],
            "score": score
        })

    print("FAQ Matches:", results)
    if results:
        print("[DEBUG] Using FAQ context for LLM.")
        return get_llm_response_with_context(user_query, results)
    else:
        print("[DEBUG] Falling back to plain LLM without context")
        return get_llm_response(user_query)



llm = OllamaLLM(model="gemma3:270m", temperature=0)

def get_llm_response(user_query):
    """Query Gemma model for a response."""
    fewShot_examples = """
    Example Conversation:
        User: What is IEEE-CS?
        You: IEEE-CS is a student chapter at VIT Vellore focused on computer science and engineering.
        User: How can I join IEEE-CS?
        You: You can join IEEE-CS by visiting our website and filling out the membership form.
        User: What events does IEEE-CS organize?
        You: IEEE-CS organizes workshops, hackathons, guest lectures, and coding competitions.
        User: Who can I contact for more information?
        You: You can contact the IEEE-CS committee via email at ieee-cs@vit.ac.in.
        User: What is IEEE Computer Society at VIT?
        You: IEEE Computer Society, VIT is a student branch of the Madras Section of IEEE Region 10. It was formed in February 2012 and organizes workshops, hackathons, and technical events to empower innovation among students.
        User: What is Assignofast by IEEE-CS VIT?
        You: Assignofast is a smart app/extension developed by IEEE-CS VIT. It syncs assignments and timetables from VTOP and sends timely reminders to help students manage deadlines.
        User: What are the benefits of joining IEEE-CS VIT?
        You: Members get opportunities to participate in exclusive technical workshops, hackathons, industry collaborations, and access to IEEE resources for career development.
    """
    prompt = f"You are a helpful assistant for a chapter at VIT Vellore called IEEE-CS that answers FAQs and engages in conversation with the user. Consider the following examples of past interactions:\n{fewShot_examples}\nNow, answer the user's question concisely in 2 lines:\nUser: {user_query}\nYou:\n You should not make up answers. If you don't know, say 'I don't know.' Don't Speak anything bad or use words in a negative sense"
    print("[DEBUG] Invoking LLM without FAQ context")
    output = llm.invoke(prompt)
    return output.strip()

def get_llm_response_with_context(user_query, relevant_faqs):
    """Query Gemma model with context from relevant FAQs."""
    prompt = f"You are a helpful FAQ assistant for a chapter at VIT Vellore called IEEE-CS that answers FAQs and engages in conversation with the user. Use ONLY the following context to answer the user's question. If the answer is not in the context, say 'I don't know.' The answer should be concise and in 2 lines.\n\nUser's Question: {user_query}\n\nRelevant FAQs:\n"
    for faq in relevant_faqs:
        prompt += f"- {faq['question']}: {faq['text']}\n"
    prompt += f"\nAnswer the user's question based solely on the above context:\nUser: {user_query}\nAssistant:"
    print("[DEBUG] Invoking LLM with FAQ context")
    output = llm.invoke(prompt)
    return output.strip()

def check_cache(query):
    """Check if query is already cached (exact match)."""
    normalized_query = query.lower().strip()
    return answer_cache.get(normalized_query, None)

def update_cache(query, answer):
    """Save new answer into cache."""
    normalized_query = query.lower().strip()
    answer_cache[normalized_query] = answer

def chatbot(user_query):
    
    print("[DEBUG] Checking cache for query...")
    cached_answer = check_cache(user_query)
    if cached_answer:
        print("[DEBUG] Cache hit")
        return f"[CACHED] {cached_answer}"

    print("[DEBUG] Attempting to answer using FAQ logic...")
    faq_answer = get_faq_answer(user_query)
    if faq_answer:
        print("[DEBUG] FAQ answer found, updating cache.")
        update_cache(user_query, faq_answer)
        return f"[FAQ] {faq_answer}"

    print("[DEBUG] Falling back to LLM.")
    llm_answer = get_llm_response(user_query)
    update_cache(user_query, llm_answer)
    return f"[LLM] {llm_answer}"


if __name__ == "__main__":
    print("FAQ + Conversational Chatbot (type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = chatbot(query)
        print(f"Bot: {response}\n")