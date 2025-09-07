# test_model_endpoint.py
import requests

# Replace this URL with your Modal deployment's endpoint
# BASE_URL = "https://hrshx3o5o6--chatbot-ollama-cpu-serve-chatbotapicpu-query.modal.run"  # e.g., https://abc123.modal.run

# The endpoint exposed in your modal_rag.py
QUERY_ENDPOINT = "https://hrshx3o5o6--chatbot-ollama-cpu-serve-chatbotapicpu-query.modal.run"

# Example question to test the model
payload = {
    "question": "what is hackbattle?"
}

# Optional: headers if your endpoint requires JSON
headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(QUERY_ENDPOINT, json=payload, headers=headers)
    response.raise_for_status()  # Raise an exception for HTTP errors
    data = response.json()
    print("Response from chatbot:")
    print(data)
except requests.exceptions.RequestException as e:
    print("Error connecting to the endpoint:", e)