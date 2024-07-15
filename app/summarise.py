import requests
from config import HUGGINGFACE_API_KEY

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}


def _api_call(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def summarise(query):
    output = _api_call({"inputs": query,"wait_for_model": True})
    output = output[0].get("summary_text")
    return output

