import requests
from config import HUGGINGFACE_API_KEY
API_URL = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}


def _api_call(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def ner(query):
    output = _api_call({"inputs": query,"wait_for_model": True})
    output = [(i["word"],i["entity_group"]) for i in output]

    return output