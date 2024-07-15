import requests
from config import HUGGINGFACE_API_KEY

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}


def _api_call(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def zero_shot_classification(text, categories):
    categories = categories.split(",")
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": categories},
    }
    output = _api_call(payload)
    new_output = {}
    for i in range(len(output["labels"])):
        new_output[output["labels"][i]] = output["scores"][i]
    return new_output
