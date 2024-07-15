import base64
from config import HUGGINGFACE_API_KEY

import requests

API_URL = "https://api-inference.huggingface.co/models/impira/layoutlm-document-qa"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def _api_call(payload):
    with open(payload["inputs"]["image"], "rb") as f:
        img = f.read()
        payload["inputs"]["image"] = base64.b64encode(img).decode("utf-8")
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def document_qna(image_url, question):
    output = _api_call({
        "inputs": {
            "image": image_url,
            "question": question
        },
        "wait_for_model": True})

    output = output[0].get("answer")
    return output
