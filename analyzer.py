import ollama
import json


def analyze_review(review):

    prompt = f"""
You are an AI customer review analyst for a small business.

Analyze the following customer review.

Review:
"{review}"

Return ONLY valid JSON using this exact structure:

{{
    "sentiment": "Positive/Neutral/Negative",
    "category": "Service/Product/Delivery/Staff/Price/Quality/Other",
    "urgency": "Low/Medium/High",
    "issue": "short description of the main issue"
}}
"""

    response = ollama.chat(
        model="gemma2:2b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response["message"]["content"]

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {
            "sentiment": "Unknown",
            "category": "Other",
            "urgency": "Unknown",
            "issue": result
        }