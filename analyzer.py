import json
import streamlit as st
from huggingface_hub import InferenceClient


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

    client = InferenceClient(
        api_key=st.secrets["HF_TOKEN"],
        provider="auto"
    )

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=[
            {
                "role": "system",
                "content": "You are a customer review analysis AI. Always return valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    result = response.choices[0].message.content

    try:
        return json.loads(result)

    except json.JSONDecodeError:
        return {
            "sentiment": "Unknown",
            "category": "Other",
            "urgency": "Unknown",
            "issue": result
        }