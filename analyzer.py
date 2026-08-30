import json
import streamlit as st
from groq import Groq


def analyze_review(review):

    prompt = f"""
You are an AI customer review analyst for a small business.

Analyze the following customer review.

Review:
"{review}"

Return ONLY valid JSON using exactly this structure:

{{
    "sentiment": "Positive/Neutral/Negative",
    "category": "Service/Product/Delivery/Staff/Price/Quality/Other",
    "urgency": "Low/Medium/High",
    "issue": "short description of the main issue"
}}
"""

    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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