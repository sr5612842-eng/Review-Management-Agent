import ollama


def generate_response(review, analysis):

    prompt = f"""
You are a professional customer service manager.

Write a personalized response to this customer review.

Customer Review:
"{review}"

Analysis:
Sentiment: {analysis['sentiment']}
Category: {analysis['category']}
Urgency: {analysis['urgency']}
Issue: {analysis['issue']}

Rules:
- Be polite and professional.
- Do not sound robotic.
- If the review is negative, apologize appropriately.
- Do not make promises the business cannot guarantee.
- Keep the response between 50 and 100 words.
- Do not use hashtags.
- Address the customer's specific problem.
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

    return response["message"]["content"]