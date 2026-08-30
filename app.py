import streamlit as st
import pandas as pd
import json
import os
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Review Management Agent",
    page_icon="⭐",
    layout="wide"
)


# ============================================================
# GROQ AI FUNCTION
# ============================================================

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

Rules:
- sentiment must be Positive, Neutral, or Negative.
- category must be Service, Product, Delivery, Staff, Price, Quality, or Other.
- urgency must be Low, Medium, or High.
- issue must be a short description.
"""

    try:

        api_key = st.secrets["GROQ_API_KEY"]

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a customer review analysis AI. "
                        "Always return valid JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        result = response.choices[0].message.content

        # Remove possible markdown code fences
        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

        return json.loads(result)

    except Exception as e:

        return {
            "sentiment": "Unknown",
            "category": "Other",
            "urgency": "Unknown",
            "issue": f"AI analysis error: {str(e)}"
        }


# ============================================================
# AI RESPONSE GENERATOR
# ============================================================

def generate_response(review, analysis):

    prompt = f"""
You are a professional customer service manager.

Write a polite, professional and personalized response to this
customer review.

Customer review:
"{review}"

Analysis:
Sentiment: {analysis.get("sentiment", "Unknown")}
Category: {analysis.get("category", "Other")}
Urgency: {analysis.get("urgency", "Unknown")}
Issue: {analysis.get("issue", "None")}

Rules:
- If the review is positive, thank the customer warmly.
- If the review is negative, apologize appropriately and offer help.
- If the review is neutral, respond professionally.
- Do not invent refunds, discounts or policies.
- Keep the response between 40 and 100 words.
- Do not mention AI.
"""

    try:

        api_key = st.secrets["GROQ_API_KEY"]

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional customer service representative."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4
        )

        return response.choices[0].message.content.strip()

    except Exception as e:

        return f"Unable to generate response: {str(e)}"


# ============================================================
# SAVE REVIEW TO CSV
# ============================================================

def save_review(review, analysis, response):

    file_name = "analyzed_reviews.csv"

    new_data = pd.DataFrame([
        {
            "Review": review,
            "Sentiment": analysis.get("sentiment", ""),
            "Category": analysis.get("category", ""),
            "Urgency": analysis.get("urgency", ""),
            "Issue": analysis.get("issue", ""),
            "AI Response": response
        }
    ])

    try:

        if os.path.exists(file_name):

            old_data = pd.read_csv(file_name)

            combined = pd.concat(
                [old_data, new_data],
                ignore_index=True
            )

            combined.to_csv(
                file_name,
                index=False
            )

        else:

            new_data.to_csv(
                file_name,
                index=False
            )

    except Exception:
        pass


# ============================================================
# TITLE
# ============================================================

st.title("⭐ AI Review Management Agent")

st.write(
    "Analyze customer reviews using AI, identify important issues, "
    "detect urgency, and generate professional responses."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 Review Management")

    st.write(
        "This AI agent helps small businesses understand and "
        "respond to customer reviews."
    )

    st.divider()

    st.info(
        "AI Model: Llama 3.3 70B\n\n"
        "Powered by Groq"
    )


# ============================================================
# REVIEW INPUT
# ============================================================

st.header("📝 Analyze Customer Review")

review = st.text_area(
    "Enter a customer review:",
    placeholder=(
        "Example: The food was excellent, but the delivery was "
        "very late and customer service did not respond."
    ),
    height=160
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Review",
    type="primary",
    use_container_width=True
):

    if not review.strip():

        st.warning("Please enter a customer review first.")

    else:

        with st.spinner("🤖 AI is analyzing the review..."):

            analysis = analyze_review(review)

        # Store analysis
        st.session_state["analysis"] = analysis
        st.session_state["review"] = review

        # Generate response
        with st.spinner("✍️ Generating customer response..."):

            ai_response = generate_response(
                review,
                analysis
            )

        st.session_state["ai_response"] = ai_response

        # Save
        save_review(
            review,
            analysis,
            ai_response
        )

        st.success("Review analyzed successfully!")


# ============================================================
# DISPLAY ANALYSIS
# ============================================================

if "analysis" in st.session_state:

    analysis = st.session_state["analysis"]

    st.divider()

    st.header("📊 AI Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Sentiment",
            analysis.get(
                "sentiment",
                "Unknown"
            )
        )

    with col2:

        st.metric(
            "Category",
            analysis.get(
                "category",
                "Other"
            )
        )

    with col3:

        st.metric(
            "Urgency",
            analysis.get(
                "urgency",
                "Unknown"
            )
        )

    st.subheader("🔎 Main Issue")

    st.info(
        analysis.get(
            "issue",
            "No issue identified."
        )
    )


# ============================================================
# AI RESPONSE
# ============================================================

if "ai_response" in st.session_state:

    st.divider()

    st.header("💬 Suggested Customer Response")

    st.text_area(
        "AI-generated response:",
        value=st.session_state["ai_response"],
        height=180
    )


# ============================================================
# REVIEW HISTORY
# ============================================================

st.divider()

st.header("📚 Review History")

file_name = "analyzed_reviews.csv"

if os.path.exists(file_name):

    try:

        history = pd.read_csv(file_name)

        if len(history) > 0:

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("📈 Review Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Total Reviews",
                    len(history)
                )

            with col2:

                if "Sentiment" in history.columns:

                    negative_count = (
                        history["Sentiment"]
                        .astype(str)
                        .str.lower()
                        .eq("negative")
                        .sum()
                    )

                    st.metric(
                        "Negative Reviews",
                        int(negative_count)
                    )

            with col3:

                if "Urgency" in history.columns:

                    high_count = (
                        history["Urgency"]
                        .astype(str)
                        .str.lower()
                        .eq("high")
                        .sum()
                    )

                    st.metric(
                        "High Urgency",
                        int(high_count)
                    )

        else:

            st.info("No reviews analyzed yet.")

    except Exception:

        st.info("Review history is currently unavailable.")

else:

    st.info(
        "No review history yet. Analyze your first review above."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Review Management Agent • Powered by Groq"
)