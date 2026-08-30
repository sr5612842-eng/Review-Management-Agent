import streamlit as st
import pandas as pd
import json
import os
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Review Management Agent",
    page_icon="⭐",
    layout="wide"
)


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():
    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


# ============================================================
# ANALYZE REVIEW
# ============================================================

def analyze_review(review):

    prompt = f"""
You are an AI customer review analyst for a small business.

Analyze this customer review:

"{review}"

Return ONLY valid JSON.

Use exactly this structure:

{{
    "sentiment": "Positive/Neutral/Negative",
    "category": "Service/Product/Delivery/Staff/Price/Quality/Other",
    "urgency": "Low/Medium/High",
    "issue": "short description of the main issue"
}}

Rules:
- Sentiment must be Positive, Neutral, or Negative.
- Category must be Service, Product, Delivery, Staff, Price, Quality, or Other.
- Urgency must be Low, Medium, or High.
- Issue must be short and clear.
"""

    try:

        client = get_groq_client()

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a customer review analysis AI. "
                        "Return valid JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        # Remove markdown if model adds it
        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

        return json.loads(result)

    except Exception as e:

        return {
            "sentiment": "Unknown",
            "category": "Other",
            "urgency": "Unknown",
            "issue": f"Analysis error: {str(e)}"
        }


# ============================================================
# GENERATE AI CUSTOMER REPLY
# ============================================================

def generate_response(review, analysis):

    prompt = f"""
You are a professional customer service representative.

Write a personalized response to this customer review.

CUSTOMER REVIEW:
"{review}"

ANALYSIS:
Sentiment: {analysis.get("sentiment", "Unknown")}
Category: {analysis.get("category", "Other")}
Urgency: {analysis.get("urgency", "Unknown")}
Issue: {analysis.get("issue", "None")}

Instructions:

1. If the review is positive:
   - Thank the customer.
   - Appreciate their feedback.
   - Keep the tone warm and friendly.

2. If the review is negative:
   - Apologize for the problem.
   - Acknowledge the customer's concern.
   - Say that the business will look into the issue.
   - Be professional and empathetic.

3. If the review is neutral:
   - Thank the customer.
   - Address their feedback professionally.

4. Do NOT:
   - Mention AI.
   - Invent refunds.
   - Invent discounts.
   - Invent company policies.
   - Make promises you cannot verify.

5. Keep the reply between 40 and 80 words.

Return ONLY the customer-facing response.
"""

    try:

        client = get_groq_client()

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional and empathetic "
                        "customer service representative."
                    )
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

        return f"Unable to generate AI reply: {str(e)}"


# ============================================================
# SAVE REVIEW
# ============================================================

def save_review(review, analysis, ai_response):

    file_name = "analyzed_reviews.csv"

    new_review = pd.DataFrame([
        {
            "Review": review,
            "Sentiment": analysis.get("sentiment", ""),
            "Category": analysis.get("category", ""),
            "Urgency": analysis.get("urgency", ""),
            "Issue": analysis.get("issue", ""),
            "AI Response": ai_response
        }
    ])

    try:

        if os.path.exists(file_name):

            old_reviews = pd.read_csv(file_name)

            all_reviews = pd.concat(
                [old_reviews, new_review],
                ignore_index=True
            )

            all_reviews.to_csv(
                file_name,
                index=False
            )

        else:

            new_review.to_csv(
                file_name,
                index=False
            )

    except Exception:
        pass


# ============================================================
# HEADER
# ============================================================

st.title("⭐ AI Review Management Agent")

st.write(
    "An AI-powered tool that analyzes customer reviews, "
    "detects important issues, identifies urgency, and "
    "generates professional customer responses."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 AI Review Agent")

    st.write(
        "This system helps small businesses manage customer "
        "reviews quickly and professionally."
    )

    st.divider()

    st.write("### AI Features")

    st.write("✅ Sentiment Analysis")
    st.write("✅ Review Categorization")
    st.write("✅ Urgency Detection")
    st.write("✅ Issue Detection")
    st.write("✅ AI Response Generation")

    st.divider()

    st.caption("Powered by Groq AI")


# ============================================================
# REVIEW INPUT
# ============================================================

st.header("📝 Customer Review")

review = st.text_area(
    "Enter a customer review below:",
    placeholder=(
        "Example: The food was excellent, but my delivery "
        "was two hours late and customer service did not respond."
    ),
    height=180
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

        st.warning(
            "⚠️ Please enter a customer review first."
        )

    else:

        # ----------------------------------------------------
        # STEP 1: ANALYZE REVIEW
        # ----------------------------------------------------

        with st.spinner(
            "🤖 AI is analyzing the customer review..."
        ):

            analysis = analyze_review(review)

        # Save analysis in session
        st.session_state["review"] = review
        st.session_state["analysis"] = analysis

        # ----------------------------------------------------
        # STEP 2: GENERATE CUSTOMER RESPONSE
        # ----------------------------------------------------

        with st.spinner(
            "✍️ AI is generating a customer response..."
        ):

            ai_response = generate_response(
                review,
                analysis
            )

        st.session_state["ai_response"] = ai_response

        # ----------------------------------------------------
        # STEP 3: SAVE
        # ----------------------------------------------------

        save_review(
            review,
            analysis,
            ai_response
        )

        st.success(
            "✅ Review analyzed and AI response generated!"
        )


# ============================================================
# ANALYSIS RESULTS
# ============================================================

if "analysis" in st.session_state:

    analysis = st.session_state["analysis"]

    st.divider()

    st.header("📊 AI Analysis")


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "😊 Sentiment",
            analysis.get(
                "sentiment",
                "Unknown"
            )
        )

    with col2:

        st.metric(
            "📂 Category",
            analysis.get(
                "category",
                "Other"
            )
        )

    with col3:

        st.metric(
            "🚨 Urgency",
            analysis.get(
                "urgency",
                "Unknown"
            )
        )


    # --------------------------------------------------------
    # ISSUE
    # --------------------------------------------------------

    st.subheader("🔎 Main Issue")

    st.info(
        analysis.get(
            "issue",
            "No issue detected."
        )
    )


# ============================================================
# AI GENERATED RESPONSE
# ============================================================

if "ai_response" in st.session_state:

    st.divider()

    st.header("💬 AI Generated Reply")

    st.write(
        "Suggested response to the customer:"
    )

    st.text_area(
        "Customer Response",
        value=st.session_state["ai_response"],
        height=180,
        label_visibility="collapsed"
    )

    st.success(
        "✅ Response generated successfully."
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

        if not history.empty:

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            st.subheader("📈 Review Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Total Reviews",
                    len(history)
                )

            with col2:

                if "Sentiment" in history.columns:

                    negative_reviews = (
                        history["Sentiment"]
                        .astype(str)
                        .str.lower()
                        .eq("negative")
                        .sum()
                    )

                    st.metric(
                        "Negative Reviews",
                        int(negative_reviews)
                    )

            with col3:

                if "Urgency" in history.columns:

                    high_urgency = (
                        history["Urgency"]
                        .astype(str)
                        .str.lower()
                        .eq("high")
                        .sum()
                    )

                    st.metric(
                        "High Urgency",
                        int(high_urgency)
                    )

        else:

            st.info(
                "No reviews analyzed yet."
            )

    except Exception as e:

        st.warning(
            "Unable to load review history."
        )

else:

    st.info(
        "No review history yet. Analyze your first review above."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⭐ AI Review Management Agent | "
    "Sentiment • Category • Urgency • AI Response"
)