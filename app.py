import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Review Management Agent",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        rgba(70, 90, 120, 0.35),
        rgba(30, 35, 45, 0.65)
    );
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    opacity: 0.85;
}

.card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 15px;
}

.success-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(50,180,100,0.4);
    background: rgba(50,180,100,0.08);
}

.warning-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(240,180,50,0.4);
    background: rgba(240,180,50,0.08);
}

.reply-box {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(100,140,255,0.35);
    background: rgba(100,140,255,0.08);
    font-size: 17px;
    line-height: 1.6;
}

.small-text {
    font-size: 13px;
    opacity: 0.7;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"
CSV_FILE = "analyzed_reviews.csv"


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def get_groq_client():

    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


# ============================================================
# ANALYZE REVIEW
# ============================================================

def analyze_review(review):

    prompt = f"""
You are an expert customer review analyst for a small business.

Analyze the customer review below.

CUSTOMER REVIEW:
"{review}"

Return ONLY a JSON object.

The JSON MUST have exactly these fields:

{{
    "sentiment": "Positive/Neutral/Negative",
    "category": "Service/Product/Delivery/Staff/Price/Quality/Other",
    "urgency": "Low/Medium/High",
    "issue": "short description of the main issue"
}}

Rules:

1. sentiment:
   - Positive
   - Neutral
   - Negative

2. category:
   - Service
   - Product
   - Delivery
   - Staff
   - Price
   - Quality
   - Other

3. urgency:
   - Low
   - Medium
   - High

4. issue:
   - Keep it short.
   - Clearly describe the main complaint or positive point.

Do not include markdown.
Do not include explanations outside JSON.
"""

    try:

        client = get_groq_client()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze customer reviews and return "
                        "structured JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            response_format={
                "type": "json_object"
            }
        )

        result = response.choices[0].message.content

        return json.loads(result)

    except Exception as e:

        return {
            "sentiment": "Unknown",
            "category": "Other",
            "urgency": "Unknown",
            "issue": f"AI analysis error: {str(e)}"
        }


# ============================================================
# GENERATE AI RESPONSE
# ============================================================

def generate_response(review, analysis):

    prompt = f"""
You are a professional customer service manager.

Write a high-quality response to the customer's review.

CUSTOMER REVIEW:
"{review}"

ANALYSIS:

Sentiment:
{analysis.get("sentiment", "Unknown")}

Category:
{analysis.get("category", "Other")}

Urgency:
{analysis.get("urgency", "Unknown")}

Main Issue:
{analysis.get("issue", "None")}

RESPONSE RULES:

For positive reviews:
- Thank the customer.
- Show appreciation.
- Mention something specific from their review when possible.
- Keep the tone warm.

For negative reviews:
- Apologize sincerely.
- Acknowledge the specific problem.
- Show empathy.
- State that the business will look into the issue.
- Avoid defensive language.

For neutral reviews:
- Thank the customer.
- Acknowledge their feedback.
- Respond professionally.

IMPORTANT:
- Never mention AI.
- Never invent refunds.
- Never invent discounts.
- Never invent company policies.
- Never make promises that are not supported by the review.
- Do not repeat the entire review.
- Do not use hashtags.
- Do not use excessive emojis.
- Keep the response between 50 and 100 words.
- Return ONLY the customer-facing response.
"""

    try:

        client = get_groq_client()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an experienced, empathetic and "
                        "professional customer service representative."
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

    new_review = pd.DataFrame([
        {
            "Date": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Review": review,
            "Sentiment": analysis.get(
                "sentiment",
                "Unknown"
            ),
            "Category": analysis.get(
                "category",
                "Other"
            ),
            "Urgency": analysis.get(
                "urgency",
                "Unknown"
            ),
            "Issue": analysis.get(
                "issue",
                ""
            ),
            "AI Response": ai_response
        }
    ])

    try:

        if os.path.exists(CSV_FILE):

            old_data = pd.read_csv(CSV_FILE)

            all_data = pd.concat(
                [old_data, new_review],
                ignore_index=True
            )

        else:

            all_data = new_review

        all_data.to_csv(
            CSV_FILE,
            index=False
        )

    except Exception as e:

        st.warning(
            f"Could not save review history: {e}"
        )


# ============================================================
# SENTIMENT ICON
# ============================================================

def sentiment_icon(sentiment):

    sentiment = str(sentiment).lower()

    if sentiment == "positive":
        return "😊"

    if sentiment == "negative":
        return "😟"

    if sentiment == "neutral":
        return "😐"

    return "❓"


# ============================================================
# URGENCY ICON
# ============================================================

def urgency_icon(urgency):

    urgency = str(urgency).lower()

    if urgency == "high":
        return "🚨"

    if urgency == "medium":
        return "⚠️"

    if urgency == "low":
        return "🟢"

    return "❓"


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>⭐ AI Review Management Agent</h1>

<p>
Turn customer reviews into actionable business intelligence.
Analyze sentiment, detect issues, identify urgent complaints,
and generate professional responses automatically.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 AI Review Agent")

    st.write(
        "Your intelligent assistant for managing customer feedback."
    )

    st.divider()

    st.subheader("✨ Features")

    st.write("😊 Sentiment Analysis")
    st.write("📂 Review Categorization")
    st.write("🚨 Urgency Detection")
    st.write("🔎 Issue Identification")
    st.write("💬 AI Response Generation")
    st.write("📊 Review History")
    st.write("📈 Business Insights")

    st.divider()

    st.caption(
        "AI Model: GPT-OSS 20B"
    )

    st.caption(
        "Powered by Groq"
    )


# ============================================================
# REVIEW INPUT
# ============================================================

st.header("📝 Analyze a Customer Review")

st.write(
    "Enter a customer review and let the AI analyze it."
)

review = st.text_area(
    "Customer Review",
    placeholder=(
        "Example:\n\n"
        "The food was delicious, but my order arrived two hours "
        "late. I tried contacting customer service several times "
        "but nobody responded."
    ),
    height=180,
    label_visibility="visible"
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

        # Save current review
        st.session_state["review"] = review

        # ----------------------------------------------------
        # ANALYSIS
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Analyzing customer review..."
        ):

            analysis = analyze_review(
                review
            )

        st.session_state["analysis"] = analysis

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        with st.spinner(
            "✍️ Generating personalized customer response..."
        ):

            ai_response = generate_response(
                review,
                analysis
            )

        st.session_state["ai_response"] = ai_response

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_review(
            review,
            analysis,
            ai_response
        )

        st.session_state["just_analyzed"] = True


# ============================================================
# RESULTS
# ============================================================

if "analysis" in st.session_state:

    analysis = st.session_state["analysis"]

    st.divider()

    st.header("📊 Review Analysis")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    sentiment = analysis.get(
        "sentiment",
        "Unknown"
    )

    category = analysis.get(
        "category",
        "Other"
    )

    urgency = analysis.get(
        "urgency",
        "Unknown"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Sentiment",
            f"{sentiment_icon(sentiment)} {sentiment}"
        )

    with col2:

        st.metric(
            "Category",
            f"📂 {category}"
        )

    with col3:

        st.metric(
            "Urgency",
            f"{urgency_icon(urgency)} {urgency}"
        )


    # --------------------------------------------------------
    # ISSUE
    # --------------------------------------------------------

    st.subheader("🔎 Main Issue")

    issue = analysis.get(
        "issue",
        "No issue identified."
    )

    if urgency.lower() == "high":

        st.error(
            f"🚨 High-priority issue: {issue}"
        )

    elif urgency.lower() == "medium":

        st.warning(
            f"⚠️ Issue requiring attention: {issue}"
        )

    else:

        st.info(
            issue
        )


    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    if "ai_response" in st.session_state:

        st.divider()

        st.header("💬 AI Generated Reply")

        st.write(
            "Here is a personalized response that your "
            "business can send to the customer:"
        )

        ai_response = st.session_state[
            "ai_response"
        ]

        st.markdown(
            f"""
            <div class="reply-box">
            {ai_response}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.text_area(
            "Copyable Response",
            value=ai_response,
            height=150
        )


# ============================================================
# DASHBOARD
# ============================================================

st.divider()

st.header("📈 Business Review Dashboard")


if os.path.exists(CSV_FILE):

    try:

        history = pd.read_csv(
            CSV_FILE
        )

        if not history.empty:

            # ------------------------------------------------
            # TOP METRICS
            # ------------------------------------------------

            total_reviews = len(
                history
            )

            positive_count = (
                history["Sentiment"]
                .astype(str)
                .str.lower()
                .eq("positive")
                .sum()
            )

            negative_count = (
                history["Sentiment"]
                .astype(str)
                .str.lower()
                .eq("negative")
                .sum()
            )

            high_urgency = (
                history["Urgency"]
                .astype(str)
                .str.lower()
                .eq("high")
                .sum()
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "📋 Total Reviews",
                    total_reviews
                )

            with col2:

                st.metric(
                    "😊 Positive",
                    int(positive_count)
                )

            with col3:

                st.metric(
                    "😟 Negative",
                    int(negative_count)
                )

            with col4:

                st.metric(
                    "🚨 High Urgency",
                    int(high_urgency)
                )


            # ------------------------------------------------
            # CHARTS
            # ------------------------------------------------

            st.subheader(
                "📊 Sentiment Distribution"
            )

            sentiment_counts = (
                history["Sentiment"]
                .value_counts()
            )

            st.bar_chart(
                sentiment_counts
            )


            st.subheader(
                "📂 Review Categories"
            )

            category_counts = (
                history["Category"]
                .value_counts()
            )

            st.bar_chart(
                category_counts
            )


            # ------------------------------------------------
            # URGENCY
            # ------------------------------------------------

            st.subheader(
                "🚨 Urgency Distribution"
            )

            urgency_counts = (
                history["Urgency"]
                .value_counts()
            )

            st.bar_chart(
                urgency_counts
            )


            # ------------------------------------------------
            # HISTORY
            # ------------------------------------------------

            st.subheader(
                "📚 Review History"
            )

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            csv_data = history.to_csv(
                index=False
            )

            st.download_button(
                label="⬇️ Download Review Report",
                data=csv_data,
                file_name="review_analysis_report.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.info(
                "No reviews have been analyzed yet."
            )

    except Exception as e:

        st.warning(
            f"Could not load dashboard data: {e}"
        )

else:

    st.info(
        "📊 Your dashboard will appear here after "
        "you analyze your first review."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
<div style="text-align:center; opacity:0.7;">

⭐ <b>AI Review Management Agent</b>

<p>
Analyze • Understand • Respond • Improve
</p>

</div>
""",
    unsafe_allow_html=True
)