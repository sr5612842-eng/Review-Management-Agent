import streamlit as st
import pandas as pd
import json
import os
import io
import re
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
# SETTINGS
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"
CSV_FILE = "analyzed_reviews.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.hero {
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 25px;
    border: 1px solid rgba(128,128,128,0.25);
    background: linear-gradient(
        135deg,
        rgba(70,90,120,0.35),
        rgba(30,35,45,0.65)
    );
}

.hero h1 {
    font-size: 44px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    opacity: 0.85;
}

.section-card {
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 15px;
}

.reply-box {
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(100,140,255,0.35);
    background: rgba(100,140,255,0.08);
    font-size: 17px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def get_groq_client():

    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


# ============================================================
# SAFE JSON PARSER
# ============================================================

def parse_json_response(text):

    text = text.strip()

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    try:
        return json.loads(text)

    except Exception:

        match = re.search(
            r'\{.*\}',
            text,
            re.DOTALL
        )

        if match:

            try:
                return json.loads(
                    match.group(0)
                )

            except Exception:
                pass

    return None


# ============================================================
# SINGLE REVIEW ANALYSIS
# ============================================================

def analyze_review(review):

    prompt = f"""
You are an expert customer review analyst for a small business.

Analyze this customer review:

"{review}"

Return ONLY valid JSON using exactly this structure:

{{
    "sentiment": "Positive/Neutral/Negative",
    "category": "Service/Product/Delivery/Staff/Price/Quality/Other",
    "urgency": "Low/Medium/High",
    "quality_score": 0,
    "issue": "short description of the main issue",
    "solution": "specific practical solution for the business",
    "strengths": "what the business did well",
    "weaknesses": "what the business needs to improve"
}}

Quality score must be between 0 and 100.

Consider:
- customer satisfaction
- product/service quality
- staff behavior
- delivery
- value for money
- clarity of complaint
- overall customer experience

Return JSON only.
"""

    try:

        client = get_groq_client()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional review analyst. "
                        "Return valid JSON only."
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

        parsed = parse_json_response(result)

        if parsed:
            return parsed

        return {
            "sentiment": "Unknown",
            "category": "Other",
            "urgency": "Unknown",
            "quality_score": 0,
            "issue": "Could not parse AI response.",
            "solution": "Review the complaint manually.",
            "strengths": "",
            "weaknesses": ""
        }

    except Exception as e:

        return {
            "sentiment": "Unknown",
            "category": "Other",
            "urgency": "Unknown",
            "quality_score": 0,
            "issue": f"AI error: {str(e)}",
            "solution": "Check the AI connection.",
            "strengths": "",
            "weaknesses": ""
        }


# ============================================================
# GENERATE CUSTOMER RESPONSE
# ============================================================

def generate_response(review, analysis):

    prompt = f"""
You are a professional customer service manager.

Write a personalized response to this customer.

CUSTOMER REVIEW:
"{review}"

ANALYSIS:
Sentiment: {analysis.get("sentiment")}
Category: {analysis.get("category")}
Urgency: {analysis.get("urgency")}
Issue: {analysis.get("issue")}
Solution: {analysis.get("solution")}

Rules:

- Thank positive customers.
- Apologize appropriately to unhappy customers.
- Acknowledge the specific problem.
- Be empathetic and professional.
- Do not invent refunds or discounts.
- Do not invent company policies.
- Do not mention AI.
- Do not make unsupported promises.
- Keep the reply between 50 and 100 words.
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
                        "You are an experienced and empathetic "
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

        return f"Unable to generate reply: {str(e)}"


# ============================================================
# BULK REVIEW ANALYSIS
# ============================================================

def analyze_bulk_reviews(reviews):

    results = []

    progress = st.progress(0)
    status = st.empty()

    total = len(reviews)

    for i, review in enumerate(reviews):

        status.write(
            f"🤖 Analyzing review {i + 1} of {total}..."
        )

        if not str(review).strip():

            continue

        analysis = analyze_review(
            str(review)
        )

        results.append({
            "Review": str(review),
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
            "Quality Score": analysis.get(
                "quality_score",
                0
            ),
            "Main Issue": analysis.get(
                "issue",
                ""
            ),
            "Solution": analysis.get(
                "solution",
                ""
            ),
            "Strengths": analysis.get(
                "strengths",
                ""
            ),
            "Weaknesses": analysis.get(
                "weaknesses",
                ""
            )
        })

        progress.progress(
            (i + 1) / total
        )

    status.success(
        f"✅ Finished analyzing {len(results)} reviews."
    )

    return pd.DataFrame(results)


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

def generate_business_insights(df):

    sample = df[
        [
            "Sentiment",
            "Category",
            "Urgency",
            "Quality Score",
            "Main Issue",
            "Solution",
            "Strengths",
            "Weaknesses"
        ]
    ].to_dict(
        orient="records"
    )

    prompt = f"""
You are a senior business consultant.

Analyze these customer review results:

{json.dumps(sample, indent=2)}

Provide business-level insights.

Return ONLY valid JSON:

{{
    "overall_assessment": "short overall assessment",
    "top_problem": "most important recurring problem",
    "top_strength": "most important business strength",
    "priority_action": "most important action the business should take",
    "recommendations": [
        "recommendation 1",
        "recommendation 2",
        "recommendation 3",
        "recommendation 4",
        "recommendation 5"
    ]
}}

Focus on practical actions a small business can actually implement.
"""

    try:

        client = get_groq_client()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a business intelligence consultant. "
                        "Return JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            response_format={
                "type": "json_object"
            }
        )

        result = parse_json_response(
            response.choices[0].message.content
        )

        return result

    except Exception as e:

        return {
            "overall_assessment": str(e),
            "top_problem": "Unable to determine",
            "top_strength": "Unable to determine",
            "priority_action": "Check AI connection",
            "recommendations": []
        }


# ============================================================
# READ UPLOADED FILE
# ============================================================

def read_uploaded_file(uploaded_file):

    file_name = uploaded_file.name.lower()

    try:

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        if file_name.endswith(".csv"):

            return pd.read_csv(
                uploaded_file
            )


        # ----------------------------------------------------
        # EXCEL
        # ----------------------------------------------------

        elif file_name.endswith(
            (".xlsx", ".xls")
        ):

            return pd.read_excel(
                uploaded_file
            )


        # ----------------------------------------------------
        # TXT
        # ----------------------------------------------------

        elif file_name.endswith(".txt"):

            content = uploaded_file.read()

            text = content.decode(
                "utf-8",
                errors="ignore"
            )

            reviews = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            return pd.DataFrame({
                "Review": reviews
            })


        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        elif file_name.endswith(".pdf"):

            from pypdf import PdfReader

            reader = PdfReader(
                uploaded_file
            )

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            return pd.DataFrame({
                "Review": lines
            })


        else:

            st.error(
                "Unsupported file type."
            )

            return None

    except Exception as e:

        st.error(
            f"Could not read file: {e}"
        )

        return None


# ============================================================
# FIND REVIEW COLUMN
# ============================================================

def find_review_column(df):

    possible_names = [
        "review",
        "reviews",
        "customer review",
        "customer reviews",
        "feedback",
        "comment",
        "comments",
        "text",
        "review text",
        "feedback text"
    ]

    columns = list(df.columns)

    for column in columns:

        clean = str(column).strip().lower()

        if clean in possible_names:

            return column

    # Find the column with the most text
    best_column = None
    best_score = 0

    for column in columns:

        try:

            values = df[column].astype(str)

            score = values.str.len().mean()

            if score > best_score:

                best_score = score
                best_column = column

        except Exception:
            pass

    return best_column


# ============================================================
# SAVE DATA
# ============================================================

def save_dataframe(df):

    try:

        df.to_csv(
            CSV_FILE,
            index=False
        )

    except Exception:
        pass


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>⭐ AI Review Management Agent</h1>

<p>
Transform customer reviews into business intelligence.
Analyze individual reviews or upload hundreds of reviews
to discover problems, solutions, customer satisfaction,
and actionable business insights.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 AI Review Agent")

    st.write(
        "Your intelligent assistant for customer feedback."
    )

    st.divider()

    st.subheader("✨ Features")

    st.write("😊 Sentiment Analysis")
    st.write("⭐ Review Quality Score")
    st.write("📂 Category Detection")
    st.write("🚨 Urgency Detection")
    st.write("🔎 Issue Detection")
    st.write("💡 Solution Recommendations")
    st.write("💬 AI Customer Replies")
    st.write("📊 Bulk Review Analysis")
    st.write("🔥 Recurring Problem Detection")
    st.write("📈 Business Intelligence")

    st.divider()

    st.caption(
        "AI Model: GPT-OSS 20B"
    )

    st.caption(
        "Powered by Groq"
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📝 Single Review",
    "📁 Upload Reviews",
    "📊 Business Dashboard"
])


# ============================================================
# TAB 1 - SINGLE REVIEW
# ============================================================

with tab1:

    st.header(
        "📝 Analyze Individual Review"
    )

    review = st.text_area(
        "Customer Review",
        placeholder=(
            "Example:\n\n"
            "The food was delicious, but my delivery was "
            "two hours late. Customer service did not respond."
        ),
        height=180
    )

    if st.button(
        "🔍 Analyze Review",
        type="primary",
        use_container_width=True
    ):

        if not review.strip():

            st.warning(
                "Please enter a review."
            )

        else:

            with st.spinner(
                "🤖 Analyzing review..."
            ):

                analysis = analyze_review(
                    review
                )

            st.session_state[
                "single_analysis"
            ] = analysis

            with st.spinner(
                "✍️ Generating customer reply..."
            ):

                reply = generate_response(
                    review,
                    analysis
                )

            st.session_state[
                "single_reply"
            ] = reply


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if "single_analysis" in st.session_state:

        analysis = st.session_state[
            "single_analysis"
        ]

        st.divider()

        st.subheader(
            "📊 Review Intelligence"
        )

        col1, col2, col3, col4 = st.columns(4)

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

        with col4:

            score = analysis.get(
                "quality_score",
                0
            )

            st.metric(
                "⭐ Quality",
                f"{score}/100"
            )


        st.subheader(
            "🔎 Main Issue"
        )

        st.info(
            analysis.get(
                "issue",
                "No issue detected."
            )
        )


        st.subheader(
            "💡 Recommended Solution"
        )

        st.success(
            analysis.get(
                "solution",
                "No solution generated."
            )
        )


        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "💪 Strengths"
            )

            st.write(
                analysis.get(
                    "strengths",
                    "None identified."
                )
            )

        with col2:

            st.subheader(
                "⚠️ Weaknesses"
            )

            st.write(
                analysis.get(
                    "weaknesses",
                    "None identified."
                )
            )


        if "single_reply" in st.session_state:

            st.divider()

            st.subheader(
                "💬 AI Generated Customer Reply"
            )

            st.text_area(
                "Suggested Response",
                value=st.session_state[
                    "single_reply"
                ],
                height=160
            )


# ============================================================
# TAB 2 - BULK UPLOAD
# ============================================================

with tab2:

    st.header(
        "📁 Upload Customer Reviews"
    )

    st.write(
        "Upload a CSV, Excel, TXT, or PDF file containing "
        "customer reviews."
    )

    uploaded_file = st.file_uploader(
        "Upload your review file",
        type=[
            "csv",
            "xlsx",
            "xls",
            "txt",
            "pdf"
        ]
    )


    if uploaded_file:

        st.success(
            f"📄 File uploaded: {uploaded_file.name}"
        )

        df = read_uploaded_file(
            uploaded_file
        )

        if df is not None and not df.empty:

            st.subheader(
                "👀 Uploaded Data"
            )

            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # FIND REVIEW COLUMN
            # ------------------------------------------------

            review_column = find_review_column(
                df
            )

            if review_column:

                st.success(
                    f"✅ Review column detected: "
                    f"`{review_column}`"
                )

                reviews = (
                    df[review_column]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                st.write(
                    f"📋 **{len(reviews)} reviews detected**"
                )


                # ------------------------------------------------
                # LIMIT
                # ------------------------------------------------

                max_reviews = st.number_input(
                    "Maximum reviews to analyze",
                    min_value=1,
                    max_value=len(reviews),
                    value=min(
                        20,
                        len(reviews)
                    ),
                    step=1
                )

                reviews_to_analyze = reviews[
                    :int(max_reviews)
                ]


                # ------------------------------------------------
                # ANALYZE BULK
                # ------------------------------------------------

                if st.button(
                    "🚀 Analyze All Reviews",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Starting bulk review intelligence..."
                    ):

                        bulk_results = analyze_bulk_reviews(
                            reviews_to_analyze
                        )

                    st.session_state[
                        "bulk_results"
                    ] = bulk_results

                    # Business insights
                    if not bulk_results.empty:

                        with st.spinner(
                            "🧠 Generating business insights..."
                        ):

                            insights = generate_business_insights(
                                bulk_results
                            )

                        st.session_state[
                            "business_insights"
                        ] = insights


    # ========================================================
    # DISPLAY BULK RESULTS
    # ========================================================

    if "bulk_results" in st.session_state:

        results = st.session_state[
            "bulk_results"
        ]

        if not results.empty:

            st.divider()

            st.header(
                "📊 Bulk Review Intelligence"
            )


            # ------------------------------------------------
            # TOP METRICS
            # ------------------------------------------------

            total = len(results)

            positive = (
                results["Sentiment"]
                .astype(str)
                .str.lower()
                .eq("positive")
                .sum()
            )

            negative = (
                results["Sentiment"]
                .astype(str)
                .str.lower()
                .eq("negative")
                .sum()
            )

            high_urgency = (
                results["Urgency"]
                .astype(str)
                .str.lower()
                .eq("high")
                .sum()
            )

            average_quality = pd.to_numeric(
                results["Quality Score"],
                errors="coerce"
            ).mean()


            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "📋 Reviews",
                    total
                )

            with col2:

                st.metric(
                    "😊 Positive",
                    int(positive)
                )

            with col3:

                st.metric(
                    "🚨 High Urgency",
                    int(high_urgency)
                )

            with col4:

                st.metric(
                    "⭐ Avg Quality",
                    f"{average_quality:.1f}/100"
                )


            # ------------------------------------------------
            # QUALITY
            # ------------------------------------------------

            st.subheader(
                "⭐ Overall Review Quality"
            )

            st.progress(
                min(
                    max(
                        int(average_quality),
                        0
                    ),
                    100
                )
            )

            if average_quality >= 80:

                st.success(
                    "Excellent customer experience."
                )

            elif average_quality >= 60:

                st.warning(
                    "Customer experience is good but has room for improvement."
                )

            else:

                st.error(
                    "Customer experience requires significant improvement."
                )


            # ------------------------------------------------
            # SENTIMENT CHART
            # ------------------------------------------------

            st.subheader(
                "😊 Sentiment Distribution"
            )

            sentiment_counts = (
                results["Sentiment"]
                .value_counts()
            )

            st.bar_chart(
                sentiment_counts
            )


            # ------------------------------------------------
            # CATEGORY CHART
            # ------------------------------------------------

            st.subheader(
                "📂 Problem Categories"
            )

            category_counts = (
                results["Category"]
                .value_counts()
            )

            st.bar_chart(
                category_counts
            )


            # ------------------------------------------------
            # URGENCY CHART
            # ------------------------------------------------

            st.subheader(
                "🚨 Urgency Distribution"
            )

            urgency_counts = (
                results["Urgency"]
                .value_counts()
            )

            st.bar_chart(
                urgency_counts
            )


            # ------------------------------------------------
            # HIGH PRIORITY REVIEWS
            # ------------------------------------------------

            st.subheader(
                "🚨 High-Priority Reviews"
            )

            high_priority = results[
                results["Urgency"]
                .astype(str)
                .str.lower()
                .eq("high")
            ]

            if not high_priority.empty:

                st.dataframe(
                    high_priority[
                        [
                            "Review",
                            "Sentiment",
                            "Category",
                            "Quality Score",
                            "Main Issue",
                            "Solution"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.success(
                    "🎉 No high-priority complaints detected."
                )


            # ------------------------------------------------
            # FULL RESULTS
            # ------------------------------------------------

            st.subheader(
                "📋 Complete AI Analysis"
            )

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # BUSINESS INSIGHTS
            # ------------------------------------------------

            if "business_insights" in st.session_state:

                insights = st.session_state[
                    "business_insights"
                ]

                st.divider()

                st.header(
                    "🧠 Business Intelligence"
                )


                st.subheader(
                    "📌 Overall Assessment"
                )

                st.info(
                    insights.get(
                        "overall_assessment",
                        "No assessment available."
                    )
                )


                col1, col2 = st.columns(2)

                with col1:

                    st.subheader(
                        "🔥 Top Problem"
                    )

                    st.error(
                        insights.get(
                            "top_problem",
                            "Not identified."
                        )
                    )

                with col2:

                    st.subheader(
                        "💪 Top Strength"
                    )

                    st.success(
                        insights.get(
                            "top_strength",
                            "Not identified."
                        )
                    )


                st.subheader(
                    "🎯 Priority Action"
                )

                st.warning(
                    insights.get(
                        "priority_action",
                        "No priority action available."
                    )
                )


                st.subheader(
                    "💡 Recommended Business Actions"
                )

                recommendations = insights.get(
                    "recommendations",
                    []
                )

                for i, recommendation in enumerate(
                    recommendations,
                    start=1
                ):

                    st.write(
                        f"**{i}.** {recommendation}"
                    )


            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            st.divider()

            csv_data = results.to_csv(
                index=False
            )

            st.download_button(
                "⬇️ Download Complete AI Review Report",
                data=csv_data,
                file_name="AI_review_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================================================
# TAB 3 - BUSINESS DASHBOARD
# ============================================================

with tab3:

    st.header(
        "📊 Business Review Dashboard"
    )

    if "bulk_results" in st.session_state:

        results = st.session_state[
            "bulk_results"
        ]

        if not results.empty:

            average_quality = pd.to_numeric(
                results["Quality Score"],
                errors="coerce"
            ).mean()

            positive_percentage = (
                results["Sentiment"]
                .astype(str)
                .str.lower()
                .eq("positive")
                .mean()
                * 100
            )

            negative_percentage = (
                results["Sentiment"]
                .astype(str)
                .str.lower()
                .eq("negative")
                .mean()
                * 100
            )


            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "⭐ Review Quality",
                    f"{average_quality:.1f}/100"
                )

            with col2:

                st.metric(
                    "😊 Positive Rate",
                    f"{positive_percentage:.1f}%"
                )

            with col3:

                st.metric(
                    "😟 Negative Rate",
                    f"{negative_percentage:.1f}%"
                )


            st.subheader(
                "📈 Sentiment"
            )

            st.bar_chart(
                results["Sentiment"]
                .value_counts()
            )


            st.subheader(
                "📂 Categories"
            )

            st.bar_chart(
                results["Category"]
                .value_counts()
            )


            st.subheader(
                "🚨 Urgency"
            )

            st.bar_chart(
                results["Urgency"]
                .value_counts()
            )


            st.subheader(
                "⭐ Quality Score Distribution"
            )

            quality_data = pd.to_numeric(
                results["Quality Score"],
                errors="coerce"
            )

            st.bar_chart(
                quality_data
                .value_counts()
                .sort_index()
            )


        else:

            st.info(
                "Upload and analyze reviews first."
            )

    else:

        st.info(
            "📁 Go to the 'Upload Reviews' tab and "
            "analyze your review file to generate the dashboard."
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
Analyze • Understand • Solve • Respond • Improve
</p>

</div>
""",
    unsafe_allow_html=True
)