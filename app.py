import streamlit as st
import pandas as pd
import plotly.express as px
import ollama
import json
from pypdf import PdfReader


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Review Management Agent",
    page_icon="⭐",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.metric-card {
    padding: 20px;
    border-radius: 10px;
    background-color: #f5f5f5;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">⭐ AI Customer Review Management Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze customer reviews, detect complaints, identify urgent issues, '
    'and generate personalized responses.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "AI Model",
    ["gemma2:2b"]
)

st.sidebar.info(
    "Upload a CSV or PDF file containing customer reviews."
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# ANALYZE ONE REVIEW
# ============================================================

def analyze_review(review):

    prompt = f"""
You are an AI customer review analyst for a small business.

Analyze the following customer review.

CUSTOMER REVIEW:
{review}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "sentiment": "Positive",
    "category": "Service",
    "urgency": "Low",
    "issue": "No major issue"
}}

Rules:

sentiment must be one of:
Positive, Neutral, Negative

category must be one of:
Service, Product, Delivery, Staff, Price, Quality, Other

urgency must be one of:
Low, Medium, High

issue:
Give a short description of the main problem.

If the review is positive and contains no complaint,
use:

"issue": "No major issue"

Do not add markdown.
Do not add explanations.
Return JSON only.
"""

    try:

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        result = response["message"]["content"].strip()

        # Remove markdown JSON formatting if AI adds it
        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

        data = json.loads(result)

        return data

    except Exception as e:

        return {
            "sentiment": "Unknown",
            "category": "Other",
            "urgency": "Unknown",
            "issue": f"Analysis error: {str(e)}"
        }


# ============================================================
# GENERATE AI RESPONSE
# ============================================================

def generate_response(review, analysis):

    prompt = f"""
You are a professional customer service manager.

Write a personalized response to the following customer review.

CUSTOMER REVIEW:
{review}

REVIEW ANALYSIS:

Sentiment:
{analysis['sentiment']}

Category:
{analysis['category']}

Urgency:
{analysis['urgency']}

Issue:
{analysis['issue']}

Instructions:

1. Be polite and professional.
2. Sound natural, not robotic.
3. If the review is negative, apologize appropriately.
4. Mention the customer's specific problem.
5. If the review is positive, thank the customer.
6. Do not make unrealistic promises.
7. Keep the response between 40 and 80 words.
8. Do not use hashtags.

Return only the response.
"""

    try:

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"].strip()

    except Exception as e:

        return f"Unable to generate response: {str(e)}"


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📂 Upload Customer Reviews")

uploaded_file = st.file_uploader(
    "Choose a CSV or PDF file",
    type=["csv", "pdf"]
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file:

    reviews = []

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if uploaded_file.name.lower().endswith(".csv"):

        try:

            df = pd.read_csv(uploaded_file)

            st.success("✅ CSV file uploaded successfully!")

            st.subheader("📄 Uploaded Reviews")

            st.dataframe(
                df,
                use_container_width=True
            )

            # Find review column automatically

            possible_columns = [
                "review",
                "Review",
                "reviews",
                "Reviews",
                "comment",
                "Comment",
                "feedback",
                "Feedback"
            ]

            review_column = None

            for column in possible_columns:

                if column in df.columns:
                    review_column = column
                    break

            if review_column is None:

                st.error(
                    "❌ Could not find a review column.\n\n"
                    "Your CSV should contain a column such as "
                    "'review', 'Review', 'feedback', or 'comment'."
                )

                st.stop()

            for review in df[review_column]:

                if pd.notna(review):

                    reviews.append(str(review))


        except Exception as e:

            st.error(f"Error reading CSV: {e}")

            st.stop()


    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    elif uploaded_file.name.lower().endswith(".pdf"):

        st.success("✅ PDF file uploaded successfully!")

        try:

            pdf_text = extract_pdf_text(uploaded_file)

            if not pdf_text.strip():

                st.error(
                    "❌ No readable text was found in the PDF."
                )

                st.stop()

            st.subheader("📄 Extracted PDF Text")

            st.text_area(
                "PDF Content",
                pdf_text,
                height=250
            )

            # ------------------------------------------------
            # Try to split PDF into reviews
            # ------------------------------------------------

            lines = pdf_text.split("\n")

            for line in lines:

                line = line.strip()

                if len(line) > 15:

                    reviews.append(line)

        except Exception as e:

            st.error(f"Error reading PDF: {e}")

            st.stop()


    # ========================================================
    # CHECK REVIEWS
    # ========================================================

    if len(reviews) == 0:

        st.warning(
            "⚠️ No reviews were found in the uploaded file."
        )

        st.stop()


    st.info(
        f"📊 {len(reviews)} review(s) found."
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🤖 Analyze Reviews",
        type="primary",
        use_container_width=True
    ):

        results = []

        progress_bar = st.progress(0)

        status_text = st.empty()


        # ----------------------------------------------------
        # PROCESS EACH REVIEW
        # ----------------------------------------------------

        for index, review in enumerate(reviews):

            status_text.write(
                f"🔍 Analyzing review {index + 1} of {len(reviews)}..."
            )

            analysis = analyze_review(review)

            response = generate_response(
                review,
                analysis
            )

            results.append({
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
                "AI Response": response
            })

            progress_bar.progress(
                (index + 1) / len(reviews)
            )


        status_text.success(
            "✅ All reviews analyzed successfully!"
        )


        # ====================================================
        # CREATE RESULT DATAFRAME
        # ====================================================

        result_df = pd.DataFrame(results)


        # ====================================================
        # SAVE RESULTS
        # ====================================================

        result_df.to_csv(
            "analyzed_reviews.csv",
            index=False
        )


        # ====================================================
        # DASHBOARD
        # ====================================================

        st.divider()

        st.header("📊 Review Dashboard")


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        total_reviews = len(result_df)

        positive_reviews = len(
            result_df[
                result_df["Sentiment"] == "Positive"
            ]
        )

        negative_reviews = len(
            result_df[
                result_df["Sentiment"] == "Negative"
            ]
        )

        urgent_reviews = len(
            result_df[
                result_df["Urgency"] == "High"
            ]
        )


        col1, col2, col3, col4 = st.columns(4)


        col1.metric(
            "📝 Total Reviews",
            total_reviews
        )

        col2.metric(
            "😊 Positive",
            positive_reviews
        )

        col3.metric(
            "😡 Negative",
            negative_reviews
        )

        col4.metric(
            "🚨 Urgent",
            urgent_reviews
        )


        # ====================================================
        # SENTIMENT CHART
        # ====================================================

        st.divider()

        col1, col2 = st.columns(2)


        with col1:

            st.subheader("😊 Sentiment Analysis")

            sentiment_counts = (
                result_df["Sentiment"]
                .value_counts()
                .reset_index()
            )

            sentiment_counts.columns = [
                "Sentiment",
                "Count"
            ]

            fig_sentiment = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                title="Customer Sentiment"
            )

            st.plotly_chart(
                fig_sentiment,
                use_container_width=True
            )


        # ====================================================
        # CATEGORY CHART
        # ====================================================

        with col2:

            st.subheader("📌 Review Categories")

            category_counts = (
                result_df["Category"]
                .value_counts()
                .reset_index()
            )

            category_counts.columns = [
                "Category",
                "Count"
            ]

            fig_category = px.bar(
                category_counts,
                x="Category",
                y="Count",
                title="Main Complaint Categories"
            )

            st.plotly_chart(
                fig_category,
                use_container_width=True
            )


        # ====================================================
        # URGENT COMPLAINTS
        # ====================================================

        st.divider()

        st.header("🚨 Urgent Complaints")


        urgent_df = result_df[
            result_df["Urgency"] == "High"
        ]


        if len(urgent_df) == 0:

            st.success(
                "🎉 No high-priority complaints detected!"
            )

        else:

            for index, row in urgent_df.iterrows():

                with st.expander(
                    f"🚨 {row['Issue']}"
                ):

                    st.write(
                        "**Customer Review:**"
                    )

                    st.write(
                        row["Review"]
                    )

                    st.write(
                        f"**Category:** {row['Category']}"
                    )

                    st.write(
                        f"**Sentiment:** {row['Sentiment']}"
                    )

                    st.write(
                        f"**Urgency:** {row['Urgency']}"
                    )

                    st.write(
                        "**AI Suggested Response:**"
                    )

                    st.info(
                        row["AI Response"]
                    )


        # ====================================================
        # ALL REVIEW RESULTS
        # ====================================================

        st.divider()

        st.header("📋 Complete AI Analysis")

        st.dataframe(
            result_df[
                [
                    "Review",
                    "Sentiment",
                    "Category",
                    "Urgency",
                    "Issue"
                ]
            ],
            use_container_width=True
        )


        # ====================================================
        # AI RESPONSES
        # ====================================================

        st.divider()

        st.header("💬 AI Generated Responses")


        for index, row in result_df.iterrows():

            with st.expander(
                f"Review {index + 1}"
            ):

                st.write(
                    "**Customer Review:**"
                )

                st.write(
                    row["Review"]
                )

                st.write(
                    "**Suggested Response:**"
                )

                st.info(
                    row["AI Response"]
                )


        # ====================================================
        # RECURRING PROBLEMS
        # ====================================================

        st.divider()

        st.header("🔁 Recurring Problems")


        negative_df = result_df[
            result_df["Sentiment"] == "Negative"
        ]


        if len(negative_df) > 0:

            recurring = (
                negative_df["Category"]
                .value_counts()
                .reset_index()
            )

            recurring.columns = [
                "Category",
                "Complaints"
            ]


            for _, row in recurring.iterrows():

                if row["Complaints"] >= 2:

                    st.warning(
                        f"⚠️ **{row['Category']}** "
                        f"appears in "
                        f"**{row['Complaints']} complaints**."
                    )

        else:

            st.success(
                "🎉 No recurring negative problems detected."
            )


        # ====================================================
        # BUSINESS INSIGHTS
        # ====================================================

        st.divider()

        st.header("💡 Business Insights")


        if len(negative_df) > 0:

            most_common_category = (
                negative_df["Category"]
                .value_counts()
                .idxmax()
            )

            count = (
                negative_df["Category"]
                .value_counts()
                .max()
            )

            st.write(
                f"🔴 The most common complaint category is "
                f"**{most_common_category}**, with "
                f"**{count} complaint(s)**."
            )


        if urgent_reviews > 0:

            st.write(
                f"🚨 There are **{urgent_reviews} high-priority "
                f"complaint(s)** that should be reviewed quickly."
            )


        if positive_reviews > 0:

            percentage = (
                positive_reviews / total_reviews
            ) * 100

            st.write(
                f"😊 Approximately **{percentage:.1f}%** "
                f"of the reviews were classified as positive."
            )


        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        st.divider()

        st.subheader("📥 Download Analysis")

        csv_data = result_df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="⬇️ Download Analyzed Reviews",
            data=csv_data,
            file_name="analyzed_reviews.csv",
            mime="text/csv",
            use_container_width=True
        )