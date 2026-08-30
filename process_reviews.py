import pandas as pd

from analyzer import analyze_review
from response_generator import generate_response


reviews = pd.read_csv("data/reviews.csv")

results = []

for _, row in reviews.iterrows():

    print(f"Analyzing review {row['review_id']}...")

    analysis = analyze_review(row["review"])

    response = generate_response(
        row["review"],
        analysis
    )

    results.append({
        "review_id": row["review_id"],
        "customer": row["customer"],
        "platform": row["platform"],
        "rating": row["rating"],
        "review": row["review"],
        "sentiment": analysis["sentiment"],
        "category": analysis["category"],
        "urgency": analysis["urgency"],
        "issue": analysis["issue"],
        "ai_response": response
    })


result_df = pd.DataFrame(results)

result_df.to_csv(
    "data/analyzed_reviews.csv",
    index=False
)

print("\nAnalysis complete!")
print(result_df)