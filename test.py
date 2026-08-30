from analyzer import analyze_review
from response_generator import generate_response


review = """
The food was cold when it arrived and delivery took more than two hours.
This is the second time this has happened.
"""

analysis = analyze_review(review)

print("\nANALYSIS")
print(analysis)

response = generate_response(review, analysis)

print("\nAI RESPONSE")
print(response)