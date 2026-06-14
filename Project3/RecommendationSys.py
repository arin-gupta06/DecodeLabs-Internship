import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv(r"Dataset\course_recommendation_dataset.csv")

# Combine important features
df["combined_features"] = (
    df["category"].fillna("") + " " +
    df["difficulty_level"].fillna("") + " " +
    df["tags"].fillna("")
)


vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(df["combined_features"])


user_category = input("Enter preferred category: ")
user_difficulty = input("Enter difficulty level: ")
user_tags = input("Enter preferred tags (space separated): ")

user_input = f"{user_category} {user_difficulty} {user_tags}"


user_vector = vectorizer.transform([user_input])

similarity_scores = cosine_similarity(user_vector, feature_vectors)


df["similarity_score"] = similarity_scores[0]


recommendations = df.sort_values(
    by="similarity_score",
    ascending=False
)

# Top 5 recommendations
print("\nTop 5 Recommended Courses:\n")

for index, row in recommendations.head(5).iterrows():
    print(f"Course: {row['course_name']}")
    print(f"Category: {row['category']}")
    print(f"Difficulty: {row['difficulty_level']}")
    print(f"Rating: {row['rating']}")
    print(f"Similarity Score: {row['similarity_score']:.4f}")
    print("-" * 40)