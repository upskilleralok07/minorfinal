import argparse
import os
import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================
# File Names
# ======================================================

CSV_FILE = os.path.join(BASE_DIR, "internship.csv")
VECTORIZER_FILE = os.path.join(BASE_DIR, "vectorizer.pkl")
TFIDF_MATRIX_FILE = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
DATA_FILE = os.path.join(BASE_DIR, "internship_data.pkl")


# ======================================================
# Clean Stipend Function
# ======================================================

def extract_stipend_value(stipend):
    """
    Converts stipend text into numeric value.

    Examples:
    ₹ 5,000 /month -> 5000
    ₹ 5,000-10,000 /month -> 7500
    Unpaid -> 0
    """
    stipend = str(stipend).lower()

    if "unpaid" in stipend:
        return 0

    numbers = re.findall(r"\d+", stipend.replace(",", ""))

    if len(numbers) == 0:
        return 0

    numbers = list(map(int, numbers))

    if len(numbers) == 1:
        return numbers[0]

    return sum(numbers) / len(numbers)


# ======================================================
# Train Model and Generate PKL Files
# ======================================================

def train_and_save_model():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    print("Loading dataset...")

    df = pd.read_csv(CSV_FILE)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    required_columns = [
        "internship_title",
        "company_name",
        "location",
        "start_date",
        "duration",
        "stipend"
    ]

    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing column - {col}")
            return

    # Clean missing values
    df = df.fillna("")

    # Create numeric stipend column
    df["stipend_numeric"] = df["stipend"].apply(extract_stipend_value)

    # Combine text features
    df["combined_features"] = (
        df["internship_title"].astype(str) + " " +
        df["company_name"].astype(str) + " " +
        df["location"].astype(str) + " " +
        df["start_date"].astype(str) + " " +
        df["duration"].astype(str) + " " +
        df["stipend"].astype(str)
    )

    # Train TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

    print("Model trained successfully.")
    print("TF-IDF matrix shape:", tfidf_matrix.shape)

    # Save PKL files
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(TFIDF_MATRIX_FILE, "wb") as f:
        pickle.dump(tfidf_matrix, f)

    with open(DATA_FILE, "wb") as f:
        pickle.dump(df, f)

    print("\nPKL files generated successfully:")
    print(f"1. {VECTORIZER_FILE}")
    print(f"2. {TFIDF_MATRIX_FILE}")
    print(f"3. {DATA_FILE}")


# ======================================================
# Load Model
# ======================================================

def load_model():
    if not os.path.exists(VECTORIZER_FILE):
        print("Model files not found. Training model first...")
        train_and_save_model()

    with open(VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)

    with open(TFIDF_MATRIX_FILE, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(DATA_FILE, "rb") as f:
        df = pickle.load(f)

    return vectorizer, tfidf_matrix, df


# ======================================================
# Recommendation Function
# ======================================================

def recommend_internships(
    skills,
    preferred_location="",
    min_stipend=0,
    work_from_home=False,
    top_n=10
):
    vectorizer, tfidf_matrix, df = load_model()

    user_query = skills

    if preferred_location:
        user_query += " " + preferred_location

    if work_from_home:
        user_query += " Work From Home"

    user_vector = vectorizer.transform([user_query])

    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    result_df = df.copy()
    result_df["similarity_score"] = similarity_scores

    # Filter by stipend
    result_df = result_df[result_df["stipend_numeric"] >= min_stipend]

    # Filter by location
    if preferred_location:
        result_df = result_df[
            result_df["location"].str.contains(
                preferred_location,
                case=False,
                na=False
            ) |
            result_df["location"].str.contains(
                "Work From Home",
                case=False,
                na=False
            )
        ]

    # Only work from home filter
    if work_from_home:
        result_df = result_df[
            result_df["location"].str.contains(
                "Work From Home",
                case=False,
                na=False
            )
        ]

    # Sort recommendations
    result_df = result_df.sort_values(
        by="similarity_score",
        ascending=False
    )

    output_columns = [
        "internship_title",
        "company_name",
        "location",
        "start_date",
        "duration",
        "stipend",
        "stipend_numeric",
        "similarity_score"
    ]

    return result_df[output_columns].head(top_n)


# ======================================================
# Main Program
# ======================================================

def main():
    parser = argparse.ArgumentParser(
        description="Internship Recommendation System"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model and generate PKL files using the local internship CSV dataset."
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Run the recommendation workflow after ensuring the model is available."
    )
    args = parser.parse_args()

    if args.train:
        train_and_save_model()
        return

    if args.recommend:
        if not os.path.exists(VECTORIZER_FILE) or not os.path.exists(TFIDF_MATRIX_FILE) or not os.path.exists(DATA_FILE):
            train_and_save_model()

        skills = input("\nEnter your skills/interests: ")
        preferred_location = input("Enter preferred location: ")

        try:
            min_stipend = int(input("Enter minimum stipend: "))
        except ValueError:
            print("Invalid stipend. Using 0 as default.")
            min_stipend = 0

        wfh_input = input("Do you want Work From Home? yes/no: ").lower()
        work_from_home = wfh_input == "yes"

        try:
            top_n = int(input("How many recommendations do you want? "))
        except ValueError:
            top_n = 10

        recommendations = recommend_internships(
            skills=skills,
            preferred_location=preferred_location,
            min_stipend=min_stipend,
            work_from_home=work_from_home,
            top_n=top_n
        )

        if recommendations.empty:
            print("\nNo matching internships found.")
        else:
            print("\nTop Recommended Internships:\n")
            print(recommendations.to_string(index=False))
        return

    # Interactive menu fallback.
    print("\n====================================")
    print("Internship Recommendation System")
    print("====================================")

    while True:
        print("\nChoose an option:")
        print("1. Train model and generate PKL files")
        print("2. Recommend internships")
        print("3. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            train_and_save_model()

        elif choice == "2":
            skills = input("\nEnter your skills/interests: ")
            preferred_location = input("Enter preferred location: ")

            try:
                min_stipend = int(input("Enter minimum stipend: "))
            except ValueError:
                print("Invalid stipend. Using 0 as default.")
                min_stipend = 0

            wfh_input = input("Do you want Work From Home? yes/no: ").lower()
            work_from_home = wfh_input == "yes"

            try:
                top_n = int(input("How many recommendations do you want? "))
            except ValueError:
                top_n = 10

            recommendations = recommend_internships(
                skills=skills,
                preferred_location=preferred_location,
                min_stipend=min_stipend,
                work_from_home=work_from_home,
                top_n=top_n
            )

            if recommendations.empty:
                print("\nNo matching internships found.")
            else:
                print("\nTop Recommended Internships:\n")
                print(recommendations.to_string(index=False))

        elif choice == "3":
            print("Exiting program...")
            break

        else:
            print("Invalid choice. Please try again.")


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()