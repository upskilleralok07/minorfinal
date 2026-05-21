import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Helper mappings
# -----------------------------
ROLE_SKILL_MAP = {
    "data scientist": ["python", "machine learning", "pandas", "numpy", "sql", "statistics"],
    "python developer": ["python", "flask", "django", "fastapi", "sql", "git"],
    "web developer": ["html", "css", "javascript", "react", "node", "sql", "git"],
    "devops engineer": ["docker", "kubernetes", "aws", "linux", "git", "ci/cd"],
    "java developer": ["java", "spring boot", "sql", "git", "rest api"],
    "database": ["sql", "mysql", "postgresql", "database"],
    "it intern": ["python", "sql", "git", "computer networks"],
    "information technology": ["python", "sql", "git", "api"],
    "sales & marketing": ["communication", "sales", "marketing", "negotiation"],
    "operations management": ["excel", "analysis", "coordination", "reporting"],
    "finance & accounting": ["excel", "finance", "accounting", "analysis"],
    "human resources": ["communication", "recruitment", "coordination"],
    "maintenance": ["technical", "safety", "equipment"],
    "customer care / service": ["communication", "customer service", "problem solving"],
}


# -----------------------------
# Text cleaning
# -----------------------------
def normalize_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


# -----------------------------
# Main recommender class
# -----------------------------
class InternshipRecommender:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.df = None
        self.vectorizer = None
        self.text_matrix = None

        self.load_data()
        self.prepare_features()

    def load_data(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.df.columns = [str(col).strip() for col in self.df.columns]

        if "internship_title" in self.df.columns and "internship_role" not in self.df.columns:
            self.df["internship_role"] = self.df["internship_title"].astype(str)

        self.df["company_name"] = self.df["company_name"].astype(str) if "company_name" in self.df.columns else ""
        self.df["internship_role"] = self.df["internship_role"].astype(str) if "internship_role" in self.df.columns else ""
        self.df["functional_area"] = self.df["functional_area"].astype(str) if "functional_area" in self.df.columns else ""
        self.df["industry"] = self.df["industry"].astype(str) if "industry" in self.df.columns else ""
        self.df["location"] = self.df["location"].astype(str) if "location" in self.df.columns else ""
        if "state" in self.df.columns:
            self.df["state"] = self.df["state"].astype(str)
        else:
            self.df["state"] = self.df["location"]
        self.df["start_date"] = self.df["start_date"].astype(str) if "start_date" in self.df.columns else ""
        self.df["duration"] = self.df["duration"].astype(str) if "duration" in self.df.columns else ""
        self.df["stipend"] = self.df["stipend"].astype(str) if "stipend" in self.df.columns else ""
        self.df["distance_km"] = pd.to_numeric(self.df["distance_km"], errors="coerce").fillna(0.0) if "distance_km" in self.df.columns else 0.0

        for col in ["company_name", "internship_role", "functional_area", "industry", "location", "state"]:
            self.df[col] = self.df[col].apply(normalize_text)

        self.df = self.df.drop_duplicates().reset_index(drop=True)

    def prepare_features(self):
        self.df["combined_text"] = (
            self.df["company_name"] + " " +
            self.df["internship_role"] + " " +
            self.df["functional_area"] + " " +
            self.df["industry"] + " " +
            self.df["state"] + " " +
            self.df["location"] + " " +
            self.df["start_date"] + " " +
            self.df["duration"] + " " +
            self.df["stipend"]
        )

        self.vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_features=5000)
        self.text_matrix = self.vectorizer.fit_transform(self.df["combined_text"])

    # -----------------------------
    # Infer skills for internship row
    # -----------------------------
    def infer_required_skills(self, internship_role, functional_area):
        internship_role = normalize_text(internship_role)
        functional_area = normalize_text(functional_area)

        inferred_skills = set()

        for key, skills in ROLE_SKILL_MAP.items():
            if key in internship_role or key in functional_area:
                inferred_skills.update(skills)

        return inferred_skills

    # -----------------------------
    # Build user query from filters
    # -----------------------------
    def build_user_query(
        self,
        preferred_role="",
        preferred_functional_area="",
        preferred_industry="",
        preferred_company="",
        preferred_location="",
        preferred_state="madhya pradesh",
    ):
        parts = [
            normalize_text(preferred_role),
            normalize_text(preferred_functional_area),
            normalize_text(preferred_industry),
            normalize_text(preferred_company),
            normalize_text(preferred_location),
            normalize_text(preferred_state),
        ]
        return " ".join([p for p in parts if p]).strip()

    # -----------------------------
    # Recommend internships
    # -----------------------------
    def recommend(
        self,
        user_skills=None,
        preferred_role="",
        preferred_functional_area="",
        preferred_industry="",
        preferred_company="",
        preferred_location="",
        preferred_state="madhya pradesh",
        max_distance=None,
        work_from_home=False,
        top_n=5,
    ):
        if user_skills is None:
            user_skills = []

        user_skills = {normalize_text(skill) for skill in user_skills if str(skill).strip()}

        user_query = self.build_user_query(
            preferred_role=preferred_role,
            preferred_functional_area=preferred_functional_area,
            preferred_industry=preferred_industry,
            preferred_company=preferred_company,
            preferred_location=preferred_location,
            preferred_state=preferred_state,
        )

        if not user_query:
            fallback_query = " ".join(list(user_skills)[:10])
            user_query = fallback_query if fallback_query else "internship"

        if user_skills and user_query:
            user_query += " " + " ".join(sorted(user_skills))

        user_vector = self.vectorizer.transform([user_query])
        text_scores = cosine_similarity(user_vector, self.text_matrix).flatten()

        results = self.df.copy()
        results["text_score"] = text_scores

        location_boost = []
        wfh_boost = []
        skill_overlap_score = []
        distance_score = []

        preferred_location_norm = normalize_text(preferred_location)

        for _, row in results.iterrows():
            row_role = row["internship_role"]
            row_function = row["functional_area"]
            row_company = row["company_name"]
            row_location = row["location"]
            row_distance = safe_float(row["distance_km"], default=0.0)

            lb = 1.0 if preferred_location_norm and preferred_location_norm in row_location else 0.0
            location_boost.append(lb)

            if work_from_home:
                wfh_boost.append(1.0 if "work from home" in row_location else 0.0)
            else:
                wfh_boost.append(0.0)

            inferred_skills = self.infer_required_skills(row_role, row_function)
            overlap = len(user_skills.intersection(inferred_skills)) / len(inferred_skills) if inferred_skills else 0.0
            skill_overlap_score.append(overlap)

            if max_distance is not None:
                ds = 1.0 if row_distance <= max_distance else -0.5
            else:
                ds = 1.0 if row_distance <= 0 else 0.5
            distance_score.append(ds)

        results["location_boost"] = location_boost
        results["wfh_boost"] = wfh_boost
        results["skill_overlap_score"] = skill_overlap_score
        results["distance_score"] = distance_score

        results["recommendation_score"] = (
            0.40 * results["text_score"] +
            0.20 * results["skill_overlap_score"] +
            0.15 * results["location_boost"] +
            0.10 * results["wfh_boost"] +
            0.10 * results["distance_score"]
        )

        if preferred_location_norm:
            results = results[
                results["location"].str.contains(preferred_location_norm, case=False, na=False) |
                results["state"].str.contains(preferred_location_norm, case=False, na=False)
            ]

        if work_from_home:
            results = results[results["location"].str.contains("work from home", case=False, na=False)]

        if max_distance is not None:
            results = results[results["distance_km"] <= max_distance]

        results = results.sort_values(
            by=["recommendation_score", "distance_km"],
            ascending=[False, True]
        ).reset_index(drop=True)

        return results[[
            "company_name",
            "internship_role",
            "functional_area",
            "industry",
            "state",
            "location",
            "start_date",
            "duration",
            "stipend",
            "distance_km",
            "recommendation_score",
            "skill_overlap_score",
        ]].head(top_n)

    # -----------------------------
    # Explain why a row was recommended
    # -----------------------------
    def explain_recommendation(self, internship_row, user_skills=None, preferred_role=""):
        if user_skills is None:
            user_skills = []

        user_skills = {normalize_text(skill) for skill in user_skills if str(skill).strip()}

        inferred_skills = self.infer_required_skills(
            internship_row.get("internship_role", ""),
            internship_row.get("functional_area", "")
        )

        matched_skills = sorted(user_skills.intersection(inferred_skills))

        reasons = []

        if preferred_role and normalize_text(preferred_role) in normalize_text(internship_row.get("internship_role", "")):
            reasons.append("role matches your preferred role")

        if matched_skills:
            reasons.append(f"matched skills: {', '.join(matched_skills[:5])}")

        if "work from home" in normalize_text(internship_row.get("location", "")):
            reasons.append("offers work-from-home option")

        if internship_row.get("location"):
            reasons.append(f"location: {internship_row.get('location')}")

        return " | ".join(reasons)


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    recommender = InternshipRecommender("models/intern/internship.csv")

    user_skills = ["python", "sql", "git", "communication"]

    top_results = recommender.recommend(
        user_skills=user_skills,
        preferred_role="it intern",
        preferred_functional_area="information technology",
        preferred_industry="oil, gas & energy",
        preferred_company="gail",
        preferred_location="Noida",
        preferred_state="madhya pradesh",
        max_distance=200,
        work_from_home=False,
        top_n=10,
    )

    print(top_results.to_string(index=False))