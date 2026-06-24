"""
feature_extractor.py
Converts a raw candidate dict into a flat feature dict and a text blob for embedding.
"""

from datetime import datetime, date

# ── Constants ──────────────────────────────────────────────────────────────────

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "hexaware",
    "mphasis", "ltimindtree", "l&t infotech", "mindtree",
    "persistent", "niit technologies", "mastech"
}

GOOD_LOCATIONS = {
    "pune": 1.0, "noida": 1.0,
    "hyderabad": 0.85, "mumbai": 0.85, "delhi": 0.85,
    "bengaluru": 0.8, "bangalore": 0.8, "chennai": 0.7,
    "gurgaon": 0.85, "gurugram": 0.85
}

# Skills the JD explicitly requires or rewards
MUST_HAVE_SKILLS = {
    "sentence-transformers", "sentence transformers", "embeddings",
    "vector database", "vector db", "pinecone", "weaviate", "qdrant",
    "milvus", "faiss", "elasticsearch", "opensearch", "hybrid search",
    "retrieval", "ranking", "ndcg", "mrr", "map", "a/b testing",
    "information retrieval", "semantic search", "dense retrieval"
}

NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "learning to rank", "xgboost", "bm25", "reranking", "re-ranking",
    "rag", "langchain", "llm", "openai", "huggingface",
    "recommendation system", "search", "nlp"
}

# Skills that are a negative signal for THIS role
NEGATIVE_SKILLS = {
    "computer vision", "image classification", "object detection",
    "speech recognition", "tts", "speech synthesis", "robotics",
    "autonomous driving", "gans", "generative adversarial"
}

TODAY = datetime.now().date()


def months_since(date_str):
    """Return how many months ago a date string (YYYY-MM-DD) was."""
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return max(0, (TODAY - d).days // 30)
    except Exception:
        return 999


def is_honeypot(candidate):
    """
    Detect impossible profiles the competition planted as traps.
    Returns True if the candidate should be excluded.
    """
    for job in candidate.get("career_history", []):
        # Check: claimed duration > company could possibly exist
        # We don't have a company founding DB, but flag extreme cases
        start_str = job.get("start_date", "")
        if start_str:
            try:
                start = datetime.strptime(start_str[:10], "%Y-%m-%d").date()
                # If they claim to have started before 2000 at a company
                # that sounds like a startup (small size), flag it
                if start.year < 1990:
                    return True
            except Exception:
                pass

    skills = candidate.get("skills", [])
    # Expert in many skills with 0 months used
    expert_zero = [
        s for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    ]
    if len(expert_zero) >= 4:
        return True

    # Impossibly high total claimed experience vs age implied by education
    edu = candidate.get("education", [])
    if edu:
        earliest_grad = min(
            (e.get("end_year", 9999) for e in edu), default=9999
        )
        yoe = candidate.get("profile", {}).get("years_of_experience", 0)
        current_year = TODAY.year
        # If they graduated in e.g. 2022 but claim 15 years experience
        if earliest_grad < 9999 and (current_year - earliest_grad) < (yoe - 3):
            return True

    return False


def is_consulting_only(career_history):
    """True if every single job has been at a consulting/IT-services company."""
    if not career_history:
        return False
    for job in career_history:
        company = job.get("company", "").lower()
        industry = job.get("industry", "").lower()
        is_consulting = any(c in company for c in CONSULTING_COMPANIES)
        is_services = "it service" in industry or "consulting" in industry
        if not (is_consulting or is_services):
            return False  # Found at least one non-consulting job
    return True


def skills_fit_score(candidate):
    """
    Score 0–1 based on how well skills match what the JD needs.
    """
    skills = candidate.get("skills", [])
    skill_names = {s["name"].lower() for s in skills}
    skill_text = " ".join(skill_names)

    must_hits = sum(
        1 for kw in MUST_HAVE_SKILLS
        if kw in skill_text
    )
    nice_hits = sum(
        1 for kw in NICE_TO_HAVE_SKILLS
        if kw in skill_text
    )
    neg_hits = sum(
        1 for kw in NEGATIVE_SKILLS
        if kw in skill_text
    )

    # Bonus for skills with actual duration and endorsements
    quality_bonus = 0.0
    for s in skills:
        name = s.get("name", "").lower()
        if any(kw in name for kw in MUST_HAVE_SKILLS):
            dur = s.get("duration_months", 0)
            end = s.get("endorsements", 0)
            if dur >= 12:
                quality_bonus += 0.05
            if end >= 10:
                quality_bonus += 0.03

    raw = (must_hits * 0.12) + (nice_hits * 0.04) - (neg_hits * 0.06) + quality_bonus
    return max(0.0, min(1.0, raw))


def career_fit_score(candidate):
    """
    Score 0–1 based on career history quality.
    Rewards: product companies, ML/AI roles, long tenures.
    Penalizes: pure services, research-only, no production.
    """
    history = candidate.get("career_history", [])
    if not history:
        return 0.0

    score = 0.5  # neutral start

    product_company_count = 0
    has_ml_role = False
    total_ml_months = 0

    for job in history:
        company = job.get("company", "").lower()
        title = job.get("title", "").lower()
        industry = job.get("industry", "").lower()
        description = job.get("description", "").lower()
        duration = job.get("duration_months", 0)

        # Is it a consulting/services company?
        is_consulting = any(c in company for c in CONSULTING_COMPANIES)
        is_services = "it service" in industry or "it consulting" in industry

        if not is_consulting and not is_services:
            product_company_count += 1

        # Does the role involve ML/AI/search?
        ml_keywords = ["machine learning", "ml engineer", "ai engineer",
                       "data scientist", "nlp", "search", "ranking", "retrieval",
                       "recommendation", "applied ml", "applied ai"]
        if any(kw in title for kw in ml_keywords):
            has_ml_role = True
            total_ml_months += duration

        # Does the description show production ML work?
        prod_keywords = ["production", "deployed", "shipped", "serving",
                         "latency", "throughput", "scale", "real-time"]
        if any(kw in description for kw in prod_keywords):
            score += 0.08

        # Bonus for embedding/retrieval work in description
        retrieval_keywords = ["embedding", "vector", "retrieval", "search",
                               "ranking", "recommendation", "faiss", "pinecone"]
        if any(kw in description for kw in retrieval_keywords):
            score += 0.10

    if product_company_count >= 2:
        score += 0.20
    elif product_company_count == 1:
        score += 0.10

    if has_ml_role:
        score += 0.15
    if total_ml_months >= 36:
        score += 0.10

    if is_consulting_only(history):
        score -= 0.35

    return max(0.0, min(1.0, score))


def behavioral_score(candidate):
    """
    Score 0–1 based on Redrob platform signals.
    An amazing profile that is inactive = not actually hireable.
    """
    signals = candidate.get("redrob_signals", {})

    score = 0.5

    # Last active
    inactive_months = months_since(signals.get("last_active_date"))
    if inactive_months <= 1:
        score += 0.20
    elif inactive_months <= 3:
        score += 0.10
    elif inactive_months <= 6:
        score += 0.00
    else:
        score -= 0.25

    # Open to work
    if signals.get("open_to_work_flag"):
        score += 0.15

    # Recruiter response rate
    resp = signals.get("recruiter_response_rate", 0)
    if resp >= 0.5:
        score += 0.15
    elif resp >= 0.3:
        score += 0.07
    elif resp < 0.1:
        score -= 0.10

    # Notice period
    notice = signals.get("notice_period_days", 999)
    if notice <= 30:
        score += 0.10
    elif notice <= 60:
        score += 0.00
    elif notice > 90:
        score -= 0.10

    # GitHub activity (proxy for engineering depth)
    github = signals.get("github_activity_score", 0)
    if github >= 50:
        score += 0.10
    elif github >= 20:
        score += 0.05

    return max(0.0, min(1.0, score))


def location_score(candidate):
    """Score 0–1 based on location match to Pune/Noida preference."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    # Not in India at all
    if country not in ("india", "in", ""):
        if willing_to_relocate:
            return 0.30
        return 0.10

    # Check known good cities
    for city, loc_score in GOOD_LOCATIONS.items():
        if city in location:
            return loc_score

    # Somewhere in India but unknown city
    if willing_to_relocate:
        return 0.55
    return 0.45


def experience_fit_score(candidate):
    """
    Score 0–1 based on years of experience matching the 5–9 year sweet spot.
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        return 1.0
    elif 4 <= yoe < 5:
        return 0.8
    elif 9 < yoe <= 12:
        return 0.75
    elif 3 <= yoe < 4:
        return 0.5
    else:
        return 0.3


def build_text_for_embedding(candidate):
    """
    Concatenate the most meaningful text fields into one string for
    semantic similarity against the JD embedding.
    """
    parts = []

    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))

    for job in candidate.get("career_history", [])[:3]:  # top 3 jobs
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))

    skill_names = [s["name"] for s in candidate.get("skills", [])]
    parts.append(", ".join(skill_names))

    return " ".join(p for p in parts if p).strip()


def extract_features(candidate):
    """
    Main entry point. Returns a dict with all numeric scores + text.
    """
    return {
        "candidate_id": candidate["candidate_id"],
        "text": build_text_for_embedding(candidate),
        "skills_score": skills_fit_score(candidate),
        "career_score": career_fit_score(candidate),
        "behavioral_score": behavioral_score(candidate),
        "location_score": location_score(candidate),
        "experience_score": experience_fit_score(candidate),
        "is_honeypot": is_honeypot(candidate),
        "is_consulting_only": is_consulting_only(
            candidate.get("career_history", [])
        ),
        "notice_days": candidate.get("redrob_signals", {}).get(
            "notice_period_days", 999
        ),
        "yoe": candidate.get("profile", {}).get("years_of_experience", 0),
        "name": candidate.get("profile", {}).get("anonymized_name", ""),
        "title": candidate.get("profile", {}).get("current_title", ""),
        "company": candidate.get("profile", {}).get("current_company", ""),
        "location": candidate.get("profile", {}).get("location", ""),
    }
