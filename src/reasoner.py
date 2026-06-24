"""
reasoner.py
Generate honest, specific 1-2 sentence reasoning for each top-100 candidate.
"""


def generate_reasoning(candidate, features, final_score):
    """
    Produce a 1-2 sentence reasoning string that references things
    actually in the candidate's profile (no hallucination).
    """
    parts = []
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    history = candidate.get("career_history", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    notice = signals.get("notice_period_days", 999)
    open_to_work = signals.get("open_to_work_flag", False)
    resp_rate = signals.get("recruiter_response_rate", 0)

    # Opening: role + experience
    parts.append(f"{yoe:.1f}y experience as {title} at {company}")

    # Career highlight: find most relevant job
    retrieval_keywords = [
        "embedding", "vector", "retrieval", "search", "ranking",
        "recommendation", "semantic", "nlp", "faiss", "pinecone",
        "qdrant", "weaviate", "milvus", "elasticsearch"
    ]
    for job in history[:3]:
        desc = (job.get("description") or "").lower()
        title_j = (job.get("title") or "").lower()
        if any(kw in desc or kw in title_j for kw in retrieval_keywords):
            parts.append(
                f"career history shows {job['title']} work at {job['company']}"
                f" with retrieval/ranking exposure"
            )
            break

    # Skills highlight
    skill_names = [s["name"].lower() for s in candidate.get("skills", [])]
    key_skills = [
        s for s in skill_names
        if any(kw in s for kw in [
            "embedding", "vector", "faiss", "pinecone", "qdrant",
            "milvus", "elasticsearch", "nlp", "retrieval", "ranking",
            "sentence-transformer", "lora", "fine-tun"
        ])
    ]
    if key_skills:
        parts.append(f"relevant skills: {', '.join(key_skills[:3])}")

    # Location
    if location:
        parts.append(f"based in {location}")

    # Notice period
    if notice <= 30:
        parts.append("available within 30 days")
    elif notice > 90:
        parts.append(f"note: {notice}-day notice period")

    # Behavioral flags
    if open_to_work and resp_rate >= 0.4:
        parts.append("actively looking, responsive to recruiters")
    elif not open_to_work:
        parts.append("not marked open-to-work")

    # Consulting penalty note
    if features.get("is_consulting_only"):
        parts.append("caution: consulting-only career history")

    # Build the final string (2 sentences max)
    if len(parts) >= 3:
        sentence1 = f"{parts[0]}; {parts[1]}."
        sentence2 = " ".join(parts[2:4]) + "."
        return f"{sentence1} {sentence2}"
    else:
        return "; ".join(parts) + "."
