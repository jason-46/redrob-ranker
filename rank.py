"""
rank.py  —  Main ranking script for Redrob Hackathon.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

What this does (in order):
  1. Load all 100,000 candidates from the JSONL file
  2. Extract features from each candidate (skills, career, behavior, location)
  3. Remove honeypots
  4. Compute semantic similarity between each candidate and the JD
  5. Combine all scores into a final weighted score
  6. Write top-100 candidates to a CSV file

Must finish in under 5 minutes on a 16 GB CPU-only machine.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src/ to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))
from feature_extractor import extract_features
from reasoner import generate_reasoning

# ── JD text (what we are ranking candidates AGAINST) ──────────────────────────
JD_TEXT = """
Senior AI Engineer at a Series A AI-native talent intelligence platform.
Role requires production experience with embeddings-based retrieval systems
such as sentence-transformers, OpenAI embeddings, BGE, E5, or similar.
Must have production experience with vector databases or hybrid search infrastructure
such as Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Strong Python required. Hands-on experience designing evaluation frameworks
for ranking systems including NDCG, MRR, MAP, offline-to-online correlation,
A/B test interpretation.
Preference for candidates who have shipped ranking, retrieval, or recommendation
systems to real users at product companies. Not consulting firms.
Located in Pune, Noida, Hyderabad, Mumbai, Delhi NCR or willing to relocate.
5 to 9 years of experience in applied ML and AI roles.
Not looking for pure researchers, pure data engineers, or computer vision specialists.
"""

# ── Score weights (must sum to 1.0) ───────────────────────────────────────────
WEIGHTS = {
    "semantic":    0.35,   # how well the candidate's text matches the JD
    "skills":      0.25,   # direct skill keyword match
    "career":      0.20,   # career history quality
    "behavioral":  0.12,   # platform activity signals
    "location":    0.05,   # location preference
    "experience":  0.03,   # years-of-experience sweet spot
}


def load_candidates(path):
    """Load all candidates from a JSONL file. Returns a list of dicts."""
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def compute_semantic_scores_bm25(texts, jd_text):
    """
    BM25-based fallback when no internet / model not downloaded.
    Faster and works completely offline. Less accurate than embeddings.
    """
    from rank_bm25 import BM25Okapi

    def tokenize(text):
        return text.lower().split()

    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    jd_tokens = tokenize(jd_text)
    raw_scores = bm25.get_scores(jd_tokens)

    # Normalize to 0-1
    max_s = raw_scores.max()
    if max_s > 0:
        raw_scores = raw_scores / max_s
    return raw_scores


def compute_semantic_scores(texts, jd_text, use_bm25=False):
    """
    Encode all candidate texts + JD text with a small fast model,
    then return cosine similarities.
    Uses all-MiniLM-L6-v2 which is fast enough for 100K on CPU.

    Falls back to BM25 if the model cannot be loaded.
    """
    if use_bm25:
        print("Using BM25 (offline mode)...")
        return compute_semantic_scores_bm25(texts, jd_text)

    try:
        print("Loading sentence-transformer model (this happens once)...")
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Encoding JD...")
        jd_embedding = model.encode([jd_text], batch_size=1, show_progress_bar=False)

        print(f"Encoding {len(texts)} candidate texts (takes ~2-3 minutes on CPU)...")
        candidate_embeddings = model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        scores = cosine_similarity(jd_embedding, candidate_embeddings)[0]
        scores = (scores + 1) / 2
        return scores

    except Exception as e:
        print(f"  Model load failed ({e})")
        print("  Falling back to BM25 (keyword-based) scoring...")
        return compute_semantic_scores_bm25(texts, jd_text)


def compute_final_score(features, semantic_score):
    """Combine all component scores into one final score."""

    # Hard disqualifiers — score to 0
    if features["is_honeypot"]:
        return 0.0

    score = (
        WEIGHTS["semantic"]   * semantic_score +
        WEIGHTS["skills"]     * features["skills_score"] +
        WEIGHTS["career"]     * features["career_score"] +
        WEIGHTS["behavioral"] * features["behavioral_score"] +
        WEIGHTS["location"]   * features["location_score"] +
        WEIGHTS["experience"] * features["experience_score"]
    )

    # Soft penalty for consulting-only (not a hard zero, but significant)
    if features["is_consulting_only"]:
        score *= 0.60

    # Soft boost for clearly active and available candidates
    signals_boost = 0.0
    if features["behavioral_score"] > 0.75 and features["notice_days"] <= 30:
        signals_boost = 0.02

    return min(1.0, score + signals_boost)


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob hackathon")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--bm25", action="store_true",
                        help="Use BM25 (offline/fast mode)")
    args = parser.parse_args()

    start_time = time.time()

    # ── Step 1: Load candidates ───────────────────────────────────────────────
    print(f"\n[1/5] Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"      Loaded {len(candidates):,} candidates")

    # ── Step 2: Extract features ──────────────────────────────────────────────
    print(f"\n[2/5] Extracting features from each candidate...")
    all_features = []
    texts = []
    for c in tqdm(candidates, desc="Extracting"):
        feat = extract_features(c)
        all_features.append(feat)
        texts.append(feat["text"])

    honeypot_count = sum(1 for f in all_features if f["is_honeypot"])
    print(f"      Detected {honeypot_count} honeypot candidates (will be scored 0)")

    # ── Step 3: Semantic similarity ───────────────────────────────────────────
    print(f"\n[3/5] Computing semantic similarity scores...")
    semantic_scores = compute_semantic_scores(texts, JD_TEXT, use_bm25=args.bm25)

    # ── Step 4: Combine scores ────────────────────────────────────────────────
    print(f"\n[4/5] Computing final scores...")
    results = []
    for i, (c, feat) in enumerate(zip(candidates, all_features)):
        final = compute_final_score(feat, semantic_scores[i])
        results.append({
            "candidate": c,
            "features": feat,
            "score": final
        })

    # Round first, THEN sort so that equal-rounded scores get tie-broken by candidate_id
    for r in results:
        r["score"] = round(r["score"], 4)
    results.sort(key=lambda x: (-x["score"], x["candidate"]["candidate_id"]))
    top_100 = results[:100]

    # ── Step 5: Write CSV ─────────────────────────────────────────────────────
    print(f"\n[5/5] Writing submission to {args.out}...")

    # Scores are already non-increasing from the sort above.
    # No clamping needed — sort by (-score, candidate_id) ensures both ordering rules.

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(top_100, start=1):
            cid = r["candidate"]["candidate_id"]
            score = round(r["score"], 4)
            reasoning = generate_reasoning(
                r["candidate"], r["features"], r["score"]
            )
            writer.writerow([cid, rank, score, reasoning])

    elapsed = time.time() - start_time
    print(f"\n✓ Done in {elapsed:.1f}s")
    print(f"✓ Submission written to: {args.out}")
    print(f"✓ Top score: {top_100[0]['score']:.4f}")
    print(f"✓ Rank 100 score: {top_100[99]['score']:.4f}")
    print(f"\nTop 5 candidates:")
    for i, r in enumerate(top_100[:5], 1):
        p = r["candidate"]["profile"]
        print(f"  {i}. {p['anonymized_name']} | {p['current_title']} @ {p['current_company']}"
              f" | {p['location']} | score={r['score']:.4f}")


if __name__ == "__main__":
    main()
