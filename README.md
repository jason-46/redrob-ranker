# Redrob Hackathon — Candidate Ranker

## What this does
Ranks 100,000 candidates against a Senior AI Engineer job description.
Produces a top-100 CSV submission in the required format.

## How to run (complete steps)

### Step 1 — Install Python (if you don't have it)
Download Python 3.11 from https://www.python.org/downloads/
During installation, check "Add Python to PATH"

### Step 2 — Set up the project folder
```bash
# Open terminal (Command Prompt on Windows, Terminal on Mac/Linux)
# Navigate to where you want the project
cd Desktop

# Clone or create the project folder
mkdir redrob-ranker
cd redrob-ranker
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
This installs: sentence-transformers, faiss-cpu, numpy, pandas, tqdm, scikit-learn

### Step 4 — Put your data file here
Copy `candidates.jsonl` into the `redrob-ranker/` folder.
(The file is 100,000 lines, one candidate per line)

### Step 5 — Run the ranker
```bash
python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv
```
Replace `team_xxx` with your actual registered team ID.

Expected runtime: 3-4 minutes on a modern laptop (CPU only)

### Step 6 — Validate your submission
```bash
python validate_submission.py team_xxx.csv
```
Copy `validate_submission.py` from the hackathon bundle into this folder first.
Fix any errors it reports before uploading.

### Step 7 — Submit
Upload `team_xxx.csv` via the hackathon portal.

---

## File structure
```
redrob-ranker/
├── rank.py                  ← main script (run this)
├── requirements.txt         ← install these first
├── validate_submission.py   ← copy from hackathon bundle
├── candidates.jsonl         ← copy from hackathon bundle
├── submission_metadata.yaml ← fill in your team info
└── src/
    ├── feature_extractor.py ← scoring logic
    └── reasoner.py          ← reasoning text generator
```

## How scoring works

Each candidate gets a weighted combination of 6 scores:

| Score | Weight | What it measures |
|---|---|---|
| Semantic similarity | 35% | How well their profile text matches the JD |
| Skills fit | 25% | Direct match to required skills (embeddings, vector DBs, etc.) |
| Career quality | 20% | Product company history, production ML work |
| Behavioral signals | 12% | Platform activity, response rate, availability |
| Location | 5% | Match to Pune/Noida/Hyderabad/Mumbai/Delhi NCR |
| Experience range | 3% | Closeness to 5-9 year sweet spot |

**Disqualifiers** (score forced to 0):
- Honeypot candidates (impossible career timelines)

**Soft penalties** (score multiplied down):
- Consulting-only career (TCS, Infosys, Wipro, etc.): ×0.60
