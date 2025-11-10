# src/paths.py (1-time helper)

from pathlib import Path

# repo root = this file -> src -> parent
ROOT = Path(__file__).resolve().parents[1]

DATASETS = ROOT / "datasets"
PREPARED = DATASETS / "prepared"
FINAL = DATASETS / "final"
JOBS_SKILLS = FINAL / "jobs_skills"
VOCAB_DIR = JOBS_SKILLS / "vocab"

# Files in your tree
PROJECT_IDEAS_JSON = PREPARED / "project_ideas.json"
JOBS_UNIFIED_PARQUET = JOBS_SKILLS / "jobs_unified_with_skills.parquet"
JOBS_UNIFIED_FIXED_PARQUET = JOBS_SKILLS / "jobs_unified_with_skills_FIXED.parquet"
SKILLS_VOCAB_CSV = VOCAB_DIR / "skills_vocab.csv"
JOBS_VECTORS_NPZ = VOCAB_DIR / "jobs_vectors.npz"

# print("Data paths:", "\nDatasets:, ", DATASETS, "\nprepared:", PREPARED, "\nfinal:", FINAL)
# print(ROOT)