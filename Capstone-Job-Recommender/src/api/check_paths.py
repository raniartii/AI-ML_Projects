# src/api/main.py
from __future__ import annotations

import os, json, re, ast
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------
# CORS & APP
# -----------------------
app = FastAPI(title="Resume→Job→Project API")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Paths (flexible)
# -----------------------
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets"
PREP = DATA / "prepared"
FINAL = DATA / "final" / "jobs_skills"
VOCAB = FINAL / "vocab"

# Likely files in your tree:
CANDIDATE_PARQUETS = [
    FINAL / "jobs_unified_with_skills_FIXED.parquet",
    FINAL / "jobs_unified_with_skills.parquet",
    PREP / "jobs_unified.parquet",
]

CANDIDATE_CSVS = [
    FINAL / "jobs_unified_with_skills.csv",
    DATA / "processed" / "jobs_merged_unified.csv",
]

SKILLS_TAXONOMY = PREP / "skills_taxonomy.json"
ALIASES_FILE = PREP / "aliases.json"

# Optional precomputed vectors (if you produced them)
VECTORS_NPZ = FINAL / "vocab" / "jobs_vectors.npz"
VECTORS_IDX = FINAL / "vocab" / "jobs_vectors.row_idx.npy"
VECTORS_ROWS = FINAL / "vocab" / "jobs_vectors.rows.txt"

# print("\nUsing paths:", "\nREPO", REPO, "\nDATA", DATA, "\nPREP", PREP, "\nFINAL", FINAL)
# print("\n\nUsing data paths:", "\nCANDIDATE_PARQUETS", CANDIDATE_PARQUETS, "\nCANDIDATE_CSVS", CANDIDATE_CSVS)
# print("\n\nUsing skills taxonomy:", "\nSKILLS_TAXONOMY", SKILLS_TAXONOMY)
# print("\n\nUsing aliases:", "\nALIASES_FILE", ALIASES_FILE)
# print("\n\nUsing vectors:", "\nVECTORS_NPZ ", VECTORS_NPZ,  "\nVECTORS_IDX", VECTORS_IDX, "\nVECTORS_ROWS", VECTORS_ROWS)
#