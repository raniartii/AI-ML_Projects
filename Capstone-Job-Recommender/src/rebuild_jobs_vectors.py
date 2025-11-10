# src/rebuild_jobs_vectors.py
from __future__ import annotations
from pathlib import Path
import os, ast, re, json, sys, time
import numpy as np
import pandas as pd
from scipy import sparse as sp

from .paths import (
    JOBS_UNIFIED_FIXED_PARQUET, JOBS_UNIFIED_PARQUET,
    SKILLS_VOCAB_CSV, JOBS_VECTORS_NPZ, PREPARED,
)

# --- controls ---
SUBSET = int(os.environ.get("ROWS", "15000"))   # quick test: ROWS=500 python -m src.rebuild_jobs_vectors
LOG_EVERY = int(os.environ.get("LOG_EVERY", "500"))

# ---------------- helpers ----------------
def log(msg: str):
    print(msg, flush=True)

def _choose_parquet() -> Path:
    p = Path(JOBS_UNIFIED_FIXED_PARQUET)
    return p if p.exists() else Path(JOBS_UNIFIED_PARQUET)

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("scikit learn","scikit-learn").replace("powerbi","power bi")
    s = re.sub(r"\s+", " ", s)
    return s

_SPLITTER = re.compile(r"[,\|;/•\n]+")
TOKENIZER = re.compile(r"[A-Za-z0-9+.#-]+")

def _coerce_list(x):
    if x is None: return []
    if isinstance(x, (list, set, tuple)): return list(x)
    if isinstance(x, str):
        # try python literal list first
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, set, tuple)): return list(v)
        except Exception:
            pass
        # fallback: split on common separators
        return [p.strip() for p in _SPITTER.split(x) if p.strip()]
    return [x]

def _unify_skills_col(df: pd.DataFrame) -> str:
    preferred = [
        "skills_final","skills","skills_parsed","skills_from_desc","skills_from_col",
        "job_skills","skill_list","skills_set"
    ]
    for c in preferred:
        if c in df.columns:
            df[c] = df[c].apply(_coerce_list)
            return c
    # any column that looks like skills
    for c in df.columns:
        if "skill" in c.lower():
            df[c] = df[c].apply(_coerce_list)
            return c
    df["skills"] = [[] for _ in range(len(df))]
    return "skills"

def _load_aliases() -> dict[str, str]:
    p = Path(PREPARED) / "aliases.json"
    if p.exists():
        try:
            d = json.loads(p.read_text("utf-8"))
            return {str(k).lower(): str(v).lower() for k, v in d.items()}
        except Exception:
            return {}
    return {}

ALIASES = _load_aliases()

def _combine_text(row: pd.Series) -> str:
    parts = []
    for c in ["job_title","title","position","role",
              "job_description","description","full_description","desc","details"]:
        if c in row and isinstance(row[c], str) and row[c].strip():
            parts.append(row[c])
    txt = " ".join(parts).replace("\n", " ").replace("\r", " ")
    return _norm(txt)

# -------------- load data --------------
t0 = time.time()
jobs_path = _choose_parquet()
log(f"[1/6] Loading jobs parquet: {jobs_path}")
df = pd.read_parquet(jobs_path)
log(f"       Loaded rows={len(df):,}")

skills_col = _unify_skills_col(df)
log(f"[2/6] Skills column = {skills_col}")

# prefer rows with non-empty skills; deterministic subset
df = df[df[skills_col].map(lambda v: len(v) > 0)].head(SUBSET).copy()
df.reset_index(drop=False, inplace=True)  # keep original index for mapping
row_index_labels = df["index"].to_numpy()
df.drop(columns=["index"], inplace=True)
log(f"[3/6] Subset rows (non-empty skills, head): {len(df):,}")

# -------------- vocab (ORDER = CSV) --------------
log(f"[4/6] Loading vocab from CSV: {SKILLS_VOCAB_CSV}")
csv = pd.read_csv(SKILLS_VOCAB_CSV, low_memory=False)
vocab_series = csv.iloc[:, 0].astype(str).str.lower().str.strip()
vocab_series = vocab_series[vocab_series != ""]
vocab = pd.unique(vocab_series).tolist()  # preserve first-seen order
tok2idx = {t: i for i, t in enumerate(vocab)}
VSET = set(vocab)
log(f"       Vocab size = {len(vocab)}")

# staple monitors (live nnz-ish during build)
STAPLES = ["python","sql","docker","aws","tableau"]
staple_idx = {k: tok2idx[k] for k in STAPLES if k in tok2idx}
staple_rows = {k: 0 for k in staple_idx.keys()}

def map_token_to_vocab(t: str) -> str | None:
    """Map raw token -> canonical vocab token (fast)."""
    t = _norm(t)
    if not t: return None
    # 1) direct
    if t in VSET: return t
    # 2) alias
    a = ALIASES.get(t)
    if a and a in VSET: return a
    # 3) quick canonicalizations
    if "python" in t and "python" in VSET: return "python"
    if "sql" in t and "sql" in VSET: return "sql"
    if ("aws" in t or "amazon web services" in t):
        return "aws" if "aws" in VSET else ("amazon web services" if "amazon web services" in VSET else None)
    if "docker" in t and "docker" in VSET: return "docker"
    if "tableau" in t and "tableau" in VSET: return "tableau"
    return None

# -------------- build CSR (with live progress) --------------
log("[5/6] Building CSR...")
data, indices, indptr = [], [], [0]
last_log = time.time()

for r, row in enumerate(df.itertuples(index=False), start=1):
    chosen: set[int] = set()

    # A) from structured skills column
    for s in getattr(row, skills_col):
        m = map_token_to_vocab(str(s))
        if m is not None:
            j = tok2idx.get(m)
            if j is not None:
                chosen.add(j)

    # B) fallback: tokenize title/desc and intersect with vocab/aliases (fast)
    if not chosen:
        blob = _combine_text(pd.Series(row._asdict()))
        if blob:
            toks = { _norm(t) for t in TOKENIZER.findall(blob) }
            # direct matches
            hits = toks & VSET
            # alias matches
            if ALIASES:
                hits |= { ALIASES[t] for t in toks if t in ALIASES and ALIASES[t] in VSET }
            for m in hits:
                j = tok2idx[m]
                chosen.add(j)

    if chosen:
        chosen_sorted = sorted(chosen)
        indices.extend(chosen_sorted)
        data.extend([1.0] * len(chosen_sorted))
        # staple live counters
        for k, j in staple_idx.items():
            if j in chosen_sorted:
                staple_rows[k] += 1

    indptr.append(len(data))

    # live progress
    if r % LOG_EVERY == 0:
        now = time.time()
        log(f"   processed {r:,}/{len(df):,} rows "
            f"(+{r - (len(df) if r==len(df) else r-LOG_EVERY):,} since last) | "
            f"staples rows so far: " +
            ", ".join(f"{k}:{staple_rows[k]}" for k in staple_rows))

X_csr = sp.csr_matrix(
    (np.array(data, dtype=np.float32),
     np.array(indices, dtype=np.int32),
     np.array(indptr, dtype=np.int32)),
    shape=(len(indptr) - 1, len(vocab))
)

# -------------- save vectors + vocab + row map --------------
out_npz = Path(JOBS_VECTORS_NPZ)
out_npz.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    out_npz,
    data=X_csr.data,
    indices=X_csr.indices,
    indptr=X_csr.indptr,
    shape=X_csr.shape,
    format=np.array(b"csr"),
    vocab=np.array(vocab, dtype="U"),  # exact column order
)
np.save(out_npz.with_suffix(".row_idx.npy"), row_index_labels)
with open(out_npz.with_suffix(".rows.txt"), "w") as f:
    f.write(str(X_csr.shape[0]))

log(f"[6/6] Saved: {out_npz.name}  shape={X_csr.shape}  vocab={len(vocab)}  rows={len(row_index_labels)}")
for k, j in staple_idx.items():
    log(f"   staple '{k}' rows with 1s: {staple_rows[k]}")

# --- exact column index + nnz per staple (matches your earlier format) ---
for k, j in staple_idx.items():
    nnz = int(X_csr[:, j].nnz)
    log(f"{k:8s} col={j:3d} nnz={nnz}")
log(f"Done in {time.time() - t0:.1f} sec")
