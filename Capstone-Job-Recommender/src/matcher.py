# src/matcher.py
from __future__ import annotations

# --- bootstrap so this file works as module OR script ---
try:
    from .paths import (
        JOBS_UNIFIED_FIXED_PARQUET, JOBS_UNIFIED_PARQUET,
        SKILLS_VOCAB_CSV, JOBS_VECTORS_NPZ, PREPARED,
    )
except ImportError:
    import sys, pathlib
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from src.paths import (  # type: ignore
        JOBS_UNIFIED_FIXED_PARQUET, JOBS_UNIFIED_PARQUET,
        SKILLS_VOCAB_CSV, JOBS_VECTORS_NPZ, PREPARED,
    )

import re, ast, json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse as sp

# ----------------------------
# Path helpers
# ----------------------------
def _choose_parquet() -> Path:
    p = Path(JOBS_UNIFIED_FIXED_PARQUET)
    return p if p.exists() else Path(JOBS_UNIFIED_PARQUET)

def _must_exist(p: Path, what: str):
    if not Path(p).exists():
        raise FileNotFoundError(f"Missing {what}: {p}")

# ----------------------------
# Skill parsing / normalization
# ----------------------------
def _norm_skill(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = s.replace("scikit learn", "scikit-learn").replace("powerbi","power bi")
    s = re.sub(r"\s+"," ", s)
    return s

def _load_aliases() -> dict:
    p = Path(PREPARED) / "aliases.json"
    if p.exists():
        try:
            return {k.lower(): v.lower() for k, v in json.loads(p.read_text("utf-8")).items()}
        except Exception:
            pass
    return {}
_ALIASES = _load_aliases()

_SPLIT = re.compile(r"[,\|;/•\n]+")
_WORDISH = re.compile(r"[a-zA-Z][a-zA-Z0-9\+\.#\- ]{0,50}")

def extract_resume_skills(resume_text: str) -> List[str]:
    if not resume_text or not isinstance(resume_text, str): return []
    raw = [t.strip() for t in _SPLIT.split(resume_text) if t.strip()]
    toks: List[str] = []
    for t in raw:
        if len(t) > 64 and "," in t:  # drop sentence-y chunks
            continue
        m = _WORDISH.findall(t.lower())
        if not m: 
            continue
        tok = _norm_skill(" ".join(m))
        if not tok: 
            continue
        # do NOT alias blindly here; aliasing can drop tokens not in vocab
        toks.append(tok)
    # de-dup keep order
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

# ----------------------------
# Vocab + vectors
# ----------------------------
def _read_vocab_csv(path) -> list[str]:
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    candidates = ["skill","skills","skill_set","skills_final","token","term","word","name"]
    col = next((c for c in df.columns if c in candidates), df.columns[0])
    vocab = (
        df[col].astype(str).str.strip().str.lower()
          .replace("", pd.NA).dropna().unique().tolist()
    )
    return vocab

# ----------------------------
# Jobs dataframe helpers
# ----------------------------
def _ensure_list(x):
    if x is None: return []
    if isinstance(x, (list, set, tuple)): return list(x)
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, (list, set, tuple)): return list(val)
        except Exception:
            return [x]
    return [x]

_SPLITTER = re.compile(r"[,\|;/•\n]+")
def _coerce_skills_any(x):
    if x is None: return []
    if isinstance(x, (list, set, tuple)): return list(x)
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, (list, set, tuple)): return list(val)
        except Exception:
            pass
        return [p.strip() for p in _SPLITTER.split(x) if p.strip()]
    return [x]

def _unify_skills_col(df):
    preferred = [
        "skills_final","skills","skills_parsed","skills_from_desc","skills_from_col",
        "job_skills","skill_list","skills_set"
    ]
    for c in preferred:
        if c in df.columns:
            df[c] = df[c].apply(_coerce_skills_any)
            return c
    for c in df.columns:
        if "skill" in c.lower():
            df[c] = df[c].apply(_coerce_skills_any)
            return c
    df["skills"] = [[] for _ in range(len(df))]
    return "skills"

def _filter_non_empty_skills(df: pd.DataFrame) -> pd.DataFrame:
    col = _unify_skills_col(df)
    return df[df[col].map(lambda v: len(v) > 0)]

def _align_df_to_vectors(df: pd.DataFrame, X: sp.spmatrix) -> pd.DataFrame:
    """Align jobs DF rows to vector rows via sidecars or heuristics."""
    n_vec = X.shape[0]
    if n_vec == len(df):
        return df
    npz = Path(JOBS_VECTORS_NPZ)
    side_idx = npz.with_suffix(".row_idx.npy")
    side_rows = npz.with_suffix(".rows.txt")

    if side_idx.exists():
        idx = np.load(side_idx)
        try:
            if isinstance(df.index, pd.RangeIndex) and np.issubdtype(idx.dtype, np.integer):
                return df.iloc[idx]
            return df.loc[idx]
        except Exception as e:
            print(f"[matcher] row_idx.npy present but could not align ({e}); falling back.")

    if side_rows.exists():
        try:
            n_side = int(open(side_rows).read().strip())
            if n_side != n_vec:
                print(f"[matcher] rows.txt={n_side} but vectors={n_vec}; proceeding with vectors={n_vec}.")
        except Exception:
            pass

    df2 = _filter_non_empty_skills(df.copy())
    if len(df2) >= n_vec:
        return df2.head(n_vec)

    print(f"[matcher] Align fallback: truncating DF {len(df)} → {n_vec} rows.")
    return df.head(n_vec)

# ----------------------------
# New helpers for mapping/augmentation
# ----------------------------
TOKENIZER = re.compile(r"[A-Za-z0-9+.#-]+")

def _map_to_vocab(tokens: list[str], vocab: list[str]) -> list[str]:
    """Map resume tokens into vocab with safe aliasing/canonicals; drop unknowns."""
    vset = set(vocab)
    out: list[str] = []
    for t in tokens:
        t0 = _norm_skill(t)
        m = _ALIASES.get(t0)
        if m and m in vset: out.append(m)
        elif t0 in vset: out.append(t0)
        else:
            if "python" in t0 and "python" in vset: out.append("python")
            elif "sql" in t0 and "sql" in vset: out.append("sql")
            elif ("aws" in t0 or "amazon web services" in t0):
                out.append("aws" if "aws" in vset else ("amazon web services" if "amazon web services" in vset else t0))
            elif "docker" in t0 and "docker" in vset: out.append("docker")
            elif "tableau" in t0 and "tableau" in vset: out.append("tableau")
    # dedupe keep order
    seen, keep = set(), []
    for x in out:
        if x not in seen:
            keep.append(x); seen.add(x)
    return keep

def _augment_job_skills_from_text(row: pd.Series, vocab: list[str]) -> list[str]:
    """Augment job skills with tokens mined from title/description (for gap view)."""
    vset = set(vocab)
    parts = []
    for c in ["job_title","title","position","role","job_description","description","full_description","desc","details"]:
        if c in row and isinstance(row[c], str) and row[c].strip():
            parts.append(row[c])
    if not parts: return []
    blob = " ".join(parts).replace("\n"," ").replace("\r"," ").lower()
    toks = {_norm_skill(t) for t in TOKENIZER.findall(blob)}
    hits = set()
    for t in toks:
        if t in vset: hits.add(t)
        elif t in _ALIASES and _ALIASES[t] in vset: hits.add(_ALIASES[t])
        else:
            if "python" in t and "python" in vset: hits.add("python")
            elif "sql" in t and "sql" in vset: hits.add("sql")
            elif ("aws" in t or "amazon web services" in t):
                hits.add("aws" if "aws" in vset else ("amazon web services" if "amazon web services" in vset else ""))
            elif "docker" in t and "docker" in vset: hits.add("docker")
            elif "tableau" in t and "tableau" in vset: hits.add("tableau")
    hits.discard("")
    return sorted(hits)

# ----------------------------
# Public API: load assets (single, final version)
# ----------------------------
def load_jobs_assets() -> tuple[pd.DataFrame, list[str], sp.spmatrix]:
    jobs_path = _choose_parquet()
    _must_exist(jobs_path, "jobs parquet")
    _must_exist(SKILLS_VOCAB_CSV, "skills vocab csv")
    _must_exist(JOBS_VECTORS_NPZ, "jobs vectors npz")

    df_jobs = pd.read_parquet(jobs_path)
    csv_vocab = _read_vocab_csv(SKILLS_VOCAB_CSV)

    # load vectors + (ordered) vocab from NPZ
    with np.load(JOBS_VECTORS_NPZ, allow_pickle=False) as z:
        X_jobs = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                               shape=tuple(z["shape"])).astype(np.float32, copy=False)
        npz_vocab = z["vocab"].astype(str).tolist() if "vocab" in z.files else None

    vocab = csv_vocab
    if npz_vocab and len(npz_vocab) == len(csv_vocab) and set(npz_vocab) == set(csv_vocab):
        print("[matcher] Using vocab order from NPZ to align with vectors.")
        vocab = npz_vocab

    if X_jobs.shape[1] != len(vocab):
        raise ValueError(f"Vector width ({X_jobs.shape[1]}) != vocab size ({len(vocab)}). Rebuild with the SAME vocab.")

    df_jobs = _align_df_to_vectors(df_jobs, X_jobs)
    return df_jobs, vocab, X_jobs

def vec_from_skills(skills: List[str], vocab_size: int, vocab: List[str] | None = None) -> np.ndarray:
    v = np.zeros((1, vocab_size), dtype=np.float32)
    if not skills or vocab is None: return v
    idx = {s: i for i, s in enumerate(vocab)}
    for s in skills:
        s = _norm_skill(s)
        if s in idx:
            v[0, idx[s]] = 1.0
    return v

def match_jobs(resume_text: str, field: str | None = None, topn: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    df_jobs, vocab, X_jobs = load_jobs_assets()
    have = extract_resume_skills(resume_text)
    have = _map_to_vocab(have, vocab)  # keep only tokens the model knows

    x = vec_from_skills(have, len(vocab), vocab=vocab)
    x = sp.csr_matrix(x)
    scores = cosine_similarity(x, X_jobs)[0]

    out = df_jobs.copy()
    out["score"] = scores

    if field and ("field" in out.columns):
        out = out[out["field"].astype(str).str.contains(str(field), case=False, na=False)]

    # normalize skills col for display/debug
    skills_col = _unify_skills_col(out)
    out[skills_col] = out[skills_col].apply(_coerce_skills_any)

    out = out.sort_values("score", ascending=False).head(topn).reset_index(drop=True)
    return out, have

def add_overlap_missing(df_top: pd.DataFrame, resume_skills: List[str], vocab: list[str] | None = None) -> pd.DataFrame:
    # allow passing vocab (if caller has it); else infer from assets
    if vocab is None:
        try:
            _, vocab_loaded, _ = load_jobs_assets()
            vocab = vocab_loaded
        except Exception:
            vocab = []

    have = set(_map_to_vocab(resume_skills, vocab or []))
    df = df_top.copy()
    skills_col = _unify_skills_col(df)

    def _row(row):
        req = {_norm_skill(s) for s in _coerce_skills_any(row[skills_col])}
        # augment with text-derived tokens (title/desc)
        aug = set(_augment_job_skills_from_text(row, vocab or []))
        req = {s for s in (req | aug) if s}  # merged requirement set

        overlap = sorted(req & have)
        missing = sorted(req - have)
        return pd.Series({
            "overlap_skills": overlap,
            "missing_skills": missing,
            "overlap_count": len(overlap),
            "missing_count": len(missing),
        })

    aug = df.apply(_row, axis=1)
    return pd.concat([df.reset_index(drop=True), aug], axis=1)

if __name__ == "__main__":
    try:
        from src.matcher import load_jobs_assets, match_jobs, add_overlap_missing
        
        top, have = match_jobs("Tableau, Docker, AWS", field="Data", topn=10)
        

        demo = "Tableau, Docker, AWS; built ETL pipelines and dashboards."
        df, vocab, X = load_jobs_assets()
        top, have = match_jobs(demo, field="Data", topn=10)
        top = add_overlap_missing(top, have, vocab=vocab)  # pass vocab to avoid double load
        print("Detected resume skills:", have)
        print(top[["score","job_title","field","missing_skills","overlap_skills"]].head(10).to_string(index=False))
    except Exception as e:
        import traceback as _tb
        print("❌ Demo failed:", e)
        _tb.print_exc()
