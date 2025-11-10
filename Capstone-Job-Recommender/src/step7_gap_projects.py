# src/step7_gap_projects.py
from __future__ import annotations

import re, ast, json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .paths import PROJECT_IDEAS_JSON
from .matcher import (
    load_jobs_assets,  # returns df_jobs, vocab(list[str]), X_jobs (CSR)
    extract_resume_skills,
    vec_from_skills,
    add_overlap_missing,
)

# -------------------------
# Normalization & helpers
# -------------------------
def _as_str(s) -> str:
    if s is None: return ""
    if isinstance(s, (list, set, tuple)):
        return " ".join(map(str, s))
    if isinstance(s, np.ndarray):
        try:
            return " ".join(map(str, s.ravel().tolist()))
        except Exception:
            return str(s)
    return str(s)

def _norm_skill(s) -> str:
    s = _as_str(s).strip().lower()
    s = s.replace("scikit learn","scikit-learn").replace("powerbi","power bi")
    s = re.sub(r"\s+"," ", s)
    return s

_SPLITTER = re.compile(r"[,\|;/•\n]+")

def _coerce_list(x):
    if x is None: return []
    if isinstance(x, (list, set, tuple)): return list(x)
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, set, tuple)): return list(v)
        except Exception:
            pass
        return [p.strip() for p in _SPLITTER.split(x) if p.strip()]
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
    for c in df.columns:
        if "skill" in c.lower():
            df[c] = df[c].apply(_coerce_list)
            return c
    df["skills"] = [[] for _ in range(len(df))]
    return "skills"

def _combine_text(row: pd.Series) -> str:
    parts = []
    for c in ["job_title","title","position","role",
              "job_description","description","full_description","desc","details"]:
        if c in row and isinstance(row[c], str) and row[c].strip():
            parts.append(row[c])
    return " ".join(parts).replace("\n"," ").replace("\r"," ")

# -------------------------
# 2) Parsing Resume
# -------------------------
def extract_skills(text: str) -> List[str]:
    return [_norm_skill(s) for s in extract_resume_skills(text)]

# put near the other regexes
_BULLET = re.compile(r"^\s*(?:[-*•–—]|[0-9]+\.)\s+")
_PROJ_HEAD = re.compile(r"^\s*(projects?|academic projects?|personal projects?|selected projects?)\s*:?\s*$", re.I)
_STOP_HEAD = re.compile(r"^\s*(education|experience|work experience|skills|summary|certifications|achievements|publications)\s*:?\s*$", re.I)

def extract_projects(text: str) -> List[Dict[str, Any]]:
    """
    Extract projects from a 'Projects' section, supporting:
      - headings like 'Projects', 'Academic Projects', 'Personal Projects'
      - bullets: -, •, *, 1., 2., ...
      - blank-line separated blocks
    Returns [{title, bullets, skills}]
    """
    if not text or not isinstance(text, str):
        return []

    t = text.replace("\r", "")
    lines = [ln.rstrip() for ln in t.split("\n")]

    # find first projects-like heading
    start = None
    for i, ln in enumerate(lines):
        if _PROJ_HEAD.match(ln):
            start = i
            break
    if start is None:
        return []  # no explicit section; policy will still use estimator

    # capture until next major heading
    chunk = []
    for ln in lines[start + 1:]:
        if _STOP_HEAD.match(ln):
            break
        chunk.append(ln)

    # split into candidate items:
    #  - every bullet line starts a new project
    #  - if no bullets, use blank-line blocks
    items: List[List[str]] = []
    cur: List[str] = []

    def flush():
        nonlocal cur, items
        if cur:
            items.append(cur)
            cur = []

    saw_bullets = any(_BULLET.match(ln) for ln in chunk)

    if saw_bullets:
        for ln in chunk:
            if _BULLET.match(ln):
                flush()
                cur = [ _BULLET.sub("", ln).strip() ]  # start new item with title line content
            else:
                if ln.strip():
                    cur.append(ln.strip())
        flush()
    else:
        # fallback: blank-line separated blocks
        for ln in chunk:
            if not ln.strip():
                flush()
            else:
                cur.append(ln.strip())
        flush()

    projects: List[Dict[str, Any]] = []
    for blk in items:
        if not blk:
            continue
        title = blk[0][:140]
        bullets = [b for b in blk[1:] if b]
        raw = " ".join(blk)
        sk = extract_skills(raw)
        projects.append({"title": title, "bullets": bullets, "skills": sk})

    return projects


# -------------------------
# 3) Models (A/B/C + Hybrid)
# -------------------------
class Ranker:
    """
    A: multi-hot cosine (precomputed job matrix)
    B: TF-IDF over job skills
    C: TF-IDF over job title+desc
    """
    def __init__(self):
        self.df_jobs, self.vocab, self.X_jobs = load_jobs_assets()
        self.skills_col = _unify_skills_col(self.df_jobs)
        skills_docs = [" ".join(_norm_skill(s) for s in row) for row in self.df_jobs[self.skills_col]]
        text_docs = [_combine_text(row) for _, row in self.df_jobs.iterrows()]
        # B: lock vocab order for interpretability
        self.vec_skills = TfidfVectorizer(
            tokenizer=None, preprocessor=None, lowercase=True,
            token_pattern=r"[A-Za-z0-9+.#-]+",
            vocabulary={t:i for i,t in enumerate(self.vocab)},
            norm="l2", use_idf=True, sublinear_tf=True
        )
        self.X_tfidf_skills = self.vec_skills.fit_transform(skills_docs)
        # C: free vocab on title+desc
        self.vec_text = TfidfVectorizer(
            tokenizer=None, preprocessor=None, lowercase=True,
            token_pattern=r"[A-Za-z0-9+.#-]+",
            ngram_range=(1,2), min_df=2, norm="l2", use_idf=True, sublinear_tf=True
        )
        self.X_tfidf_text = self.vec_text.fit_transform(text_docs)

    def _vec_resume_A(self, resume_text: str) -> sp.csr_matrix:
        have = [h for h in extract_skills(resume_text) if h in self.vocab]
        x = vec_from_skills(have, len(self.vocab), vocab=self.vocab)
        return sp.csr_matrix(x)

    def _vec_resume_B(self, resume_text: str) -> sp.csr_matrix:
        tokens = extract_skills(resume_text)
        return self.vec_skills.transform([" ".join(tokens)])

    def _vec_resume_C(self, resume_text: str) -> sp.csr_matrix:
        return self.vec_text.transform([resume_text])

    def score(self, resume_text: str, alpha=0.3, beta=0.3, gamma=0.4) -> pd.DataFrame:
        rA = self._vec_resume_A(resume_text); sA = cosine_similarity(rA, self.X_jobs)[0]
        rB = self._vec_resume_B(resume_text); sB = cosine_similarity(rB, self.X_tfidf_skills)[0]
        rC = self._vec_resume_C(resume_text); sC = cosine_similarity(rC, self.X_tfidf_text)[0]
        out = self.df_jobs.copy()
        out["score_A"], out["score_B"], out["score_C"] = sA, sB, sC
        out["score"] = 0.3*out["score_A"] + 0.3*out["score_B"] + 0.4*out["score_C"] if (alpha,beta,gamma)==(0.3,0.3,0.4) \
                       else alpha*out["score_A"] + beta*out["score_B"] + gamma*out["score_C"]
        return out

def rank_jobs_hybrid(resume_text: str, field: str|None=None,
                     alpha=0.3, beta=0.3, gamma=0.4, topn=15) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r = Ranker()
    scored = r.score(resume_text, alpha=alpha, beta=beta, gamma=gamma)
    if field and ("field" in scored.columns):
        scored = scored[scored["field"].astype(str).str.contains(str(field), case=False, na=False)]
    top = scored.sort_values("score", ascending=False).head(topn).reset_index(drop=True)
    # gap view; pass vocab to avoid reloading in matcher
    top_gap = add_overlap_missing(top, extract_skills(resume_text), vocab=r.vocab)
    debug = {"alpha_beta_gamma": (alpha,beta,gamma), "skills_col": r.skills_col, "top_cols": list(top_gap.columns)}
    return top_gap, {"debug": debug, "vocab": r.vocab}

# -------------------------
# 4) Gap Analysis & Project Policy (your 4 rules)
# -------------------------
def _alignment_score(project_skills: List[str], resume_skills: List[str]) -> float:
    ps = { _norm_skill(s) for s in _coerce_list(project_skills) if s }
    rs = { _norm_skill(s) for s in _coerce_list(resume_skills) if s }
    if not ps: return 0.0
    return len(ps & rs) / float(len(ps))

# put this ABOVE evaluate_projects_policy(...)
def _estimate_project_count(resume_text: str) -> int:
    """
    Heuristic count when parsing undercounts:
    - counts project-ish keywords
    - counts bullet/numbered lines and assumes ~2 bullets per project
    """
    t = (resume_text or "")
    t_lower = t.lower()

    # keywords like "Projects", "Capstone", "Hackathon", "Case study"
    kw_hits = re.findall(r"\b(projects?|capstone|hackathon|case study|case-study)\b", t_lower)

    # bullet/numbered lines: -, *, •, –, —, 1., 2., ...
    bullet_lines = re.findall(r"(?m)^\s*(?:[-*•–—]|\d+\.)\s+\S+", t)

    # rough estimate: each project usually has ~2 bullets on average
    est_from_bullets = max(len(bullet_lines) // 2, 0)

    # final: take the larger, but cap to something sane
    return max(len(kw_hits), min(est_from_bullets, 12))

# 1) Add this helper (just above evaluate_projects_policy)

_BULLET_LINE = re.compile(r"^\s*(?:[-*•–—]|\d+\.)\s+\S+")

def _count_project_groups(text: str) -> int:
    """
    Count groups of consecutive bullet/numbered lines across the *whole* resume.
    Each group is treated as one project block. Works even without a 'Projects' header.
    """
    if not text:
        return 0
    groups = 0
    in_group = False
    for ln in text.splitlines():
        if _BULLET_LINE.match(ln):
            if not in_group:
                groups += 1
                in_group = True
        else:
            in_group = False
    return groups

def evaluate_projects_policy(resume_text: str, resume_skills: List[str]) -> Dict[str, Any]:
    """
    Adaptive action plan driven by:
      - n_parsed: explicit 'Projects' section items
      - n_est: heuristic estimator (keywords + bullets)
      - n_groups: bullet groups anywhere in the doc
    Uses n_eff = max(n_parsed, n_est, n_groups).
    Alignment is computed on parsed projects when available.
    """
    # 1) counts
    projs = extract_projects(resume_text)
    n_parsed = len(projs)
    n_est    = _estimate_project_count(resume_text)
    n_groups = _count_project_groups(resume_text)
    n_eff    = max(n_parsed, n_est, n_groups)

    # 2) alignment on parsed (if we have titles & skills)
    per_proj = []
    for p in projs:
        score = _alignment_score(p.get("skills", []), resume_skills)
        per_proj.append({"title": p.get("title","(untitled)"), "score": score, "skills": p.get("skills", [])})
    aligned    = [p for p in per_proj if p["score"] > 0.0]
    misaligned = [p for p in per_proj if p["score"] == 0.0]

    # 3) thresholds & ratios
    good_min, good_max = 3, 4
    ratio = (len(aligned) / n_parsed) if n_parsed else 0.0  # only meaningful if we parsed

    policy = {
        "verdict": "",
        "actions": [],
        "n_projects_parsed": n_parsed,
        "n_projects_estimated": n_est,
        "n_projects_groups": n_groups,
        "n_projects_effective": n_eff,
        "aligned": aligned,
        "misaligned": misaligned,
    }

    # 4) decisions (n_eff drives quantity branch; parsed alignment refines the message)
    if n_eff == 0:
        policy["verdict"] = "no_projects"
        policy["actions"] = [
            "Add 3 projects that clearly use your core skills (from your Skills section).",
            "For each project, include stack (tools) and 2–3 measurable outcomes."
        ]
        policy["recommend_mode"] = "depth_from_resume"
        return policy

    if n_eff <= 2:
        policy["verdict"] = "add_more_projects"
        msg = f"You currently have about {n_eff} project(s). Add 1–2 more projects aligned with your listed skills."
        policy["actions"] = [msg, "Keep titles specific (e.g., 'Sales Forecasting with XGBoost on Retail Data')."]
        if misaligned:  # only when we parsed enough to know
            names = [p["title"] for p in misaligned][:2]
            policy["actions"].append(f"Fix mismatch: these project(s) don't use your listed skills → {names}.")
        policy["recommend_mode"] = "depth_from_resume"
        return policy

    # 3–4 projects: good volume — check alignment when we can
    if good_min <= n_eff <= good_max:
        if n_parsed == 0:
            # Can't measure alignment reliably, but quantity looks good
            policy["verdict"] = "good_volume_estimated"
            policy["actions"] = [
                f"Good — about {n_eff} projects detected.",
                "Ensure each project explicitly mentions the tools that match your Skills section."
            ]
            policy["recommend_mode"] = "none"
            return policy

        if ratio >= 0.75 and len(misaligned) <= 1:
            policy["verdict"] = "good_to_go"
            policy["actions"] = [
                f"Great — {n_parsed} projects with strong alignment ({len(aligned)}/{n_parsed}).",
                "Optionally add one depth project that showcases end-to-end ownership."
            ]
            policy["recommend_mode"] = "none"
            return policy
        else:
            policy["verdict"] = "cleanup_and_add_aligned"
            to_remove = sorted(per_proj, key=lambda d: d["score"])[: min(2, len(per_proj))]
            names = [p["title"] for p in to_remove] or ["one low-alignment project"]
            policy["actions"] = [
                f"You have {n_parsed} projects but alignment is weak ({len(aligned)}/{n_parsed}).",
                f"Remove or rework 1–2 low-alignment projects → {names}.",
                "Replace them with projects that explicitly use your core skills."
            ]
            policy["recommend_mode"] = "depth_from_resume"
            return policy

    # 5+ projects: quality > quantity
    if n_eff >= 5:
        if n_parsed and ratio < 0.6:
            policy["verdict"] = "too_many_misaligned"
            to_remove = sorted(per_proj, key=lambda d: d["score"])[: min(2, len(per_proj))]
            names = [p["title"] for p in to_remove] or ["one low-alignment project"]
            policy["actions"] = [
                f"You list ~{n_eff} projects, but many don’t match your skills.",
                f"Trim 1–2 weaker items → {names}.",
                "Keep your best 3–4 aligned projects and expand them with outcomes/impact."
            ]
            policy["recommend_mode"] = "depth_from_resume"
            return policy
        else:
            policy["verdict"] = "lots_good_focus"
            policy["actions"] = [
                f"You list ~{n_eff} projects. Great volume — focus on quality.",
                "Keep the best 3–4 aligned projects; add measurable results and links (repo/demo)."
            ]
            policy["recommend_mode"] = "none"
            return policy

    # fallback (should rarely trigger)
    policy["verdict"] = "add_targeted"
    policy["actions"] = [
        "Add 1–2 targeted projects to cover gaps for the roles you want.",
    ]
    policy["recommend_mode"] = "targeted_to_gaps"
    return policy


def _coerce_idea_item(x) -> Dict[str, Any]:
    if isinstance(x, dict):
        d = dict(x)
        if "title" not in d:
            for alt in ("name","project","idea"):
                if alt in d:
                    d["title"] = _as_str(d.pop(alt)); break
        d["title"] = _as_str(d.get("title","")).strip() or "(untitled)"
        return d
    return {"title": _as_str(x).strip()}

def load_project_ideas(path: Path = Path(PROJECT_IDEAS_JSON)) -> Dict[str, List[Dict[str, Any]]]:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ideas: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(raw, dict):
        for k, lst in raw.items():
            lst = lst if isinstance(lst, list) else [lst]
            ideas[_norm_skill(k)] = [_coerce_idea_item(c) for c in lst]
    elif isinstance(raw, list):
        ideas["general"] = [_coerce_idea_item(c) for c in raw]
    return ideas

def _rank_missing_skills(df_gap: pd.DataFrame, weight_col: str = "score", k: int = 20) -> List[Tuple[str, float]]:
    weights = df_gap[weight_col].astype(float).tolist() if weight_col in df_gap.columns else [1.0]*len(df_gap)
    counts: Dict[str, float] = {}
    for w, miss in zip(weights, df_gap["missing_skills"].tolist()):
        for m in _coerce_list(miss):
            s = _norm_skill(m)
            counts[s] = counts.get(s, 0.0) + float(w)
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:k]

def _choose_projects(skill_list: List[str],
                     ideas: Dict[str, List[Dict[str, Any]]],
                     n_projects: int = 5) -> List[Dict[str, Any]]:
    picks, used = [], set()
    for sk in skill_list:
        skn = _norm_skill(sk)
        if skn in ideas:
            for cand in ideas[skn]:
                d = _coerce_idea_item(cand)
                title = d.get("title","").strip()
                if title and title not in used:
                    picks.append({"target_skill": skn, **d})
                    used.add(title)
                    break
        if len(picks) >= n_projects: break
    # fill if needed
    if len(picks) < n_projects and "general" in ideas:
        for cand in ideas["general"]:
            d = _coerce_idea_item(cand)
            title = d.get("title","").strip()
            if title and title not in used:
                picks.append({"target_skill": "general", **d})
                used.add(title)
            if len(picks) >= n_projects: break
    return picks[:n_projects]

# -------------------------
# Final: recommend projects (combines models + policy)
# -------------------------
def recommend_projects(resume_text: str, field: str|None=None,
                       topn_jobs: int = 15, n_projects: int = 5,
                       alpha=0.3, beta=0.3, gamma=0.4) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # hybrid ranking + gaps
    top_gap, ctx = rank_jobs_hybrid(resume_text, field=field, alpha=alpha, beta=beta, gamma=gamma, topn=topn_jobs)

    # resume skills and projects
    have_skills = extract_skills(resume_text)
    projects = extract_projects(resume_text)

    # apply your policy
    policy = evaluate_projects_policy(resume_text, have_skills)

    # pick ideas source
    ideas = load_project_ideas(Path(PROJECT_IDEAS_JSON))
    picks: List[Dict[str, Any]] = []
    if policy["recommend_mode"] == "depth_from_resume":
        # prioritize resume skillset (rule 1 & 4)
        base = have_skills[:10] or ["portfolio"]
        picks = _choose_projects(base, ideas, n_projects=max(3, n_projects))
    elif policy["recommend_mode"] == "targeted_to_gaps":
        # weighted gaps from jobs (fallback rule)
        missing_ranked = _rank_missing_skills(top_gap, weight_col="score", k=20)
        base = [sk for sk, _w in missing_ranked]
        if not base: base = have_skills[:10]  # still guarantee suggestions
        picks = _choose_projects(base, ideas, n_projects=max(3, n_projects))
    else:
        # "good_to_go" → still offer optional depth ideas (1–2)
        base = have_skills[:5] or ["portfolio"]
        picks = _choose_projects(base, ideas, n_projects=2)

    projects_df = pd.DataFrame(picks) if picks else pd.DataFrame(columns=["title","summary","target_skill","stack","deliverables"])
    cols = [c for c in ["title","summary","target_skill","stack","deliverables","references"] if c in projects_df.columns]
    projects_df = projects_df[cols] if cols else projects_df

    debug = {
        "policy": policy,
        "alpha_beta_gamma": (alpha, beta, gamma),
        "field": field,
        "resume_skills": have_skills,
        "n_resume_projects": len(projects),
        "top_jobs_preview": top_gap[["job_title","score","missing_skills","overlap_skills"]].head(5).to_dict(orient="records")
    }
    return projects_df, debug

# CLI demo
if __name__ == "__main__":
    demo = (
        "Skills: Python, SQL, Tableau, Docker, AWS\n"
        "Projects:\n"
        "- Customer Churn Predictor (XGBoost, Python, SQL)\n"
        "- Sales Dashboard (Tableau, Python; ETL in SQL)\n"
    )
    recs, dbg = recommend_projects(demo, field="Data", topn_jobs=15, n_projects=5, alpha=0.3, beta=0.3, gamma=0.4)
    print("\nRecommended Projects")
    print(recs.to_string(index=False) if not recs.empty else "No project ideas found.")
    print("\nAdvice:")
    print(" • " + "\n • ".join(dbg["policy"]["actions"]))
