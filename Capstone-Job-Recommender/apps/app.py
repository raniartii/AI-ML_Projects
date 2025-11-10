# apps/app.py
from __future__ import annotations

import re
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# --- repo bootstrap so "streamlit run apps/app.py" works from repo root ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# our modules
from src.step7_gap_projects import (
    recommend_projects,
    rank_jobs_hybrid,
    extract_skills as parse_skills,
    extract_projects as parse_projects,
    _estimate_project_count,  # show parsed vs estimated projects in sidebar card
)
from src.matcher import load_jobs_assets

# --- cached vocab for display filtering ---
@st.cache_resource
def _get_vocab_only() -> list[str]:
    _, vocab, _ = load_jobs_assets()
    return vocab

# --- section-aware, noise-proof skill display (for the right panel only) ---
_SKILLS_HEAD = re.compile(r"^\s*(skills|technical skills|skills & tools|tech stack|skills summary|core competencies)\s*:?\s*$", re.I)
_STOP_HEAD   = re.compile(r"^\s*(education|experience|work experience|projects|publications|certifications|achievements|summary)\s*:?\s*$", re.I)
_EMAIL = re.compile(r"\b\S+@\S+\b")
_URL   = re.compile(r"https?://\S+|www\.\S+")
_PHONE = re.compile(r"\b(?:\+?\d[\d\-\s]{7,})\b")

#Helper for detecting skills section and cleaning noise
def _detect_skills_for_display(text: str, vocab: list[str]) -> list[str]:
    """
    Extract tokens *primarily* from a 'Skills' section, strip contacts/URLs/phones,
    then keep only tokens present in your skills vocab to avoid noise.
    """
    if not text:
        return []
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if _SKILLS_HEAD.match(ln.strip()):
            start = i
            break
    if start is not None:
        buf = []
        for ln in lines[start+1:]:
            if _STOP_HEAD.match(ln.strip()):
                break
            buf.append(ln)
        block = "\n".join(buf)
    else:
        # fallback to whole text if no explicit Skills heading
        block = text

    # remove obvious noise
    block = _EMAIL.sub(" ", block)
    block = _URL.sub(" ", block)
    block = _PHONE.sub(" ", block)

    # reuse your extractor but filter to vocab
    raw = parse_skills(block)  # already normalizes tokens
    vset = set(vocab)
    seen, keep = set(), []
    for t in raw:
        t = t.strip().lower()
        if t in vset and t not in seen:
            keep.append(t); seen.add(t)
    return keep


# 1) Imports: try pypdf first, then PyPDF2
# optional PDF reader (support both packages)
try:
    from pypdf import PdfReader as _PdfReader   # modern package
    _PDF_LIB = "pypdf"
except Exception:
    try:
        from PyPDF2 import PdfReader as _PdfReader  # legacy name you have
        _PDF_LIB = "PyPDF2"
    except Exception:
        _PdfReader = None
        _PDF_LIB = None

# 2) Helper: use whichever reader is available
def read_pdf_bytes(file_bytes: bytes) -> str:
    if _PdfReader is None:
        return ""
    try:
        reader = _PdfReader(io.BytesIO(file_bytes))
        parts = []
        pages = getattr(reader, "pages", [])
        for page in pages:
            text = page.extract_text() or ""
            parts.append(text)
        return "\n".join(parts).strip()
    except Exception:
        return ""

# ---------- small helpers ----------
ASSETS = ROOT / "assets"

def _img(name: str) -> str | None:
    p = ASSETS / name
    return str(p) if p.exists() else None

def _show_image_if_exists(names: list[str]) -> bool:
    """Try to display the first existing image from names; return True if shown."""
    for n in names:
        path = _img(n)
        if path:
            try:
                st.image(path, use_container_width=True)
                return True
            except Exception:
                # try next candidate
                pass
    return False

def read_pdf_bytes(file_bytes: bytes) -> str:
    # _PdfReader must be defined in your imports:
    # try:
    #     from pypdf import PdfReader as _PdfReader
    #     _PDF_LIB = "pypdf"
    # except Exception:
    #     try:
    #         from PyPDF2 import PdfReader as _PdfReader
    #         _PDF_LIB = "PyPDF2"
    #     except Exception:
    #         _PdfReader = None
    #         _PDF_LIB = None
    if _PdfReader is None:
        return ""
    try:
        reader = _PdfReader(io.BytesIO(file_bytes))
        parts = []
        for page in getattr(reader, "pages", []):
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts).strip()
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def get_field_options() -> list[str]:
    try:
        df_jobs, _, _ = load_jobs_assets()
        if "field" in df_jobs.columns:
            vals = (
                df_jobs["field"].astype(str)
                .replace("", pd.NA).dropna().unique().tolist()
            )
            vals = sorted({v.strip() for v in vals if v.strip()})
            return ["(all)"] + vals
    except Exception:
        pass
    return ["(all)"]

def sample_resume() -> str:
    return (
        "Skills: Python, SQL, Tableau, Docker, AWS\n"
        "Experience: Built ETL pipelines and dashboards for sales analytics.\n"
        "Projects:\n"
        "- Customer Churn Predictor (XGBoost, Python, SQL)\n"
        "- Sales Dashboard (Tableau, Python; ETL in SQL)\n"
    )

# ---------- page setup ----------
st.set_page_config(page_title="Project&Job Recommender", layout="wide")

left, right = st.columns([1, 3], vertical_alignment="center")
with left:
    # SAFE: only show if a file exists; no more None to st.image()
    shown = _show_image_if_exists(["hero.png", "upload.png", "logo.png"])
with right:
    st.markdown("<h1>Project&Job Recommender</h1>", unsafe_allow_html=True)
    st.caption("Upload your resume as PDF or paste text. We’ll find matching jobs and recommend projects to strengthen your profile.")

# ---------- sidebar (plain-English) ----------
with st.sidebar:
    st.header("Options")
    field_choice = st.selectbox("Target field", get_field_options(), index=0, help="Filter results to a field if you like.")
    topn = st.slider("How many jobs to show", 5, 25, 15, 1)

    st.markdown("**How we score (A/B/C):**")
    st.caption("A: skills overlap • B: TF-IDF over job skills • C: TF-IDF over job text")
    alpha = st.slider("Weight A", 0.0, 1.0, 0.3, 0.05)
    beta  = st.slider("Weight B", 0.0, 1.0, 0.3, 0.05)
    gamma = st.slider("Weight C", 0.0, 1.0, 0.4, 0.05)
    s = max(alpha + beta + gamma, 1e-6)
    alpha, beta, gamma = alpha / s, beta / s, gamma / s

    use_sample = st.toggle("Use sample resume", value=False)
    run = st.button("Run recommender ▶️", use_container_width=True)

# ---------- STEP 1: Upload / Paste ----------
st.markdown("## 1) Upload your resume (PDF) **or** paste text")

col_up, col_det = st.columns([2, 1])

with col_up:
    uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"], accept_multiple_files=False)
    resume_text = ""

    if uploaded is not None:
        # 3) UI upload section: update the warning
        if uploaded.type == "application/pdf":
            if _PdfReader is None:
                st.warning("PDF support needs **pypdf** or **PyPDF2**. Add one to requirements.txt, then restart.")
            _data = uploaded.read()
            resume_text = read_pdf_bytes(_data)
            if not resume_text:
                st.warning("We couldn’t extract text from this PDF. It might be scanned (image-only). "
                   "Please paste your resume text below or export a text-based PDF.")


    default_text = sample_resume() if use_sample else ""
    resume_text = st.text_area(
        "Or paste resume text below",
        value=(resume_text or default_text),
        height=220,
        placeholder="Paste your resume text here…",
    )

with col_det:
    st.markdown("#### What we detected")
    if resume_text:
        vocab = _get_vocab_only()
        sk_det = _detect_skills_for_display(resume_text, vocab)[:30]
        st.write("**Skills:**", ", ".join(sk_det) if sk_det else "(none)")
        projs = parse_projects(resume_text)
        est = _estimate_project_count(resume_text)
        st.write(f"**Projects found:** {len(projs)} (est: {est})")
    else:
        st.write("**Skills:** (none)")
        st.write("**Projects found:** 0")

st.divider()

# ---------- STEP 2 & 3: Jobs + Projects ----------
if run:
    if not resume_text.strip():
        st.error("Please upload a PDF or paste your resume text.")
        st.stop()

    field = None if field_choice == "(all)" else field_choice

    tab_jobs, tab_projects, tab_explain = st.tabs(["💼 Job matches", "🧱 Project advice", "ℹ️ How this works"])

    with tab_jobs:
        st.markdown("## 2) Jobs you match today")
        with st.spinner("Scoring jobs…"):
            top_jobs, _ctx = rank_jobs_hybrid(
                resume_text, field=field, alpha=alpha, beta=beta, gamma=gamma, topn=topn
            )
        nice_cols = [c for c in ["job_title", "field", "score", "score_A", "score_B", "score_C",
                                 "overlap_skills", "missing_skills"] if c in top_jobs.columns]
        st.dataframe(top_jobs[nice_cols], use_container_width=True)
        st.caption("Tip: ‘Overlap skills’ = what you have. ‘Missing skills’ = what to learn next for that job.")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download job matches (CSV)",
                data=top_jobs.to_csv(index=False).encode("utf-8"),
                file_name="top_jobs.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_projects:
        st.markdown("## 3) Projects to add (or tidy up)")
        with st.spinner("Generating project recommendations…"):
            projects_df, dbg = recommend_projects(
                resume_text, field=field, topn_jobs=topn, n_projects=5, alpha=alpha, beta=beta, gamma=gamma
            )

        # Advice, plain-English (now adaptive thanks to backend policy fix)
        st.markdown("### Action plan")
        actions = dbg.get("policy", {}).get("actions", [])
        if actions:
            for a in actions:
                st.markdown(f"- {a}")
        else:
            st.info("No changes needed at the moment.")

        # Projects table
        st.markdown("### Recommended projects")
        if projects_df.empty:
            st.info("No targeted ideas found. Add entries to `datasets/prepared/project_ideas.json` to populate this list.")
        else:
            show_cols = [c for c in ["title","summary","target_skill","stack","deliverables","references"] if c in projects_df.columns]
            st.dataframe(projects_df[show_cols], use_container_width=True)

        st.download_button(
            "Download recommended projects (CSV)",
            data=projects_df.to_csv(index=False).encode("utf-8"),
            file_name="recommended_projects.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab_explain:
        st.markdown("## How this works (simple)")
        st.markdown("""
- **Step 1:** We read skills and projects from your resume.
- **Step 2:** We score jobs three ways:
  - **A**: direct skill overlap,
  - **B**: TF-IDF over job skills,
  - **C**: TF-IDF over job text (title + description).
- **Step 3:** We show skills you already have vs skills to learn.
- **Step 4:** We suggest project ideas that fit your skills or cover gaps.
""")
        st.caption("Weights let you emphasize certain signals. Defaults: A=0.3, B=0.3, C=0.4")
else:
    st.info("👆 Upload a PDF or paste text, then click **Run recommender ▶️**")

st.caption("Developed by Arti Rani")
