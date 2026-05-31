"""
Resume X-Ray â Fintech Edition (v1)
Author: Neelima Verma | MS Data Science, Pace University

A diagnostic tool that audits resumes against Fintech data/analytics roles
and surfaces hidden rejection reasons â before you apply.

Built with: Groq (Llama 3.3 70B), Gradio, pdfplumber

Install:
    pip install gradio pdfplumber requests python-dotenv

Run:
    python resume_xray.py

Required env (in .env):
    GROQ_API_KEY=your_key_here   # optional; deterministic fallback if absent
"""

import csv
import os
import re
import threading
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import gradio as gr
import pdfplumber
import requests
from dotenv import load_dotenv

warnings.filterwarnings('ignore')


# ============================================================
# CONFIG
# ============================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class CandidateProfile:
    raw_text: str
    years_experience: int
    has_outcomes: bool
    has_ownership: bool


@dataclass
class DimensionScore:
    score: int  # 0-100
    evidence: List[str]
    gaps: List[str]


# ============================================================
# FINTECH ROLE LIBRARIES
# Real domain knowledge per role: what Fintech hiring managers
# actually scan for. Each role has core skills, differentiators,
# Fintech-specific KPIs, and common silent-rejection patterns.
# ============================================================
FINTECH_ROLES = {
    "Data Analyst": {
        "core_skills": ["sql", "excel", "python", "data analysis", "statistics",
                        "tableau", "power bi"],
        "differentiator_skills": ["dbt", "airflow", "looker", "a/b testing",
                                  "experimentation", "window functions", "snowflake"],
        "modern_tools": ["dbt", "airflow", "snowflake", "databricks", "looker",
                         "bigquery", "fivetran", "mode", "hex"],
        "fintech_kpis": ["default rate", "approval rate", "conversion rate", "ltv",
                         "cac", "churn", "revenue per user", "transaction volume"],
        "rejection_patterns": [
            {
                "trigger": "no_business_impact",
                "message": "Resume shows reporting work but no business decisions driven. Recruiters can't tell what changed because of your analysis."
            },
            {
                "trigger": "no_fintech_kpis",
                "message": "Dashboards mentioned but no Fintech-specific KPIs (default rate, LTV, conversion). Reads as generic BI, not domain expertise."
            },
        ],
    },
    "Data Scientist": {
        "core_skills": ["python", "sql", "machine learning", "statistics",
                        "pandas", "modeling", "scikit-learn"],
        "differentiator_skills": ["mlops", "feature engineering", "causal inference",
                                  "a/b testing", "deployed models", "credit risk",
                                  "fraud detection", "tensorflow", "pytorch"],
        "modern_tools": ["mlflow", "kubeflow", "sagemaker", "feature store",
                         "dbt", "airflow", "snowflake", "pytorch", "tensorflow",
                         "transformers", "huggingface", "vector database",
                         "langchain", "llama", "rag"],
        "fintech_kpis": ["auc", "default rate", "fraud catch rate", "false positive rate",
                         "model latency", "lift", "precision", "recall"],
        "rejection_patterns": [
            {
                "trigger": "no_production_signal",
                "message": "Models appear in notebooks but no signal of production deployment. Fintech wants ML that ships, not ML that runs once."
            },
            {
                "trigger": "no_fintech_kpis",
                "message": "ML experience is generic. No mention of credit, fraud, or AML use cases that define Fintech ML."
            },
        ],
    },
    "Product Analyst": {
        "core_skills": ["sql", "a/b testing", "analytics", "metrics", "excel",
                        "dashboard"],
        "differentiator_skills": ["amplitude", "mixpanel", "experimentation",
                                  "cohort analysis", "funnel analysis",
                                  "causal inference", "python"],
        "modern_tools": ["amplitude", "mixpanel", "looker", "mode", "hex", "dbt",
                         "snowflake", "statsig", "optimizely", "segment"],
        "fintech_kpis": ["activation rate", "conversion", "retention", "ltv",
                         "feature adoption", "transaction frequency", "churn"],
        "rejection_patterns": [
            {
                "trigger": "no_ab_testing",
                "message": "Fintech product teams run constant experiments. No A/B testing or experimentation signal on the resume."
            },
            {
                "trigger": "no_business_impact",
                "message": "Analytics shown but no product decisions changed because of it. Fintech product teams need analysts who drive launches, not just measure them."
            },
        ],
    },
    "Risk Analyst": {
        "core_skills": ["sql", "excel", "statistics", "risk", "regulatory",
                        "modeling"],
        "differentiator_skills": ["python", "sas", "credit scoring", "basel",
                                  "stress testing", "fair lending", "ecoa",
                                  "cfpb", "occ", "model risk"],
        "modern_tools": ["sas", "python", "r", "snowflake", "tableau",
                         "aif360", "fairlearn", "h2o", "datarobot", "shap"],
        "fintech_kpis": ["default rate", "charge-off rate", "loss given default",
                         "probability of default", "disparate impact"],
        "rejection_patterns": [
            {
                "trigger": "no_risk_terminology",
                "message": "Resume reads like general analytics, not risk. Missing regulatory terminology (ECOA, Basel, MRM, stress testing) that risk hiring managers scan for."
            },
            {
                "trigger": "no_business_impact",
                "message": "Risk roles want quantified loss reduction. Resume has no $-impact or rate-improvement metrics tied to risk decisions."
            },
        ],
    },
    "Financial Analyst": {
        "core_skills": ["excel", "financial modeling", "accounting", "sql"],
        "differentiator_skills": ["python", "anaplan", "hyperion", "tableau",
                                  "unit economics", "valuation", "fp&a"],
        "modern_tools": ["anaplan", "netsuite", "tableau", "power bi", "python",
                         "snowflake", "looker", "dbt", "workday"],
        "fintech_kpis": ["unit economics", "cac payback", "ltv", "arr", "mrr",
                         "burn rate", "gross margin", "payback period"],
        "rejection_patterns": [
            {
                "trigger": "no_fintech_unit_econ",
                "message": "FP&A experience present but no Fintech unit economics (CAC payback, LTV/CAC, burn). Fintech finance roles expect SaaS-style metric fluency."
            },
            {
                "trigger": "no_business_impact",
                "message": "Reporting cycle work shown but no decisions guided. Senior Fintech finance roles need analysts who shape capital allocation, not just close books."
            },
        ],
    },
    "Quantitative Analyst": {
        "core_skills": ["python", "statistics", "mathematics", "modeling",
                        "probability"],
        "differentiator_skills": ["c++", "kdb+", "time series", "derivatives",
                                  "stochastic", "backtesting", "optimization",
                                  "signal processing", "low latency"],
        "modern_tools": ["kdb+", "c++", "python", "numpy", "scipy",
                         "jax", "ray", "pytorch", "pandas"],
        "fintech_kpis": ["sharpe ratio", "alpha", "drawdown", "backtest",
                         "latency", "win rate"],
        "rejection_patterns": [
            {
                "trigger": "ml_without_quant_math",
                "message": "Resume shows ML/DS skills but no quantitative finance math (stochastic calculus, time series, derivatives). Quant roles want different math depth."
            },
            {
                "trigger": "no_backtest_signal",
                "message": "No mention of backtesting, alpha generation, or strategy validation. Quant teams hire for proven systematic thinking."
            },
        ],
    },
    "Business Analyst": {
        "core_skills": ["excel", "sql", "stakeholder", "requirements",
                        "analysis", "reporting"],
        "differentiator_skills": ["python", "tableau", "agile", "jira",
                                  "process mapping", "financial products"],
        "modern_tools": ["jira", "confluence", "tableau", "power bi", "sql",
                         "snowflake", "asana", "notion", "miro", "lucidchart"],
        "fintech_kpis": ["process efficiency", "time to market", "cost reduction",
                         "customer satisfaction", "operational metrics"],
        "rejection_patterns": [
            {
                "trigger": "no_data_depth",
                "message": "Resume reads as pure operations/PM, not analytical. Fintech BA roles want hybrid analyst-PM skills with SQL depth."
            },
            {
                "trigger": "no_fintech_product_context",
                "message": "BA experience is industry-agnostic. No Fintech product context (payments, lending, banking, trading) hiring managers scan for."
            },
        ],
    },
}


SENIORITY_EXPECTATIONS = {
    "Junior": {
        "min_years": 0,
        "max_years": 2,
        "expected_ownership_words": 1,
        "expected_leadership_words": 0,
        "expected_impact_metrics": 1,
        "description": "Junior IC: entry-level execution",
    },
    "Mid": {
        "min_years": 2,
        "max_years": 5,
        "expected_ownership_words": 3,
        "expected_leadership_words": 1,
        "expected_impact_metrics": 3,
        "description": "Mid IC: independent execution with measurable impact",
    },
    "Senior": {
        "min_years": 5,
        "max_years": 100,
        "expected_ownership_words": 5,
        "expected_leadership_words": 3,
        "expected_impact_metrics": 5,
        "description": "Senior IC: ownership, mentorship, strategic impact",
    },
}


# ============================================================
# SKILL SYNONYMS â word-boundary matched so "r" doesn't match
# inside words like "reporting" or "engineer"
# ============================================================
SKILL_SYNONYMS = {
    "sql": ["mysql", "postgresql", "postgres", "tsql", "plsql", "bigquery"],
    "python": ["pandas", "numpy", "scikit-learn"],
    "excel": ["microsoft excel", "ms excel", "vlookup", "pivot table"],
    "tableau": ["tableau desktop", "tableau server"],
    "power bi": ["powerbi", "power-bi", "microsoft power bi"],
    "data analysis": ["data analytics", "exploratory data analysis", "eda"],
    "machine learning": ["ml", "ml engineering", "ai/ml", "ai / ml"],
    "deep learning": ["neural networks", "neural net"],
    "statistics": ["statistical", "statistical analysis"],
    "a/b testing": ["ab testing", "experimentation", "experimental design"],
    "modeling": ["model", "models", "predictive modeling", "model evaluation"],
    "stakeholder": ["stakeholder management", "stakeholder engagement"],
    "risk": ["risk management", "risk analytics", "risk modeling"],
    "regulatory": ["compliance", "regulation"],
    "fp&a": ["financial planning and analysis", "fpa"],
    "ecoa": ["equal credit opportunity act"],
    "basel": ["basel iii", "basel ii"],
    "airflow": ["apache airflow"],
}


# ============================================================
# HELPERS
# ============================================================
def has_word(needle: str, haystack: str) -> bool:
    """Word-boundary match. 'r' won't match inside 'reporting' or 'engineer'.

    For multi-word skills, treats spaces and hyphens as interchangeable:
    'time series' matches 'time-series', 'timeseries', or 'time series'.
    """
    needle_lower = needle.lower()
    haystack_lower = haystack.lower()

    if " " not in needle_lower and "-" not in needle_lower:
        # Single word â straight word-boundary match
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(needle_lower) + r'(?![a-zA-Z0-9])'
        return bool(re.search(pattern, haystack_lower))

    # Multi-word: allow space/hyphen/empty between tokens
    tokens = re.split(r'[\s\-]+', needle_lower)
    flexible_middle = r'[\s\-]?'.join(re.escape(t) for t in tokens if t)
    pattern = r'(?<![a-zA-Z0-9])' + flexible_middle + r'(?![a-zA-Z0-9])'
    return bool(re.search(pattern, haystack_lower))


def skill_present(skill: str, text: str) -> bool:
    """Check skill OR its synonyms against text, word-boundary aware."""
    skill_lower = skill.lower()
    if has_word(skill_lower, text):
        return True
    for variant in SKILL_SYNONYMS.get(skill_lower, []):
        if has_word(variant, text):
            return True
    for canonical, variants in SKILL_SYNONYMS.items():
        if skill_lower in variants:
            if has_word(canonical, text):
                return True
    return False


# ============================================================
# RESUME PARSER
# ============================================================
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"PDF read error: {e}")
    return text


def extract_years_experience(text: str) -> int:
    """Estimate years of experience from resume text.

    Strategy:
    1. Try explicit 'X+ years ...' patterns first â most reliable signal.
       Catches: 'X years of experience', 'X+ years in [domain]',
                'over X years', 'X years building/leading/managing'.
    2. Fall back to summing year ranges. Handles both 'YYYY â YYYY'
       and 'Month YYYY â Month YYYY' formats.
    """
    text_lower = text.lower()

    explicit_patterns = [
        # 'X+ years (of) experience'
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?experience',
        # 'experience (of) X+ years'
        r'experience\s*(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)',
        # 'X+ years <verb/preposition>' â catches '10 years applying analytical methods',
        # '10+ years in regulated financial services', 'X years working in', etc.
        r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:in|across|with|of|at|leading|managing|building|'
        r'hands[\s-]on|spanning|developing|applying|delivering|supporting|working|driving|'
        r'doing|performing|executing|running|combining)',
        # 'Over/More than/Nearly X+ years'
        r'(?:over|more than|nearly|almost)\s+(\d+)\+?\s*(?:years?|yrs?)',
    ]
    for p in explicit_patterns:
        m = re.search(p, text_lower)
        if m:
            return int(m.group(1))

    # Year-range fallback. The optional '(?:[a-zA-Z]+\.?\s+)?' before the
    # end year handles 'Feb 2014 â Jul 2024' style (skips 'Jul ' to find 2024).
    year_range = r'(20\d{2}|19\d{2})\s*[-â]\s*(?:[a-zA-Z]+\.?\s+)?(20\d{2}|19\d{2}|present|current)'
    matches = re.findall(year_range, text_lower)
    if matches:
        total = 0
        current_year = datetime.now().year
        for start, end in matches:
            start_year = int(start)
            end_year = current_year if end in ['present', 'current'] else int(end)
            total += max(0, end_year - start_year)
        return min(total, 30)

    return 0


def parse_resume(pdf_path: str) -> CandidateProfile:
    text = extract_text_from_pdf(pdf_path)
    text_lower = text.lower()

    outcome_words = ["increased", "decreased", "drove", "saved", "generated",
                     "improved", "reduced", "delivered", "achieved"]
    has_outcomes = any(has_word(w, text_lower) for w in outcome_words)

    ownership_words = ["led", "owned", "built", "designed", "architected",
                       "managed", "directed", "established", "launched"]
    has_ownership = any(has_word(w, text_lower) for w in ownership_words)

    return CandidateProfile(
        raw_text=text,
        years_experience=extract_years_experience(text),
        has_outcomes=has_outcomes,
        has_ownership=has_ownership,
    )


# ============================================================
# SCORING ENGINE â 5 dimensions
# ============================================================
def score_skill_match(candidate: CandidateProfile, role: str) -> DimensionScore:
    role_def = FINTECH_ROLES[role]
    text = candidate.raw_text.lower()

    core = role_def["core_skills"]
    diff = role_def["differentiator_skills"]

    core_matched = [s for s in core if skill_present(s, text)]
    diff_matched = [s for s in diff if skill_present(s, text)]
    core_missing = [s for s in core if not skill_present(s, text)]

    core_pct = len(core_matched) / max(len(core), 1)
    diff_pct = len(diff_matched) / max(len(diff), 1)
    score = round((core_pct * 0.7 + diff_pct * 0.3) * 100)

    evidence = [f"Core skills present: {', '.join([s.title() for s in core_matched[:6]])}"] if core_matched else []
    if diff_matched:
        evidence.append(f"Differentiators: {', '.join([s.title() for s in diff_matched[:4]])}")

    gaps = []
    if core_missing:
        gaps.append(f"Missing core skills: {', '.join([s.title() for s in core_missing[:3]])}")
    if not diff_matched:
        gaps.append(f"No differentiator skills (e.g., {', '.join([s.title() for s in diff[:3]])})")

    return DimensionScore(score=score, evidence=evidence, gaps=gaps)


def score_seniority_signals(candidate: CandidateProfile, level: str) -> DimensionScore:
    text = candidate.raw_text.lower()
    exp = SENIORITY_EXPECTATIONS[level]

    leadership_words = [
        "led", "managed", "directed", "supervised", "mentored",
        "coached", "headed", "spearheaded", "oversaw", "guided",
        "championed", "trained",
    ]
    ownership_words = [
        "owned", "built", "designed", "architected", "established", "launched",
        "created", "developed", "engineered", "authored", "founded", "initiated",
        "drove", "delivered", "productionized", "pioneered", "implemented", "defined",
    ]
    strategic_words = [
        "strategy", "roadmap", "vision", "stakeholder", "executive",
        "cross-functional", "partnered", "aligned", "influenced", "advocated",
    ]

    leadership_found = [w for w in leadership_words if has_word(w, text)]
    ownership_found = [w for w in ownership_words if has_word(w, text)]
    strategic_found = [w for w in strategic_words if has_word(w, text)]

    leadership_target = max(exp["expected_leadership_words"], 1)
    ownership_target = max(exp["expected_ownership_words"], 1)

    leadership_ratio = min(len(leadership_found) / leadership_target, 1.0)
    ownership_ratio = min(len(ownership_found) / ownership_target, 1.0)
    strategic_ratio = min(len(strategic_found) / 3, 1.0)

    score = round((leadership_ratio * 0.4 + ownership_ratio * 0.5 + strategic_ratio * 0.1) * 100)

    evidence = []
    if leadership_found:
        evidence.append(f"Leadership words: {', '.join(leadership_found[:4])}")
    if ownership_found:
        evidence.append(f"Ownership words: {', '.join(ownership_found[:4])}")
    if strategic_found:
        evidence.append(f"Strategic words: {', '.join(strategic_found[:3])}")

    gaps = []
    if exp["expected_leadership_words"] > 0 and len(leadership_found) < exp["expected_leadership_words"]:
        gaps.append(f"{level} bar expects {exp['expected_leadership_words']}+ leadership words; resume has {len(leadership_found)}")
    if len(ownership_found) < exp["expected_ownership_words"]:
        gaps.append(f"{level} bar expects {exp['expected_ownership_words']}+ ownership words; resume has {len(ownership_found)}")

    return DimensionScore(score=score, evidence=evidence, gaps=gaps)


def score_business_impact(candidate: CandidateProfile, role: str, level: str) -> DimensionScore:
    text = candidate.raw_text.lower()
    role_def = FINTECH_ROLES[role]
    exp = SENIORITY_EXPECTATIONS[level]

    dollar_matches = re.findall(r'\$[\d,]+[kmb]?|\d+\s*(?:million|billion|thousand)', text)
    percent_matches = re.findall(r'\d+%|\d+\s*percent', text)
    scale_matches = [w for w in ["enterprise", "global", "millions", "thousands"] if has_word(w, text)]

    total_metrics = len(dollar_matches) + len(percent_matches) + len(scale_matches)

    fintech_kpis = role_def["fintech_kpis"]
    kpis_found = [k for k in fintech_kpis if has_word(k, text)]

    base_score = min(total_metrics / max(exp["expected_impact_metrics"], 1), 1.0) * 70
    kpi_bonus = min(len(kpis_found) * 10, 30)
    score = round(base_score + kpi_bonus)

    evidence = []
    if dollar_matches:
        evidence.append(f"$ metrics: {', '.join(dollar_matches[:3])}")
    if percent_matches:
        evidence.append(f"% metrics: {', '.join(percent_matches[:3])}")
    if kpis_found:
        evidence.append(f"Fintech KPIs detected: {', '.join(kpis_found[:3])}")

    gaps = []
    if total_metrics < exp["expected_impact_metrics"]:
        gaps.append(f"{level} bar expects {exp['expected_impact_metrics']}+ quantified metrics; resume has {total_metrics}")
    if not kpis_found:
        gaps.append(f"No Fintech-specific KPIs (e.g., {', '.join(fintech_kpis[:3])})")

    return DimensionScore(score=min(score, 100), evidence=evidence, gaps=gaps)


def score_keyword_narrative_balance(candidate: CandidateProfile) -> DimensionScore:
    text = candidate.raw_text
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    total_words = len(words)

    if total_words < 50:
        return DimensionScore(
            score=30,
            evidence=["Resume text too short to analyze"],
            gaps=["Resume may be image-heavy or too brief â extract text-based PDF"]
        )

    narrative_connectors = ["which", "because", "led to", "resulting", "by",
                            "through", "while", "after", "during", "across"]
    narrative_count = sum(1 for w in narrative_connectors if has_word(w, text.lower()))

    action_verbs = ["built", "designed", "led", "managed", "created", "developed",
                    "drove", "delivered", "implemented", "optimized", "analyzed",
                    "owned", "architected", "launched", "applied", "conducted",
                    "validated", "extended", "engineered", "recruited", "enforced",
                    "productionized", "surfaced", "quantified", "restored",
                    "executed", "partnered", "scaled", "transformed", "automated",
                    "streamlined", "generated", "presented", "established",
                    "deployed", "shipped", "automated", "trained", "tuned",
                    "evaluated", "investigated", "diagnosed", "modeled"]
    action_count = sum(1 for w in action_verbs if has_word(w, text.lower()))

    # Density per 100 words
    norm = max(total_words / 100, 1)
    action_density = action_count / norm
    narrative_density = narrative_count / norm

    if action_density >= 2 and narrative_density >= 1:
        score = 90
        evidence = [f"Good balance: {action_count} action verbs + {narrative_count} narrative connectors"]
        gaps = []
    elif action_density >= 2 and narrative_density < 1:
        score = 60
        evidence = [f"Strong action verbs ({action_count}) but light narrative ({narrative_count} connectors)"]
        gaps = ["Resume reads like a keyword list. Add 'which led to X', 'resulting in Y', 'by doing Z' to show causation."]
    elif action_density < 2 and narrative_density >= 1:
        score = 55
        evidence = [f"Narrative present but few action verbs ({action_count})"]
        gaps = ["Resume tells stories but lacks ownership verbs. Replace 'worked on' with 'built', 'led', 'drove', 'designed'."]
    else:
        score = 35
        evidence = [f"Sparse: {action_count} action verbs, {narrative_count} narrative connectors"]
        gaps = ["Resume reads like a job description, not a record of accomplishments."]

    return DimensionScore(score=score, evidence=evidence, gaps=gaps)


def score_modern_stack(candidate: CandidateProfile, role: str) -> DimensionScore:
    text = candidate.raw_text.lower()
    role_def = FINTECH_ROLES[role]
    modern = role_def["modern_tools"]

    found = [t for t in modern if skill_present(t, text)]
    score = round(min(len(found) / max(len(modern), 1), 1.0) * 100)

    evidence = [f"Modern tools: {', '.join([t.title() for t in found])}"] if found else []
    gaps = []
    if not found:
        gaps.append(f"No modern stack signals. For {role}, consider: {', '.join([t.title() for t in modern[:3]])}")
    elif len(found) < len(modern) / 2:
        missing = [t for t in modern if t not in found][:3]
        gaps.append(f"Partial modern stack. Add: {', '.join([t.title() for t in missing])}")

    return DimensionScore(score=score, evidence=evidence, gaps=gaps)


# ============================================================
# HIDDEN REJECTION DETECTOR â the killer feature
# ============================================================
def detect_hidden_rejections(
    candidate: CandidateProfile,
    role: str,
    level: str,
    scores: Dict[str, DimensionScore],
) -> List[str]:
    rejections = []
    role_def = FINTECH_ROLES[role]
    exp = SENIORITY_EXPECTATIONS[level]
    text = candidate.raw_text.lower()

    for pattern in role_def["rejection_patterns"]:
        trigger = pattern["trigger"]
        fired = False

        if trigger == "no_business_impact" and scores["impact"].score < 40:
            fired = True
        elif trigger == "no_fintech_kpis":
            kpis = [k for k in role_def["fintech_kpis"] if has_word(k, text)]
            if not kpis:
                fired = True
        elif trigger == "no_production_signal":
            prod_words = ["production", "deployed", "shipped", "live"]
            if not any(has_word(w, text) for w in prod_words):
                fired = True
        elif trigger == "no_ab_testing":
            if not skill_present("a/b testing", text):
                fired = True
        elif trigger == "no_risk_terminology":
            risk_terms = ["basel", "ecoa", "cfpb", "occ", "stress test", "credit risk"]
            if not any(has_word(t, text) for t in risk_terms):
                fired = True
        elif trigger == "no_fintech_unit_econ":
            terms = ["cac", "ltv", "unit economics", "payback", "burn"]
            if not any(has_word(t, text) for t in terms):
                fired = True
        elif trigger == "ml_without_quant_math":
            ml_present = skill_present("machine learning", text)
            quant_math = ["stochastic", "derivatives", "time series", "backtest"]
            if ml_present and not any(has_word(t, text) for t in quant_math):
                fired = True
        elif trigger == "no_backtest_signal":
            if not has_word("backtest", text) and not has_word("backtesting", text):
                fired = True
        elif trigger == "no_data_depth":
            if not skill_present("sql", text) and not skill_present("python", text):
                fired = True
        elif trigger == "no_fintech_product_context":
            products = ["payments", "lending", "banking", "trading", "credit"]
            if not any(has_word(p, text) for p in products):
                fired = True

        if fired:
            rejections.append(pattern["message"])

    # Universal rejections
    if scores["seniority"].score < 40 and level == "Senior":
        rejections.append(
            f"Senior role targeted but resume reads junior/mid-level. "
            f"Expected {exp['expected_ownership_words']}+ ownership words and "
            f"{exp['expected_leadership_words']}+ leadership words."
        )

    if candidate.years_experience < exp["min_years"]:
        rejections.append(
            f"Targeting {level} ({exp['min_years']}+ yrs required) but resume "
            f"shows only {candidate.years_experience} years."
        )

    return rejections


# ============================================================
# TOP FIXES â deterministic, ranked by lowest dimension scores
# ============================================================
def generate_top_fixes(
    scores: Dict[str, DimensionScore],
    role: str,
    level: str,
) -> List[Dict]:
    fixes = []
    role_def = FINTECH_ROLES[role]
    ranked = sorted(scores.items(), key=lambda x: x[1].score)

    for dim_name, dim_score in ranked[:3]:
        if dim_score.score >= 80:
            continue

        fix = {"dimension": dim_name, "score": dim_score.score}

        if dim_name == "skill_match":
            missing_gaps = [g for g in dim_score.gaps if "Missing core" in g]
            if missing_gaps:
                fix["action"] = f"Add missing core skills to resume. {missing_gaps[0]}"
            else:
                fix["action"] = f"Add differentiator skills like {', '.join(role_def['differentiator_skills'][:3])}."
        elif dim_name == "seniority":
            fix["action"] = (
                f"For {level} level, add ownership language: 'Owned X', 'Led Y', "
                f"'Designed Z'. Replace 'worked on' with verbs that show what you did."
            )
        elif dim_name == "impact":
            fix["action"] = (
                f"Add 3+ Fintech-specific metrics. "
                f"Use KPIs like {', '.join(role_def['fintech_kpis'][:3])}. "
                f"Format: 'Reduced [KPI] by X%' or 'Drove $Y in [outcome]'."
            )
        elif dim_name == "narrative":
            fix["action"] = (
                "Convert keyword bullets into mini-stories. "
                "Format: 'Did X using Y, which led to Z.' Causation is what recruiters scan for."
            )
        elif dim_name == "modern_stack":
            tools = role_def["modern_tools"][:2]
            fix["action"] = (
                f"Add modern tooling for {role}: {', '.join([t.title() for t in tools])}. "
                f"Path: tutorial â portfolio project â resume bullet."
            )

        if "action" in fix:
            fixes.append(fix)

    return fixes


# ============================================================
# VERDICT â single Groq call with deterministic fallback
# ============================================================
def call_groq(prompt: str, system: str = None, max_tokens: int = 200) -> str:
    if not GROQ_API_KEY:
        return ""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        r = requests.post(
            GROQ_URL,
            headers=headers,
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": max_tokens,
            },
            timeout=20,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            print(f"Groq error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"Groq error: {e}")
    return ""


def generate_verdict(
    role: str,
    level: str,
    location: str,
    ats: int,
    recruiter: int,
    readiness: int,
    rejections: List[str],
    top_fixes: List[Dict],
) -> str:
    rejection_summary = "; ".join(rejections[:2]) if rejections else "no major flags"
    top_fix = top_fixes[0]["action"] if top_fixes else "minor refinements only"

    system = (
        "You are a senior hiring manager at a Fintech company giving a candidate "
        "an honest 2-sentence diagnostic of their resume. Be direct, specific, "
        "and reference the role and level. No filler. No emojis. No bullet points."
    )
    prompt = (
        f"Resume targets: {level} {role} in Fintech, location {location}.\n"
        f"ATS sim: {ats}%. Recruiter sim: {recruiter}%. Overall readiness: {readiness}%.\n"
        f"Hidden rejection signals: {rejection_summary}.\n"
        f"Top fix needed: {top_fix}.\n\n"
        f"Write a 2-sentence verdict: first sentence = honest summary of where "
        f"this resume stands for {level} {role}. Second sentence = the single most "
        f"important thing to fix. No preamble. Start with the verdict."
    )

    verdict = call_groq(prompt, system, max_tokens=150).strip()

    if not verdict:
        if readiness >= 75:
            verdict = (
                f"Strong fit for {level} {role} in Fintech with {readiness}% overall readiness. "
                f"Focus: {top_fix}"
            )
        elif readiness >= 50:
            verdict = (
                f"Qualified for {level} {role} but with addressable gaps "
                f"({readiness}% readiness). Priority: {top_fix}"
            )
        else:
            verdict = (
                f"Resume currently reads below the {level} {role} bar in Fintech "
                f"({readiness}% readiness). Critical fix: {top_fix}"
            )

    return verdict


# ============================================================
# REPORT BUILDER
# ============================================================
def score_emoji(s: int) -> str:
    if s >= 75: return "ð¢"
    if s >= 50: return "ð¡"
    return "ð´"


def readiness_label(s: int) -> str:
    if s >= 75: return "STRONG"
    if s >= 55: return "COMPETITIVE"
    if s >= 35: return "DEVELOPING"
    return "BUILDING"


def build_report_markdown(
    candidate: CandidateProfile,
    role: str,
    level: str,
    location: str,
    ats: int,
    recruiter: int,
    readiness: int,
    scores: Dict[str, DimensionScore],
    rejections: List[str],
    top_fixes: List[Dict],
    verdict: str,
) -> str:
    # Seniority mismatch detection
    exp = SENIORITY_EXPECTATIONS[level]
    mismatch_banner = ""
    if candidate.years_experience > exp["max_years"] + 2:
        over_by = candidate.years_experience - exp["max_years"]
        suggested = "Senior" if level != "Senior" else "Senior"
        if level == "Junior":
            suggested = "Mid" if candidate.years_experience <= 5 else "Senior"
        elif level == "Mid":
            suggested = "Senior"
        mismatch_banner = (
            f"\n> â ï¸ **Heads up â possible level mismatch.** You selected **{level}** "
            f"(typical {exp['min_years']}â{exp['max_years']} yrs) but your resume "
            f"shows **{candidate.years_experience} years** of experience, "
            f"about {over_by} years above the typical {level} range. "
            f"For more accurate scoring, consider re-running with **{suggested}** level.\n"
        )
    elif candidate.years_experience < exp["min_years"]:
        short_by = exp["min_years"] - candidate.years_experience
        mismatch_banner = (
            f"\n> â ï¸ **Heads up â possible level mismatch.** You selected **{level}** "
            f"(typical {exp['min_years']}+ yrs) but your resume shows "
            f"**{candidate.years_experience} years**, {short_by} years below the typical bar. "
            f"Scoring assumes you're aiming higher than current experience suggests.\n"
        )

    md = f"""# ð©» Resume X-Ray Report

**Target:** {level} {role} in Fintech ({location})
**Resume Experience Detected:** {candidate.years_experience} years
{mismatch_banner}
---

## ð¯ The Verdict

> {verdict}

---

## ð Score Card

| Diagnostic | Score | Read |
|------------|-------|------|
| **ATS Simulation** | {score_emoji(ats)} **{ats}%** | {"Will pass automated filters" if ats >= 70 else "May get filtered out" if ats >= 50 else "Likely auto-rejected"} |
| **Recruiter Simulation** | {score_emoji(recruiter)} **{recruiter}%** | {"Strong shortlist candidate" if recruiter >= 70 else "Borderline shortlist" if recruiter >= 50 else "Unlikely to shortlist"} |
| **Overall Readiness** | {score_emoji(readiness)} **{readiness}%** | **{readiness_label(readiness)}** |

---

## ð Diagnostic Breakdown

### 1. Skill Match â {score_emoji(scores['skill_match'].score)} {scores['skill_match'].score}/100
"""
    for ev in scores["skill_match"].evidence:
        md += f"- {ev}\n"
    if scores["skill_match"].gaps:
        md += "\n**Gaps:**\n"
        for g in scores["skill_match"].gaps:
            md += f"- {g}\n"

    md += f"\n### 2. Seniority Signals ({level} bar) â {score_emoji(scores['seniority'].score)} {scores['seniority'].score}/100\n"
    for ev in scores["seniority"].evidence:
        md += f"- {ev}\n"
    if scores["seniority"].gaps:
        md += "\n**Gaps:**\n"
        for g in scores["seniority"].gaps:
            md += f"- {g}\n"

    md += f"\n### 3. Business Impact â {score_emoji(scores['impact'].score)} {scores['impact'].score}/100\n"
    for ev in scores["impact"].evidence:
        md += f"- {ev}\n"
    if scores["impact"].gaps:
        md += "\n**Gaps:**\n"
        for g in scores["impact"].gaps:
            md += f"- {g}\n"

    md += f"\n### 4. Keyword vs Narrative Balance â {score_emoji(scores['narrative'].score)} {scores['narrative'].score}/100\n"
    for ev in scores["narrative"].evidence:
        md += f"- {ev}\n"
    if scores["narrative"].gaps:
        md += "\n**Gaps:**\n"
        for g in scores["narrative"].gaps:
            md += f"- {g}\n"

    md += f"\n### 5. Modern Stack â {score_emoji(scores['modern_stack'].score)} {scores['modern_stack'].score}/100\n"
    for ev in scores["modern_stack"].evidence:
        md += f"- {ev}\n"
    if scores["modern_stack"].gaps:
        md += "\n**Gaps:**\n"
        for g in scores["modern_stack"].gaps:
            md += f"- {g}\n"

    md += "\n---\n\n## ð¨ Hidden Rejection Reasons\n\n"
    if rejections:
        md += "*The silent reasons recruiters pass on resumes â even when keywords match.*\n\n"
        for i, r in enumerate(rejections, 1):
            md += f"**{i}.** {r}\n\n"
    else:
        md += "â No major hidden rejection signals detected. Your resume should clear initial screens.\n"

    md += "\n---\n\n## ð  Top Fixes (Priority Order)\n\n"
    if top_fixes:
        for i, f in enumerate(top_fixes, 1):
            dim_label = f["dimension"].replace("_", " ").title()
            md += f"**{i}. {dim_label}** ({f['score']}/100)\n"
            md += f"   â {f['action']}\n\n"
    else:
        md += "Your resume is in strong shape. Focus on tailoring it to specific job descriptions before applying.\n"

    md += """
---

## â ï¸ Methodology

**What this X-Ray IS:** A rubric-based diagnostic. Word-boundary skill matching, role-specific seniority calibration, Fintech-KPI detection, and pattern-based rejection heuristics. One LLM call (Groq Llama 3.3 70B) generates the verdict.

**What it is NOT:** A guarantee of interview success. A comparison to other candidates. A substitute for human review.

*Built by Neelima Verma | MS Data Science, Pace University*
"""
    return md


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================
# ============================================================
# USAGE LOGGING â append a row to usage_log.csv each run AND
# (optionally) submit to a Google Form so you can see metrics
# in a Google Sheet.
# Logs metadata only (no resume content, no PII).
# Read locally with: open in Excel, or `wc -l usage_log.csv`.
# ============================================================
USAGE_LOG_PATH = Path(__file__).parent / "usage_log.csv"
USAGE_LOG_FIELDS = [
    "timestamp", "role", "level", "location",
    "years_experience", "readiness", "ats_score", "recruiter_score",
    "skill_match", "seniority", "impact", "narrative", "modern_stack",
]

# Google Form sink (set GOOGLE_FORM_LOGGING=false in env to disable)
GOOGLE_FORM_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLScU_ATkra-lGlPL1eYfXTd7f3Bl4jGDIr4ib_H93BEVa6hpOA/formResponse"
)
GOOGLE_FORM_FIELDS = {
    "timestamp":        "entry.1406310904",
    "role":             "entry.703901366",
    "level":            "entry.1319379145",
    "location":         "entry.1337686403",
    "years_experience": "entry.745550804",
    "readiness":        "entry.2011475978",
    "ats_score":        "entry.166697791",
    "recruiter_score":  "entry.1562382524",
    "skill_match":      "entry.1021699068",
    "seniority":        "entry.329651916",
    "impact":           "entry.1804577215",
    "narrative":        "entry.1660894031",
    "modern_stack":     "entry.517567882",
    "feedback":         "entry.485952234",
}
GOOGLE_FORM_ENABLED = os.getenv("GOOGLE_FORM_LOGGING", "true").lower() != "false"


def _post_to_google_form(row: dict) -> None:
    """Submit one row to the Google Form. Silent on failure."""
    try:
        payload = {
            GOOGLE_FORM_FIELDS[k]: str(v)
            for k, v in row.items() if k in GOOGLE_FORM_FIELDS
        }
        requests.post(GOOGLE_FORM_URL, data=payload, timeout=5)
    except Exception as e:
        print(f"[usage_log] form POST failed: {e}")


def _write_csv_row(row: dict) -> None:
    """Append row to local usage_log.csv. Silent on failure."""
    try:
        fields = USAGE_LOG_FIELDS + (["feedback"] if "feedback" not in USAGE_LOG_FIELDS else [])
        file_exists = USAGE_LOG_PATH.exists()
        with open(USAGE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[usage_log] CSV write failed: {e}")


def _post_to_form_async(row: dict) -> None:
    """Fire-and-forget Google Form POST so user doesn't wait on network."""
    if GOOGLE_FORM_ENABLED:
        threading.Thread(
            target=_post_to_google_form, args=(row,), daemon=True,
        ).start()


def run_xray(resume_file, role: str, level: str, location: str, previous_state=None):
    """Returns (report_markdown, run_data_dict_for_state).
    If a previous_state exists (from a previous run not yet flushed), post it
    to the form with empty feedback before starting the new run.
    """
    print(f"[run_xray] called. previous_state present: {previous_state is not None}")

    # Auto-flush previous unsaved run (so we don't lose it when the new run overwrites state)
    if previous_state:
        print(f"[run_xray] flushing previous state to Google Form")
        flush_row = dict(previous_state)
        flush_row.setdefault("feedback", "")
        _post_to_form_async(flush_row)

    if resume_file is None:
        return "â **Please upload your resume (PDF format).**", None

    if role not in FINTECH_ROLES:
        return f"â Role '{role}' not supported in v1. Available: {list(FINTECH_ROLES.keys())}", None

    if level not in SENIORITY_EXPECTATIONS:
        return "â Level must be Junior, Mid, or Senior.", None

    candidate = parse_resume(resume_file.name)
    if not candidate.raw_text:
        return "â Could not read resume. Make sure it's a text-based PDF (not a scanned image).", None

    scores = {
        "skill_match": score_skill_match(candidate, role),
        "seniority": score_seniority_signals(candidate, level),
        "impact": score_business_impact(candidate, role, level),
        "narrative": score_keyword_narrative_balance(candidate),
        "modern_stack": score_modern_stack(candidate, role),
    }

    # ATS sim: skill-focused
    ats_score = min(round(scores["skill_match"].score * 0.95), 95)

    # Recruiter sim: holistic
    recruiter_score = round(
        scores["skill_match"].score * 0.30 +
        scores["seniority"].score * 0.25 +
        scores["impact"].score * 0.25 +
        scores["narrative"].score * 0.10 +
        scores["modern_stack"].score * 0.10
    )
    recruiter_score = min(recruiter_score, 90)

    # Overall readiness
    readiness = round(
        scores["skill_match"].score * 0.25 +
        scores["seniority"].score * 0.25 +
        scores["impact"].score * 0.20 +
        scores["narrative"].score * 0.15 +
        scores["modern_stack"].score * 0.15
    )

    rejections = detect_hidden_rejections(candidate, role, level, scores)
    top_fixes = generate_top_fixes(scores, role, level)
    verdict = generate_verdict(
        role, level, location,
        ats_score, recruiter_score, readiness,
        rejections, top_fixes,
    )

    run_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "level": level,
        "location": location,
        "years_experience": candidate.years_experience,
        "readiness": readiness,
        "ats_score": ats_score,
        "recruiter_score": recruiter_score,
        "skill_match": scores["skill_match"].score,
        "seniority": scores["seniority"].score,
        "impact": scores["impact"].score,
        "narrative": scores["narrative"].score,
        "modern_stack": scores["modern_stack"].score,
        "feedback": "",
    }

    # Always write to local CSV (so nothing is lost even if user closes tab)
    _write_csv_row(run_data)

    print(f"[run_xray] returning state with scores: skill_match={run_data['skill_match']}, readiness={run_data['readiness']}")

    report = build_report_markdown(
        candidate, role, level, location,
        ats_score, recruiter_score, readiness,
        scores, rejections, top_fixes, verdict,
    )
    return report, run_data


def submit_feedback(feedback_text: str, role: str, level: str, location: str, pending_state):
    """Combine feedback with pending X-Ray data (if any) and submit one row.
    Returns (status_message, new_state) â new_state cleared after submission.
    """
    print(f"[submit_feedback] called. pending_state present: {pending_state is not None}")
    if pending_state:
        print(f"[submit_feedback] state has scores: skill_match={pending_state.get('skill_match')}, readiness={pending_state.get('readiness')}")

    feedback_text = (feedback_text or "").strip()
    if not feedback_text:
        return "â ï¸ Please write some feedback before sending.", pending_state

    if pending_state:
        # Combine: scores from the run + feedback text in one row
        row = dict(pending_state)
        row["feedback"] = feedback_text[:1500]
        print(f"[submit_feedback] combining feedback with state, posting full row")
    else:
        # Feedback without an X-Ray run â empty scores
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "role": role or "", "level": level or "", "location": location or "",
            "years_experience": "", "readiness": "", "ats_score": "",
            "recruiter_score": "", "skill_match": "", "seniority": "",
            "impact": "", "narrative": "", "modern_stack": "",
            "feedback": feedback_text[:1500],
        }
        print(f"[submit_feedback] no pending state â posting feedback-only row")

    _write_csv_row(row)
    _post_to_form_async(row)

    return "â Thanks â your feedback was received. It helps me make this tool better.", None




def clear_ui_only():
    """Clear visible UI when user changes inputs.
    Does NOT touch state - state is preserved until next run or feedback.
    """
    print(f"[clear_ui_only] user changed an input - clearing visible UI (state preserved)")
    return (
        "*Upload your resume and click Run X-Ray to get your diagnostic.*",
        "",   # run_status
        "",   # feedback_box
        "",   # feedback_status
    )


# ============================================================
# GRADIO UI
# ============================================================
def build_ui():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Resume X-Ray",
        css="""
        .run-btn { font-size: 1.1em !important; }
        """
    ) as demo:
        gr.Markdown("""
        # U0001fa7b Resume X-Ray
        ### Find out why your resume is getting rejected — before you apply.

        A diagnostic tool for **Fintech data and analytics roles**. Audits your resume against role + level + location and surfaces the hidden reasons recruiters silently pass on you.

        *5 diagnostic dimensions. One honest verdict. ~10 seconds.*
        """)

        pending_run_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                resume_file = gr.File(label="U0001f4c4 Upload Resume (PDF)", file_types=[".pdf"])
                role = gr.Dropdown(
                    label="U0001f3af Target Role",
                    choices=list(FINTECH_ROLES.keys()),
                    value="Data Analyst",
                )
                level = gr.Dropdown(
                    label="U0001f4c8 Target Seniority Level",
                    choices=list(SENIORITY_EXPECTATIONS.keys()),
                    value="Senior",
                )
                location = gr.Dropdown(
                    label="U0001f4cd Location",
                    choices=["Remote", "New York", "San Francisco", "Boston",
                             "Chicago", "Austin", "Charlotte", "Other"],
                    value="New York",
                )
                gr.Markdown("*Industry: **Fintech** (v1). Other industries coming soon.*")
                submit = gr.Button("U0001fa7b Run X-Ray", variant="primary", size="lg", elem_classes=["run-btn"])

            with gr.Column(scale=2):
                run_status = gr.Markdown(value="", elem_id="run-status")
                output = gr.Markdown(value="*Upload your resume and click Run X-Ray to get your diagnostic.*")

                gr.Markdown("---")
                gr.Markdown("### U0001f4ac Tell us what to improve")
                gr.Markdown(
                    "*Optional. Anything off? Score too harsh? Missing a role you'd want? "
                    "Type below and hit Send. Anonymous — no email captured.*"
                )
                feedback_box = gr.Textbox(
                    label="Your feedback",
                    placeholder="e.g. 'The seniority bar felt too strict' or 'Add Marketing Analyst role'",
                    lines=3,
                )
                send_feedback_btn = gr.Button("U0001f4e4 Send Feedback", variant="primary")
                feedback_status = gr.Markdown("")

        with gr.Accordion("⚙️ API Status", open=False):
            gr.Markdown(
                f"- **Groq API (LLM verdict):** "
                f"{'\u2705 Configured' if GROQ_API_KEY else '\u274c Not set \u2014 verdict uses deterministic fallback'}"
            )

        # -------------------------------------------------------
        # Run X-Ray: show loading first, then run analysis
        # -------------------------------------------------------
        def set_loading():
            return "⏳ **Analyzing your resume\u2026 this takes ~10 seconds. Please wait.**", ""

        submit.click(
            fn=set_loading,
            inputs=[],
            outputs=[run_status, output],
            show_progress="hidden",
        ).then(
            fn=run_xray,
            inputs=[resume_file, role, level, location, pending_run_state],
            outputs=[output, pending_run_state],
            show_progress="hidden",
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[run_status],
            show_progress="hidden",
        )

        # Send Feedback
        send_feedback_btn.click(
            fn=submit_feedback,
            inputs=[feedback_box, role, level, location, pending_run_state],
            outputs=[feedback_status, pending_run_state],
            show_progress="hidden",
        )

        # When inputs change - clear UI but PRESERVE state
        resume_file.upload(
            fn=clear_ui_only,
            inputs=[],
            outputs=[output, run_status, feedback_box, feedback_status],
        )

        for dd in [role, level, location]:
            dd.input(
                fn=clear_ui_only,
                inputs=[],
                outputs=[output, run_status, feedback_box, feedback_status],
            )

    return demo


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("U0001fa7b RESUME X-RAY v1 — Fintech Edition")
    print("=" * 60)
    print(f"Roles: {len(FINTECH_ROLES)} | Levels: {len(SENIORITY_EXPECTATIONS)}")
    print(f"Groq API: {'✅' if GROQ_API_KEY else '❌'}")
    print("=" * 60)
    demo = build_ui()
    demo.launch()
