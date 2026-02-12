# Career-Intelligence-Bot
# 🚀 Career Intelligence Bot

**An AI-Powered Platform for Job Search Optimization Using LLMs, RAG, and Agentic Design**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **CS667 - Practical Data Science | Fall 2025 | Pace University**

📄 [Research Paper- In description] | 🎥 [Video Demo](https://youtu.be/YOUR_VIDEO_ID) | 🌐 [Live Demo](#quick-start)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [AI Topics Demonstrated](#ai-topics-demonstrated)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [API Keys Setup](#api-keys-setup)
- [Evaluation Results](#evaluation-results)
- [Future Work](#future-work)
- [Author](#author)
- [License](#license)

---

## 🎯 Overview

Job searching is broken. Candidates submit 100+ applications with ~5% callback rates, facing an opaque process where:
- **75% of resumes** are filtered by ATS before human review
- Candidates don't know **why** they're rejected
- Generic advice ("learn Python") doesn't convert to interviews

**Career Intelligence Bot** solves this with AI-powered analysis that tells you:
- ✅ Your **exact ATS score** for each job
- ✅ **Why recruiters might reject you** (simulated screening)
- ✅ **Which skills to add** (weighted by importance)
- ✅ A **90-day execution plan** designed for interview conversion

---

## ✨ Key Features

### 1. 📊 Dual-Score ATS Analysis
Not just keywords — we simulate both automated filters AND human recruiters.

| Score Type | What It Measures |
|------------|------------------|
| **ATS Pass Probability** | Keyword/skill match (will you pass the filter?) |
| **Recruiter Shortlist** | Seniority signals, impact, experience (will humans pick you?) |

### 2. 🎯 Job-Specific Skill Matching
Unlike tools using static templates, we parse **each job description** to show different requirements per job.

### 3. 🧠 Seniority Signal Detection
For senior roles, we detect leadership language:
- Leadership: "led", "managed", "mentored"
- Ownership: "owned", "built", "architected"
- Strategic: "roadmap", "vision", "executive"

### 4. 📈 Business Impact Quantification
We scan your resume for metrics ($, %, scale) and tell you if you're under-selling yourself.

### 5. 🗺️ 90-Day Execution Plan
**Not a learning plan — an interview conversion engine.**
- Phase 1: Resume Signal Upgrade (Weeks 1-2)
- Phase 2: Market-Aligned Portfolio (Weeks 3-6)
- Phase 3: Interview Conversion (Weeks 7-12)

### 6. 🏢 Multi-Industry Support
| Industry | Companies Tracked | Roles |
|----------|-------------------|-------|
| FAANG / Big Tech | 35 | 10 |
| Fintech | 37 | 11 |
| Healthcare | 24 | 8 |
| Consulting | 17 | 7 |
| E-Commerce | 15 | 8 |
| AI / ML | 21 | 8 |
| Startups | 14 | 8 |

---

## 🤖 AI Topics Demonstrated

| # | AI Topic | Implementation |
|---|----------|----------------|
| 1 | **Large Language Models (LLMs)** | Groq API (Llama 3.3 70B) for text generation |
| 2 | **RAG (Retrieval-Augmented Generation)** | Real-time market data as context |
| 3 | **Agentic AI** | Perceive → Analyze → Recommend loop |
| 4 | **Prompt Engineering** | Role-specific, few-shot prompts |
| 5 | **NLP / Information Extraction** | Resume parsing, skill detection |
| 6 | **Domain-Specific AI** | ATS simulation, career coaching |
| 7 | **Responsible AI** | Bias detection, transparency disclaimers |
| 8 | **Explainable AI (XAI)** | Transparent scoring rubrics |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input                                │
│              (Resume PDF + Preferences)                      │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Resume Parser                               │
│         (pdfplumber → CandidateProfile)                     │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Market Data Fetcher                             │
│         (NewsAPI + Adzuna Jobs API)                         │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  RAG Engine                                  │
│    (Retrieve context → Augment prompts → Generate)          │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 Career Agent                                 │
│      (Agentic reasoning: Perceive → Analyze → Act)          │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               Content Generator                              │
│   (ATS Scores, Cover Letters, Interview Prep, Roadmaps)     │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│             Responsible AI Guard                             │
│      (Bias detection, Disclaimers, Methodology notes)       │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 Gradio Interface                             │
│              (8 tabs, interactive UI)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/career-intelligence-bot.git
cd career-intelligence-bot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys
```bash
# Create .env file or export directly
export GROQ_API_KEY="your_groq_key"        # FREE at console.groq.com
export NEWS_API_KEY="your_newsapi_key"     # FREE at newsapi.org
export ADZUNA_APP_ID="your_adzuna_id"      # FREE at developer.adzuna.com
export ADZUNA_APP_KEY="your_adzuna_key"
```

### 5. Run the App
```bash
python fintech_ai_bot_v2.py
```

### 6. Open in Browser
```
http://localhost:7860
```

---

## 📖 Usage

### Step 1: Upload Your Resume
- Supported format: PDF
- The system extracts skills, experience, and achievements

### Step 2: Select Your Target
- **Industry**: FAANG, Fintech, Healthcare, etc.
- **Role**: Data Analyst, Data Scientist, Software Engineer, etc.
- **Location**: Any Location, New York, San Francisco, etc.

### Step 3: Click "Analyze with AI"
The system will:
1. Parse your resume
2. Fetch real-time job listings
3. Analyze each job against your profile
4. Generate personalized recommendations

### Step 4: Explore the 8 Tabs
| Tab | What You Get |
|-----|--------------|
| 📄 Resume Analysis | Role fit scores, salary estimate, recruiter view |
| 📈 Market Intelligence | Industry news, company momentum signals |
| 🎯 AI Job Matches | Jobs ranked by ATS score with skill gaps |
| 📊 Resume Optimizer | Dual-score analysis, rejection reasons |
| 📊 Role Readiness | Transparent rubric-based assessment |
| 🚀 90-Day Plan | Outcome-focused execution roadmap |
| ✉️ Cover Letters | AI-generated, job-tailored letters |
| 🎤 Interview Prep | Company-specific Q&A preparation |

---

## 📸 Screenshots

### Resume Analysis
```
Role Readiness Score: 72/100

| Dimension         | Weight | Your Score |
|-------------------|--------|------------|
| Required Skills   | 30%    | 83/100     |
| Experience Level  | 25%    | 100/100    |
| Seniority Signals | 20%    | 30/100     |
| Business Impact   | 15%    | 40/100     |
| Modern Tools      | 10%    | 25/100     |
```

### Dual-Score ATS Analysis
```
| Score Type              | Result | Meaning                     |
|-------------------------|--------|------------------------------|
| ATS Pass Probability    | 78%    | ✅ Will pass automated filters |
| Recruiter Shortlist     | 52%    | ⚠️ May lose to stronger cand. |

⚠️ Gap Alert: You'll pass ATS but may struggle during human review.
Focus on seniority signals and impact metrics.
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Groq API (Llama 3.3 70B) - FREE |
| **Frontend** | Gradio 4.0+ |
| **PDF Parsing** | pdfplumber |
| **News Data** | NewsAPI.org |
| **Job Data** | Adzuna API |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
career-intelligence-bot/
├── fintech_ai_bot_v2.py      # Main application (3400+ lines)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── cs667_2025fall_neelima_verma.pdf   # Research paper
├── cs667_2025fall_neelima_verma.tex   # LaTeX source
├── .env.example               # API keys template
└── screenshots/               # Demo images
    ├── resume_analysis.png
    ├── job_matches.png
    └── roadmap.png
```

---

## 🔑 API Keys Setup

All APIs used have **FREE tiers**:

| API | Free Tier | Get Key |
|-----|-----------|---------|
| **Groq** | 30 req/min | [console.groq.com](https://console.groq.com) |
| **NewsAPI** | 100 req/day | [newsapi.org](https://newsapi.org) |
| **Adzuna** | 250 req/day | [developer.adzuna.com](https://developer.adzuna.com) |

Create a `.env` file:
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
NEWS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
ADZUNA_APP_ID=xxxxxxxx
ADZUNA_APP_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 📊 Evaluation Results

### Skill Detection Accuracy
| Metric | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Technical Skills | 0.92 | 0.85 | 0.88 |
| Soft Skills | 0.78 | 0.72 | 0.75 |
| **Overall** | **0.87** | **0.81** | **0.84** |

### User Feedback (n=5 graduate students)
- ✅ Dual-score breakdown was "much more actionable"
- ✅ Seniority signal detection helped identify language gaps
- ✅ 90-day roadmap preferred over generic course recommendations

---

## 🔮 Future Work

1. **Empirical Calibration**: Validate recruiter score weights with actual interview outcome data
2. **Real ATS Integration**: Partner with ATS vendors for ground-truth validation
3. **Company Intelligence**: RAG over Glassdoor reviews for interview prep
4. **Resume Auto-Rewrite**: Generate optimized resume versions automatically
5. **Application Tracking**: Built-in tracker with outcome analytics

---

## 👤 Author

**Neelima Verma**  
MS Data Science, Pace University  
Seidenberg School of CSIS

- 📧 Email:vneelima44@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/neelima-verma-data-science/
- 🐙 GitHub: https://github.com/vneelima44

---



---

## 🙏 Acknowledgments

- **Professor** - Professor Yiqiao Yin - CS667 Practical Data Science, Pace University
- **Groq** - Free LLM API access
- **Gradio** - Easy ML web interfaces
- **Anthropic Claude** - Development assistance

---

<p align="center">
  <b>⭐ Star this repo if you found it helpful!</b>
</p>
