"""
🚀 Fintech Career Intelligence Bot v2.0 - AI-Powered Edition
Author: Neelima Verma | MS Data Science, Pace University

KEY AI FEATURES:
- LLM Integration (OpenAI/Claude) for intelligent career coaching
- RAG Architecture for grounding responses in real-time market data
- Generative AI for personalized cover letters & interview prep
- Agentic Reasoning for autonomous opportunity evaluation

APIs Used:
- OpenAI API (GPT-4) - For LLM intelligence
- NewsAPI - Real-time market intelligence
- Adzuna - Live job listings

TOPICS COVERED:
✅ Large Language Models (LLMs)
✅ Generative AI Applications
✅ Retrieval-Augmented Generation (RAG)
✅ Agentic Design & Reasoning
✅ Prompt Engineering
✅ Domain-specific AI (Finance/Fintech)
✅ Responsible AI (bias mitigation, transparency)
✅ AI for Good (career accessibility)
"""

# ============================================
# 1. IMPORTS
# ============================================
import gradio as gr
import pandas as pd
import pdfplumber
import requests
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


# ============================================
# 2. API CONFIGURATION
# ============================================
# Get your API keys:
# Groq: https://console.groq.com/keys (FREE!)
# NewsAPI: https://newsapi.org/register
# Adzuna: https://developer.adzuna.com/signup

GROQ_API_KEY = ""    # FREE! Get from console.groq.com/keys
NEWS_API_KEY = ""    # Your NewsAPI key
ADZUNA_APP_ID = ""   # Your Adzuna App ID
ADZUNA_APP_KEY = ""  # Your Adzuna App Key


# ============================================
# 3. DATA CLASSES & TYPES
# ============================================
@dataclass
class CandidateProfile:
    """Structured representation of a candidate"""
    raw_text: str
    skills: Dict[str, List[str]]
    years_experience: int
    education: List[str]
    previous_companies: List[str]
    career_trajectory: str  # LLM-generated insight


@dataclass
class JobOpportunity:
    """Structured job opportunity with AI enrichment"""
    title: str
    company: str
    location: str
    salary_range: str
    description: str
    requirements: List[str]
    url: str
    posted: str
    # AI-enriched fields
    company_momentum: int
    growth_signals: List[str]
    culture_indicators: List[str]
    ai_fit_analysis: str  # LLM-generated


@dataclass
class MarketIntelligence:
    """Real-time market context for RAG"""
    company_signals: Dict
    recent_news: List[Dict]
    funding_events: List[Dict]
    hiring_trends: Dict
    timestamp: datetime


# ============================================
# 4. LLM CLIENT - Core AI Integration (GROQ - FREE!)
# ============================================
class LLMClient:
    """
    Unified LLM client using Groq (FREE tier available).
    Groq provides fast inference for open-source models.
    
    TOPIC: Large Language Models (LLMs)
    """
    
    def __init__(self, provider: str = "groq"):
        self.provider = provider
        self.api_key = GROQ_API_KEY
        # Using Llama 3.3 70B - latest and FREE on Groq
        self.model = "llama-3.3-70b-versatile"
        
    def generate(self, 
                 prompt: str, 
                 system_prompt: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1000) -> str:
        """
        Generate text using Groq's LLM API (FREE).
        
        TOPIC: Prompt Engineering
        - System prompts establish AI persona and constraints
        - Temperature controls creativity vs consistency
        - Token limits manage response length
        """
        
        if not self.api_key:
            return self._fallback_response(prompt)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"LLM API Error: {response.status_code} - {response.text[:200]}")
                return self._fallback_response(prompt)
                
        except Exception as e:
            print(f"LLM Error: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Fallback when API unavailable - demonstrates graceful degradation"""
        return "[AI Analysis Unavailable - Add GROQ_API_KEY for full features. Get FREE key at console.groq.com/keys]"


# ============================================
# 5. RAG ENGINE - Retrieval-Augmented Generation
# ============================================
class RAGEngine:
    """
    Retrieval-Augmented Generation for grounding LLM responses
    in real-time market data.
    
    TOPIC: RAG Algorithms
    
    Architecture:
    1. RETRIEVE: Fetch relevant context from news/jobs APIs
    2. AUGMENT: Combine context with user query
    3. GENERATE: LLM produces grounded response
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.context_window = []  # Stores retrieved context
        self.max_context_tokens = 4000
        
    def retrieve_context(self, 
                        query: str,
                        market_intel: MarketIntelligence,
                        candidate: CandidateProfile) -> str:
        """
        RETRIEVE phase: Build relevant context from multiple sources
        """
        context_parts = []
        
        # 1. Market Intelligence Context
        if market_intel and market_intel.company_signals:
            try:
                top_companies = sorted(
                    market_intel.company_signals.items(),
                    key=lambda x: x[1].get('momentum_score', 0) if isinstance(x[1], dict) else 0,
                    reverse=True
                )[:5]
                
                market_context = "CURRENT MARKET INTELLIGENCE:\n"
                for company, data in top_companies:
                    if isinstance(data, dict):
                        signals = data.get('signals', [])
                        signals_str = ', '.join(str(s) for s in signals) if isinstance(signals, list) else ''
                        market_context += f"- {company}: Momentum {data.get('momentum_score', 0)}/100, "
                        market_context += f"Signals: {signals_str}\n"
                context_parts.append(market_context)
            except Exception as e:
                print(f"Error building market context: {e}")
        
        # 2. Recent News Context
        if market_intel and market_intel.recent_news:
            try:
                news_context = "RECENT FINTECH NEWS:\n"
                for news in market_intel.recent_news[:5]:
                    if isinstance(news, dict):
                        title = str(news.get('title', '') or '')[:100]
                        news_context += f"- {title}\n"
                context_parts.append(news_context)
            except Exception as e:
                print(f"Error building news context: {e}")
        
        # 3. Candidate Context
        try:
            tech_skills = candidate.skills.get('technical', []) if isinstance(candidate.skills, dict) else []
            analytics_skills = candidate.skills.get('analytics', []) if isinstance(candidate.skills, dict) else []
            business_skills = candidate.skills.get('business', []) if isinstance(candidate.skills, dict) else []
            finance_skills = candidate.skills.get('finance', []) if isinstance(candidate.skills, dict) else []
            
            candidate_context = f"""CANDIDATE PROFILE:
- Experience: {candidate.years_experience} years
- Technical Skills: {', '.join(str(s) for s in tech_skills)}
- Analytics Skills: {', '.join(str(s) for s in analytics_skills)}
- Business Skills: {', '.join(str(s) for s in business_skills)}
- Finance Skills: {', '.join(str(s) for s in finance_skills)}
"""
            context_parts.append(candidate_context)
        except Exception as e:
            print(f"Error building candidate context: {e}")
        
        return "\n\n".join(context_parts)
    
    def augmented_generate(self,
                          user_query: str,
                          context: str,
                          task_type: str = "general") -> str:
        """
        AUGMENT + GENERATE phases: Combine context with query and generate
        
        TOPIC: Prompt Engineering - Task-specific prompts
        """
        
        # Task-specific system prompts (Prompt Engineering)
        system_prompts = {
            "career_advice": """You are an expert fintech career coach with deep knowledge of the industry.
Your advice is grounded in the real-time market data provided.
Be specific, actionable, and encouraging. Reference actual companies and trends from the context.
Format your response with clear sections and bullet points where helpful.""",

            "cover_letter": """You are an expert career writer specializing in fintech applications.
Write compelling, personalized cover letters that:
1. Open with a hook relevant to the company's recent news/momentum
2. Connect the candidate's specific experience to the role
3. Demonstrate knowledge of the company's position in the market
4. Close with enthusiasm and a clear call to action
Keep it under 300 words. Be professional but personable.""",

            "interview_prep": """You are a fintech interview coach who has helped hundreds of candidates succeed.
Based on the company's recent activity and the role requirements, generate:
1. Likely interview questions (technical and behavioral)
2. Suggested answers leveraging the candidate's background
3. Questions the candidate should ask to show market awareness
Be specific to fintech and reference current market conditions.""",

            "salary_negotiation": """You are a compensation negotiation expert in the fintech industry.
Based on the market data, company momentum, and candidate profile, provide:
1. Estimated salary range for this role
2. Key leverage points for negotiation
3. Specific talking points based on the company's growth stage
4. When and how to negotiate
Be confident but realistic.""",

            "general": """You are an AI career intelligence assistant specializing in fintech.
Provide helpful, accurate, and actionable guidance based on the real-time market data provided.
Be concise and specific."""
        }
        
        system_prompt = system_prompts.get(task_type, system_prompts["general"])
        
        # Construct RAG prompt
        rag_prompt = f"""CONTEXT (Use this real-time data to ground your response):
{context}

USER REQUEST:
{user_query}

Provide a response that specifically references the context data where relevant."""
        
        return self.llm.generate(rag_prompt, system_prompt)


# ============================================
# 6. CAREER AGENT - Agentic AI Design
# ============================================
class CareerAgent:
    """
    Autonomous career intelligence agent that reasons about opportunities.
    
    TOPIC: Agentic Design & Multi-Agent Concepts
    
    Agent Capabilities:
    1. PERCEIVE: Gather information from environment (APIs, resume)
    2. REASON: Analyze opportunities using structured thinking
    3. ACT: Generate recommendations and content
    4. EXPLAIN: Provide transparent reasoning (Responsible AI)
    """
    
    def __init__(self, llm_client: LLMClient, rag_engine: RAGEngine):
        self.llm = llm_client
        self.rag = rag_engine
        self.reasoning_trace = []  # For explainability
        
    def analyze_opportunity(self, 
                           job: Dict, 
                           candidate: CandidateProfile,
                           market_intel: MarketIntelligence) -> Dict:
        """
        Agentic analysis of a single opportunity.
        
        TOPIC: Agentic Reasoning with Chain-of-Thought
        """
        
        company = job.get('company', 'Unknown')
        company_data = market_intel.company_signals.get(company, {})
        
        # Step 1: PERCEIVE - Gather relevant information
        perception = {
            "job_title": job.get('title'),
            "company": company,
            "company_momentum": company_data.get('momentum_score', 50),
            "company_signals": company_data.get('signals', []),
            "is_hiring": company_data.get('is_hiring', False),
            "candidate_experience": candidate.years_experience,
            "skill_overlap": self._calculate_skill_overlap(candidate, job)
        }
        
        # Step 2: REASON - Structured analysis
        reasoning_prompt = f"""Analyze this job opportunity for the candidate.

JOB: {job.get('title')} at {company}
Description: {job.get('description', '')[:500]}

COMPANY INTELLIGENCE:
- Momentum Score: {perception['company_momentum']}/100
- Recent Signals: {', '.join(perception['company_signals']) if perception['company_signals'] else 'No recent signals'}
- Actively Hiring: {'Yes' if perception['is_hiring'] else 'Unknown'}

CANDIDATE:
- Years Experience: {candidate.years_experience}
- Skills: {', '.join(candidate.skills.get('technical', []) + candidate.skills.get('finance', []))}

Provide a brief analysis (2-3 sentences) covering:
1. Fit assessment
2. Key opportunity or concern
3. Strategic recommendation

Be direct and specific."""

        ai_analysis = self.llm.generate(
            reasoning_prompt,
            system_prompt="You are a strategic career advisor. Be concise and insightful.",
            temperature=0.5,
            max_tokens=200
        )
        
        # Step 3: Record reasoning trace (Responsible AI - Explainability)
        self.reasoning_trace.append({
            "job": job.get('title'),
            "company": company,
            "perception": perception,
            "analysis": ai_analysis,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "job": job,
            "perception": perception,
            "ai_analysis": ai_analysis,
            "reasoning_trace": self.reasoning_trace[-1]
        }
    
    def _calculate_skill_overlap(self, candidate: CandidateProfile, job: Dict) -> float:
        """Calculate skill match percentage"""
        candidate_skills = set()
        for skills in candidate.skills.values():
            if isinstance(skills, list):
                candidate_skills.update([s.lower() for s in skills if isinstance(s, str)])
        
        # Ensure job description is a string
        job_desc = job.get('description', '')
        if not isinstance(job_desc, str):
            job_desc = str(job_desc) if job_desc else ''
        job_text = job_desc.lower()
        
        matched = sum(1 for skill in candidate_skills if skill in job_text)
        
        return round(matched / max(len(candidate_skills), 1) * 100, 1)
    
    def generate_action_plan(self,
                            top_opportunities: List[Dict],
                            candidate: CandidateProfile,
                            market_intel: MarketIntelligence) -> str:
        """
        Generate a strategic action plan based on agent analysis.
        
        TOPIC: Agentic Planning
        """
        
        context = self.rag.retrieve_context("career action plan", market_intel, candidate)
        
        opportunities_summary = "\n".join([
            f"- {opp['job'].get('title')} at {opp['job'].get('company')} (Fit: {opp['perception']['skill_overlap']}%)"
            for opp in top_opportunities[:5]
        ])
        
        plan_prompt = f"""Based on the market analysis and candidate profile, create a 2-week action plan.

TOP OPPORTUNITIES IDENTIFIED:
{opportunities_summary}

Create a specific, actionable plan with:
1. Week 1 priorities (which jobs to apply to first and why)
2. Week 2 follow-ups
3. Skills to highlight or develop
4. Networking targets based on market momentum

Be specific and reference the actual opportunities."""

        return self.rag.augmented_generate(plan_prompt, context, "career_advice")


# ============================================
# 7. GENERATIVE CONTENT ENGINE
# ============================================
class ContentGenerator:
    """
    Generative AI for creating personalized career content.
    
    TOPIC: Generative AI Applications
    """
    
    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
        
    def generate_cover_letter(self,
                             job: Dict,
                             candidate: CandidateProfile,
                             market_intel: MarketIntelligence) -> str:
        """
        Generate a personalized cover letter using RAG.
        
        TOPIC: Generative AI + RAG for personalization
        """
        
        company = job.get('company', 'the company')
        company_data = market_intel.company_signals.get(company, {})
        
        # Build rich context for generation
        context = f"""TARGET COMPANY: {company}
Recent Activity: {', '.join(company_data.get('signals', ['No recent signals']))}
Company Momentum: {company_data.get('momentum_score', 'Unknown')}/100
Recent News: {company_data.get('latest_news', [{}])[0].get('title', 'N/A') if company_data.get('latest_news') else 'N/A'}

JOB DETAILS:
Title: {job.get('title')}
Description: {job.get('description', '')[:500]}

CANDIDATE BACKGROUND:
Years of Experience: {candidate.years_experience}
Technical Skills: {', '.join(candidate.skills.get('technical', []))}
Finance Skills: {', '.join(candidate.skills.get('finance', []))}
Business Skills: {', '.join(candidate.skills.get('business', []))}
"""
        
        prompt = f"""Write a compelling cover letter for this candidate applying to {job.get('title')} at {company}.

The letter should:
1. Open by referencing the company's recent momentum/news to show market awareness
2. Connect 2-3 specific skills from the candidate's background to the job requirements
3. Demonstrate understanding of fintech industry trends
4. Close with enthusiasm and clear interest in interviewing

Keep it professional, concise (under 300 words), and personalized."""

        return self.rag.augmented_generate(prompt, context, "cover_letter")
    
    def generate_interview_prep(self,
                               job: Dict,
                               candidate: CandidateProfile,
                               market_intel: MarketIntelligence) -> str:
        """
        Generate customized interview preparation guide.
        """
        
        company = job.get('company', 'the company')
        company_data = market_intel.company_signals.get(company, {})
        
        context = f"""COMPANY: {company}
Momentum: {company_data.get('momentum_score', 50)}/100
Signals: {', '.join(company_data.get('signals', []))}
Is Hiring: {company_data.get('is_hiring', 'Unknown')}

ROLE: {job.get('title')}
Description: {job.get('description', '')[:400]}

CANDIDATE:
Experience: {candidate.years_experience} years
Skills: {', '.join(candidate.skills.get('technical', []) + candidate.skills.get('finance', []))}
"""
        
        prompt = """Generate a comprehensive interview preparation guide including:

1. **Likely Technical Questions** (3 questions with suggested answers)
2. **Behavioral Questions** (3 questions with STAR-format answer frameworks)
3. **Questions to Ask Them** (3 questions showing market awareness)
4. **Company-Specific Prep** (what to research about their recent activity)

Make all suggestions specific to this role and company."""

        return self.rag.augmented_generate(prompt, context, "interview_prep")
    
    def generate_salary_insights(self,
                                job: Dict,
                                candidate: CandidateProfile,
                                market_intel: MarketIntelligence) -> str:
        """
        Generate salary negotiation guidance based on market data.
        """
        
        company = job.get('company', 'the company')
        company_data = market_intel.company_signals.get(company, {})
        
        context = f"""COMPANY: {company}
Momentum: {company_data.get('momentum_score', 50)}/100
Recent Signals: {', '.join(company_data.get('signals', []))}
Listed Salary: {job.get('salary_range', 'Not disclosed')}

ROLE: {job.get('title')}

CANDIDATE:
Experience: {candidate.years_experience} years
Skills: {', '.join(candidate.skills.get('technical', []))}

MARKET CONTEXT:
High-momentum companies in current market: {len([c for c, d in market_intel.company_signals.items() if d.get('momentum_score', 0) > 60])}
"""
        
        prompt = """Provide salary negotiation guidance including:

1. **Estimated Range** for this role based on experience and market
2. **Leverage Points** based on candidate's background and company's momentum
3. **Negotiation Script** - specific phrases to use
4. **Timing Advice** - when to bring up compensation

Be realistic but advocate for the candidate."""

        return self.rag.augmented_generate(prompt, context, "salary_negotiation")
    
    def generate_ats_analysis(self,
                              job: Dict,
                              candidate: CandidateProfile,
                              market_intel: MarketIntelligence) -> Dict:
        """
        Generate comprehensive ATS + Recruiter analysis.
        
        FIXED: Actually parses EACH job description instead of using static requirements
        
        TOPIC: Domain-specific AI - ATS Optimization
        """
        
        job_description = str(job.get('description', '') or '').lower()
        job_title = str(job.get('title', '') or '').lower()
        company = str(job.get('company', '') or '')
        
        # Combine job text for analysis
        job_text = job_description + " " + job_title
        
        # Get all candidate skills as flat list
        candidate_skills = []
        for category, skills in candidate.skills.items():
            if isinstance(skills, list):
                candidate_skills.extend([s.lower() for s in skills if isinstance(s, str)])
        candidate_skills_str = ' '.join(candidate_skills)
        
        # Get raw resume text for deeper analysis
        resume_text = candidate.raw_text.lower() if candidate.raw_text else ''
        
        # ============================================
        # 1. EXTRACT SKILLS FROM THIS SPECIFIC JOB
        # ============================================
        
        # Comprehensive skill dictionary to look for
        all_skills_to_check = {
            # Technical - High weight
            "sql": 15, "python": 15, "excel": 12, "tableau": 10, "power bi": 10,
            "r": 8, "java": 10, "javascript": 10, "c++": 8, "scala": 8,
            "aws": 10, "azure": 10, "gcp": 8, "docker": 8, "kubernetes": 8,
            "spark": 10, "hadoop": 8, "airflow": 8, "dbt": 8,
            "snowflake": 10, "databricks": 10, "redshift": 8,
            "postgresql": 8, "mysql": 8, "mongodb": 8, "nosql": 6,
            "git": 6, "api": 8, "rest": 6, "graphql": 6,
            
            # Analytics - Medium weight
            "machine learning": 12, "deep learning": 10, "nlp": 10,
            "statistics": 10, "statistical": 10, "regression": 8,
            "a/b testing": 10, "ab testing": 10, "experimentation": 8,
            "forecasting": 8, "time series": 8, "predictive": 8,
            "data analysis": 10, "analytics": 8, "metrics": 6,
            "visualization": 8, "dashboard": 8, "reporting": 6,
            "etl": 8, "data pipeline": 8, "data engineering": 10,
            
            # Business - Lower weight
            "stakeholder": 6, "communication": 5, "presentation": 5,
            "strategy": 6, "business": 5, "cross-functional": 6,
            "project management": 6, "agile": 5, "scrum": 5,
            "financial": 8, "modeling": 8, "valuation": 8,
            
            # Tools
            "pandas": 8, "numpy": 6, "scikit-learn": 8, "tensorflow": 10,
            "pytorch": 10, "keras": 8, "xgboost": 6,
            "looker": 8, "mode": 6, "amplitude": 6, "mixpanel": 6,
            "jira": 4, "confluence": 4, "slack": 3,
        }
        
        # Find skills mentioned in THIS job description
        job_required_skills = {}
        for skill, base_weight in all_skills_to_check.items():
            if skill in job_text:
                # Adjust weight based on frequency/emphasis
                count = job_text.count(skill)
                weight = base_weight * (1 + min(count - 1, 2) * 0.2)  # Boost for repeated mentions
                job_required_skills[skill] = round(weight)
        
        # If job description is too short/vague, add baseline requirements based on title
        detected_role = "analyst"  # Default
        if len(job_required_skills) < 3:
            if "data analyst" in job_title or "analyst" in job_title:
                job_required_skills.update({"sql": 15, "excel": 12, "data analysis": 10, "reporting": 6})
                detected_role = "data analyst"
            elif "data scientist" in job_title:
                job_required_skills.update({"python": 15, "machine learning": 12, "sql": 10, "statistics": 10})
                detected_role = "data scientist"
            elif "engineer" in job_title:
                job_required_skills.update({"python": 15, "sql": 10, "git": 8, "aws": 8})
                detected_role = "software engineer"
        else:
            # Detect role from title for reporting
            if "scientist" in job_title:
                detected_role = "data scientist"
            elif "engineer" in job_title:
                detected_role = "software engineer"
            elif "analyst" in job_title:
                detected_role = "data analyst"
        
        # ============================================
        # 2. MATCH CANDIDATE SKILLS TO JOB REQUIREMENTS
        # ============================================
        
        skill_breakdown = []
        matched_skills = []
        missing_skills = []
        total_weight = 0
        matched_weight = 0
        
        # Sort by weight for importance ordering
        sorted_job_skills = sorted(job_required_skills.items(), key=lambda x: x[1], reverse=True)
        
        for skill, weight in sorted_job_skills:
            total_weight += weight
            # Check if candidate has this skill
            has_skill = skill in candidate_skills_str or skill in resume_text
            
            if has_skill:
                matched_weight += weight
                matched_skills.append(skill.title())
                status = "✅"
            else:
                missing_skills.append(skill.title())
                status = "❌"
            
            # Determine importance level
            if weight >= 12:
                importance = "critical"
                importance_emoji = "🔥"
            elif weight >= 8:
                importance = "important"
                importance_emoji = "⭐"
            else:
                importance = "nice_to_have"
                importance_emoji = "🟡"
            
            skill_breakdown.append({
                "skill": skill.title(),
                "importance": importance,
                "importance_emoji": importance_emoji,
                "weight": weight,
                "status": status,
                "has_skill": has_skill,
                "in_job_desc": True
            })
        
        # Calculate skill score (weighted)
        skill_score = round((matched_weight / total_weight) * 100) if total_weight > 0 else 50
        
        # ============================================
        # 2. SENIORITY SIGNAL DETECTION
        # ============================================
        
        seniority_signals = {
            "leadership": {
                "keywords": ["led", "lead", "managed", "directed", "supervised", "mentored", "coached"],
                "found": [],
                "score": 0
            },
            "ownership": {
                "keywords": ["owned", "built", "created", "designed", "architected", "established", "launched"],
                "found": [],
                "score": 0
            },
            "strategic": {
                "keywords": ["strategy", "roadmap", "vision", "executive", "c-level", "board", "stakeholder"],
                "found": [],
                "score": 0
            },
            "cross_functional": {
                "keywords": ["cross-functional", "cross functional", "collaborated", "partnered", "aligned"],
                "found": [],
                "score": 0
            }
        }
        
        for signal_type, data in seniority_signals.items():
            for keyword in data["keywords"]:
                if keyword in resume_text:
                    data["found"].append(keyword)
            data["score"] = min(len(data["found"]) * 25, 100)  # Cap at 100
        
        # Calculate overall seniority score
        seniority_score = round(sum(d["score"] for d in seniority_signals.values()) / len(seniority_signals))
        
        # Determine seniority level expected from job title
        expected_seniority = "mid"
        if any(x in job_title for x in ["senior", "sr.", "lead", "principal", "staff", "director", "head"]):
            expected_seniority = "senior"
        elif any(x in job_title for x in ["junior", "jr.", "entry", "associate", "intern"]):
            expected_seniority = "junior"
        
        # ============================================
        # 3. IMPACT SCORE (Business Lens)
        # ============================================
        
        # Look for quantified achievements
        # Patterns for impact indicators
        dollar_pattern = r'\$[\d,]+[kmb]?|\d+\s*(?:million|billion|thousand)'
        percent_pattern = r'\d+%|\d+\s*percent'
        number_pattern = r'\d+[x×]\s*|\d+\s*(?:users|customers|clients|transactions|records)'
        scale_words = ["enterprise", "global", "nationwide", "company-wide", "millions", "thousands", "scale"]
        
        dollar_matches = re.findall(dollar_pattern, resume_text, re.IGNORECASE)
        percent_matches = re.findall(percent_pattern, resume_text, re.IGNORECASE)
        number_matches = re.findall(number_pattern, resume_text, re.IGNORECASE)
        scale_matches = [w for w in scale_words if w in resume_text]
        
        # Count metrics
        metrics_count = len(dollar_matches) + len(percent_matches) + len(number_matches)
        
        # Calculate impact score
        impact_components = {
            "metrics_present": min(metrics_count * 15, 40),  # Up to 40 points
            "financial_impact": min(len(dollar_matches) * 20, 30),  # Up to 30 points
            "scale_indicators": min(len(scale_matches) * 10, 30)  # Up to 30 points
        }
        
        impact_score = sum(impact_components.values())
        
        # ============================================
        # 4. EXPERIENCE MATCH
        # ============================================
        
        # Extract required experience from job
        exp_patterns = [r'(\d+)\+?\s*years?', r'(\d+)\+?\s*yrs?']
        required_exp = 0
        for pattern in exp_patterns:
            match = re.search(pattern, job_text)  # job_text is already lowercase
            if match:
                required_exp = int(match.group(1))
                break
        
        # Infer from title if not found
        if required_exp == 0:
            if expected_seniority == "senior":
                required_exp = 5
            elif expected_seniority == "junior":
                required_exp = 0
            else:
                required_exp = 2
        
        exp_match = candidate.years_experience >= required_exp
        exp_gap = max(required_exp - candidate.years_experience, 0)
        
        # ============================================
        # 5. CALCULATE DUAL SCORES
        # ============================================
        
        # ATS Score (keyword/skill focused)
        ats_score = skill_score
        if not exp_match:
            ats_score = max(ats_score - (exp_gap * 10), 20)
        
        # Recruiter Score (holistic evaluation)
        # Weighted: Skills 40%, Seniority 25%, Impact 25%, Experience 10%
        recruiter_score = round(
            (skill_score * 0.40) +
            (seniority_score * 0.25) +
            (impact_score * 0.25) +
            (100 if exp_match else max(100 - exp_gap * 15, 30)) * 0.10
        )
        
        # Penalties for senior roles without seniority signals
        if expected_seniority == "senior" and seniority_score < 40:
            recruiter_score = max(recruiter_score - 15, 30)
        
        # ============================================
        # 6. GENERATE REJECTION REASONS
        # ============================================
        
        rejection_reasons = []
        
        # Check critical skills
        critical_missing = [s for s in skill_breakdown if s["importance"] == "critical" and not s["has_skill"]]
        if critical_missing:
            rejection_reasons.append(f"Missing critical skills: {', '.join([s['skill'] for s in critical_missing])}")
        
        # Check seniority mismatch
        if expected_seniority == "senior" and seniority_score < 40:
            rejection_reasons.append("Senior role but resume lacks leadership/ownership language")
        
        # Check impact
        if impact_score < 30:
            rejection_reasons.append("No quantified business impact (missing %, $, or scale metrics)")
        
        # Check experience
        if not exp_match:
            rejection_reasons.append(f"Experience gap: Role needs {required_exp}+ years, you have {candidate.years_experience}")
        
        return {
            "ats_score": min(ats_score, 100),
            "recruiter_score": min(recruiter_score, 100),
            "skill_score": skill_score,
            "skill_breakdown": skill_breakdown,
            "seniority_score": seniority_score,
            "seniority_signals": seniority_signals,
            "expected_seniority": expected_seniority,
            "impact_score": impact_score,
            "impact_components": impact_components,
            "metrics_found": {
                "dollars": dollar_matches[:3],
                "percentages": percent_matches[:3],
                "numbers": number_matches[:3],
                "scale": scale_matches
            },
            "required_experience": required_exp,
            "candidate_experience": candidate.years_experience,
            "experience_match": exp_match,
            "experience_gap": exp_gap,
            "rejection_reasons": rejection_reasons,
            "matched_skills": [s["skill"] for s in skill_breakdown if s["has_skill"]],
            "missing_skills": [s["skill"] for s in skill_breakdown if not s["has_skill"]],
            "detected_role": detected_role
        }
    
    def generate_resume_recommendations(self,
                                        job: Dict,
                                        candidate: CandidateProfile,
                                        ats_analysis: Dict,
                                        market_intel: MarketIntelligence) -> str:
        """
        Generate specific recommendations to improve resume match.
        
        TOPIC: Generative AI for personalized career coaching
        """
        
        if ats_analysis['ats_score'] >= 90:
            return "✅ **Excellent match!** Your resume is well-aligned with this role."
        
        missing = ats_analysis['missing_skills']
        matched = ats_analysis['matched_skills']
        company = job.get('company', 'the company')
        title = job.get('title', 'this role')
        
        # Combine matched + missing to get all required skills
        all_required = matched + missing
        
        context = f"""JOB: {title} at {company}
Required Skills for Role: {', '.join(all_required)}
Candidate Has: {', '.join(matched)}
Candidate Missing: {', '.join(missing)}
Required Experience: {ats_analysis['required_experience']} years
Candidate Experience: {ats_analysis['candidate_experience']} years
Current ATS Score: {ats_analysis['ats_score']}%
Recruiter Score: {ats_analysis.get('recruiter_score', ats_analysis['ats_score'])}%
"""
        
        prompt = f"""The candidate's resume has a {ats_analysis['ats_score']}% ATS match for this role.

Missing skills: {', '.join(missing) if missing else 'None'}

Generate specific, actionable recommendations:

1. **Quick Wins** - Keywords to add to resume TODAY (exact phrases from job posting)
2. **Project Ideas** - 2-3 specific projects they could build/add to demonstrate missing skills
3. **Experience Reframing** - How to reword existing experience to match job requirements
4. **Priority Order** - Which gaps to address first (most important for this role)

Be specific. Instead of "learn SQL", say "Add a project: 'Built a customer analytics dashboard using SQL and Tableau analyzing 100K+ transactions'"
"""
        
        return self.rag.augmented_generate(prompt, context, "career_advice")
    
    def generate_top_performer_benchmark(self,
                                         target_role: str,
                                         candidate: CandidateProfile,
                                         job: Dict = None) -> str:
        """
        Generate HONEST, DEFENSIBLE role readiness assessment.
        
        NO fake percentiles. NO unsubstantiated claims.
        Uses transparent rubric that can be explained and defended.
        
        TOPIC: Domain-specific AI - Career Benchmarking
        """
        
        # Get job details
        job_title = job.get('title', target_role) if job else target_role
        company = job.get('company', 'Target Company') if job else 'Target Company'
        
        # Get all candidate skills
        candidate_skills = []
        for skills in candidate.skills.values():
            if isinstance(skills, list):
                candidate_skills.extend([s.lower() for s in skills])
        candidate_skills_str = ' '.join(candidate_skills)
        resume_text = candidate.raw_text.lower() if candidate.raw_text else ''
        
        # ============================================
        # DIMENSION 1: Required Skills (30% weight)
        # ============================================
        
        role_core_skills = {
            "data analyst": ["sql", "excel", "python", "tableau", "data analysis", "statistics"],
            "data scientist": ["python", "machine learning", "sql", "statistics", "pandas", "modeling"],
            "software engineer": ["python", "git", "api", "testing", "software development", "debugging"],
            "product manager": ["product", "roadmap", "stakeholder", "strategy", "metrics", "user research"],
            "consultant": ["excel", "powerpoint", "analysis", "strategy", "presentations", "problem solving"],
            "default": ["sql", "excel", "python", "analysis", "communication", "problem solving"]
        }
        
        # Detect role type
        role_key = "default"
        for key in role_core_skills.keys():
            if key in target_role.lower():
                role_key = key
                break
        
        required_skills = role_core_skills[role_key]
        skills_matched = [s for s in required_skills if s in candidate_skills_str or s in resume_text]
        skills_missing = [s for s in required_skills if s not in candidate_skills_str and s not in resume_text]
        skills_score = round(len(skills_matched) / len(required_skills) * 100) if required_skills else 50
        
        # ============================================
        # DIMENSION 2: Experience Level (25% weight)
        # ============================================
        
        # Infer required experience from job title
        if any(x in job_title.lower() for x in ["senior", "sr.", "lead", "principal", "staff"]):
            required_exp = 5
            exp_level = "Senior"
        elif any(x in job_title.lower() for x in ["junior", "jr.", "entry", "associate"]):
            required_exp = 0
            exp_level = "Junior"
        else:
            required_exp = 2
            exp_level = "Mid-level"
        
        exp_score = 100 if candidate.years_experience >= required_exp else max(0, 100 - (required_exp - candidate.years_experience) * 20)
        exp_status = "EXCEEDS" if candidate.years_experience > required_exp else ("MEETS" if candidate.years_experience >= required_exp else "BELOW")
        
        # ============================================
        # DIMENSION 3: Seniority Signals (20% weight)
        # ============================================
        
        leadership_keywords = ["led", "lead", "managed", "directed", "supervised", "mentored", "owned", "built", "architected"]
        leadership_found = [k for k in leadership_keywords if k in resume_text]
        seniority_score = min(len(leadership_found) * 15, 100)
        
        # ============================================
        # DIMENSION 4: Business Impact (15% weight)
        # ============================================
        
        import re
        metrics_patterns = [r'\d+%', r'\$[\d,]+', r'\d+x', r'\d+\s*million', r'\d+\s*users']
        metrics_found = []
        for pattern in metrics_patterns:
            metrics_found.extend(re.findall(pattern, resume_text, re.IGNORECASE))
        
        impact_score = min(len(metrics_found) * 20, 100)
        
        # ============================================
        # DIMENSION 5: Modern Tools (10% weight)
        # ============================================
        
        modern_tools = ["dbt", "airflow", "spark", "snowflake", "databricks", "kubernetes", "docker", "mlops"]
        modern_found = [t for t in modern_tools if t in candidate_skills_str or t in resume_text]
        modern_score = min(len(modern_found) * 25, 100)
        
        # ============================================
        # CALCULATE OVERALL SCORE
        # ============================================
        
        overall_score = round(
            (skills_score * 0.30) +
            (exp_score * 0.25) +
            (seniority_score * 0.20) +
            (impact_score * 0.15) +
            (modern_score * 0.10)
        )
        
        # Determine readiness level (NOT percentile - we can't claim that)
        if overall_score >= 80:
            readiness = "✅ STRONG"
            readiness_desc = "You meet or exceed requirements across most dimensions."
        elif overall_score >= 60:
            readiness = "🟡 COMPETITIVE"
            readiness_desc = "You're qualified but have addressable gaps."
        elif overall_score >= 40:
            readiness = "🟠 DEVELOPING"
            readiness_desc = "You have foundations but need to strengthen key areas."
        else:
            readiness = "🔴 BUILDING"
            readiness_desc = "Focus on building core skills before applying."
        
        # ============================================
        # IDENTIFY TOP GAPS
        # ============================================
        
        gaps = []
        if skills_score < 80:
            gaps.append(("Required Skills", skills_score, f"Missing: {', '.join(skills_missing[:3])}"))
        if seniority_score < 50 and exp_level == "Senior":
            gaps.append(("Seniority Signals", seniority_score, "Add leadership language: 'led', 'owned', 'drove'"))
        if impact_score < 40:
            gaps.append(("Business Impact", impact_score, "Add quantified achievements (%, $, scale)"))
        if modern_score < 30:
            gaps.append(("Modern Tools", modern_score, f"Consider learning: {', '.join([t for t in modern_tools if t not in candidate_skills_str][:2])}"))
        
        # Sort gaps by score (lowest first = highest priority)
        gaps.sort(key=lambda x: x[1])
        
        # ============================================
        # BUILD OUTPUT
        # ============================================
        
        output = f"""### 📊 Role Readiness Score: {overall_score}/100

**For:** {job_title} at {company}

**Readiness Level:** {readiness}

{readiness_desc}

---

### 🔍 Transparent Scoring Rubric

| Dimension | Weight | Your Score | How We Calculated |
|-----------|--------|------------|-------------------|
| Required Skills | 30% | **{skills_score}/100** | {len(skills_matched)}/{len(required_skills)} core skills detected |
| Experience Level | 25% | **{exp_score}/100** | {candidate.years_experience} yrs vs {required_exp}+ required ({exp_status}) |
| Seniority Signals | 20% | **{seniority_score}/100** | {len(leadership_found)} leadership keywords found |
| Business Impact | 15% | **{impact_score}/100** | {len(metrics_found)} quantified metrics detected |
| Modern Tools | 10% | **{modern_score}/100** | {len(modern_found)}/{len(modern_tools)} modern tools |

---

### ✅ Your Strengths

"""
        
        # Add strengths
        if skills_score >= 70:
            output += f"- **Core skills solid:** You have {len(skills_matched)}/{len(required_skills)} required skills ({', '.join(skills_matched[:4])})\n"
        if exp_score >= 80:
            output += f"- **Experience sufficient:** {candidate.years_experience} years {'exceeds' if exp_status == 'EXCEEDS' else 'meets'} the {required_exp}+ year requirement\n"
        if seniority_score >= 50:
            output += f"- **Leadership evident:** Resume shows ownership language ({', '.join(leadership_found[:3])})\n"
        if impact_score >= 50:
            output += f"- **Impact quantified:** {len(metrics_found)} metrics found in resume\n"
        
        if skills_score < 70 and exp_score < 80 and seniority_score < 50 and impact_score < 50:
            output += "- *Building foundations — focus on the gaps below*\n"
        
        output += """
---

### 🎯 Priority Gaps to Address

"""
        
        if gaps:
            for i, (dimension, score, action) in enumerate(gaps[:3], 1):
                output += f"**{i}. {dimension}** ({score}/100)\n"
                output += f"   → Action: {action}\n\n"
        else:
            output += "✅ No critical gaps detected. Focus on tailoring your application.\n"
        
        output += """---

### 📅 Action Plan

"""
        
        # Generate specific, time-bound actions
        output += "**This Week:**\n"
        if impact_score < 50:
            output += "- [ ] Add 3 quantified achievements to your resume (%, $, or scale)\n"
        if seniority_score < 50 and exp_level == "Senior":
            output += "- [ ] Replace 'worked on' with 'led', 'owned', 'drove' in 3 bullet points\n"
        if skills_missing:
            output += f"- [ ] Add '{skills_missing[0]}' keyword to your resume (if you have this skill)\n"
        
        output += "\n**This Month:**\n"
        if modern_score < 50:
            output += f"- [ ] Complete a tutorial on {modern_tools[0] if modern_tools[0] not in candidate_skills_str else modern_tools[1]} (2-4 hours)\n"
        output += "- [ ] Update LinkedIn to match optimized resume\n"
        
        output += "\n**Before Applying:**\n"
        output += "- [ ] Tailor resume keywords to this specific job description\n"
        output += "- [ ] Prepare 2-3 STAR stories demonstrating impact\n"
        
        # Company-specific insights if available
        company_benchmark = COMPANY_BENCHMARKS.get(company, {})
        if company_benchmark:
            output += f"""
---

### 🏢 {company}-Specific Insights

- **Culture:** {company_benchmark.get('culture', 'N/A')}
- **Interview Focus:** {', '.join(company_benchmark.get('interview_focus', ['N/A']))}
- **They Value:** {company_benchmark.get('what_they_value', 'N/A')}
- **Stand Out By:** {company_benchmark.get('standout_for_them', 'N/A')}
"""
        
        output += """
---

### ⚠️ Methodology Note

**What this score IS:**
- A rubric-based assessment of your resume against typical role requirements
- Calculated from keyword detection and pattern matching
- Transparent and reproducible

**What this score is NOT:**
- A comparison to other candidates (we don't have that data)
- A guarantee of interview success
- A substitute for human judgment

*This assessment is based on resume text analysis. Actual hiring decisions involve many factors we cannot measure.*
"""
        
        return output
    
    def generate_career_roadmap(self,
                                target_role: str,
                                candidate: CandidateProfile,
                                target_industry: str,
                                ats_analysis: Dict = None) -> str:
        """
        Generate OUTCOME-FOCUSED 90-day execution plan.
        
        NOT a learning plan. An INTERVIEW CONVERSION ENGINE.
        
        TOPIC: Generative AI - Personalized Career Planning
        """
        
        # Get candidate's current scores for baseline
        current_ats = ats_analysis.get('ats_score', 50) if ats_analysis else 50
        current_recruiter = ats_analysis.get('recruiter_score', 50) if ats_analysis else 50
        current_impact = ats_analysis.get('impact_score', 30) if ats_analysis else 30
        current_seniority = ats_analysis.get('seniority_score', 30) if ats_analysis else 30
        
        # Calculate targets
        target_ats = min(current_ats + 20, 95)
        target_recruiter = min(current_recruiter + 25, 90)
        
        # Get skills info
        candidate_skills = []
        for skills in candidate.skills.values():
            if isinstance(skills, list):
                candidate_skills.extend([s.lower() for s in skills])
        
        has_sql = any('sql' in s for s in candidate_skills)
        has_python = any('python' in s for s in candidate_skills)
        has_tableau = any('tableau' in s or 'power bi' in s for s in candidate_skills)
        
        # Determine industry-specific KPIs
        industry_kpis = {
            "Fintech": ["default rate", "approval rate", "revenue per user", "LTV", "CAC", "churn rate"],
            "FAANG / Big Tech": ["DAU/MAU", "conversion rate", "retention", "engagement", "latency", "uptime"],
            "Healthcare / Healthtech": ["patient outcomes", "readmission rate", "cost per patient", "utilization", "compliance rate"],
            "E-Commerce / Retail Tech": ["conversion rate", "AOV", "cart abandonment", "repeat purchase rate", "inventory turnover"],
            "Consulting / Professional Services": ["utilization rate", "revenue per consultant", "project margin", "client retention"],
            "AI / Machine Learning": ["model accuracy", "latency", "throughput", "A/B test lift", "false positive rate"]
        }
        
        kpis = industry_kpis.get(target_industry, industry_kpis["Fintech"])
        
        # Build the roadmap
        output = f"""### 🎯 NORTH STAR GOAL

**By Day 90:** Increase interview callback rate from ~5% → 15%+ for {target_role} roles in {target_industry}.

| Metric | Current | Target | How We'll Get There |
|--------|---------|--------|---------------------|
| ATS Pass Rate | {current_ats}% | {target_ats}% | Resume signal upgrade |
| Recruiter Shortlist | {current_recruiter}% | {target_recruiter}% | Impact + seniority proof |
| Interview Callbacks | ~5% | 15%+ | Market-aligned portfolio |

**This is an execution engine, not a learning plan.**

---

## ✅ PHASE 1: Resume Signal Upgrade (Weeks 1–2)

### 🎯 Goal: Pass ATS + recruiter skim in <10 seconds

**🚫 No courses allowed in this phase.**

### Required Outputs:

**1. Resume Upgrade Checklist:**
- [ ] Add ≥5 quantified impact bullets (%, $, scale)
- [ ] Include ≥3 {target_industry}-relevant KPIs: *{', '.join(kpis[:3])}*
- [ ] Write ≥1 end-to-end data story (problem → analysis → business outcome)
- [ ] Add ownership language: "Led", "Owned", "Drove" in 3+ bullets

**2. LinkedIn Optimization:**
- [ ] Headline rewritten for {target_industry} {target_role}
  - Before: "Data Professional | Analytics | SQL"
  - After: "{target_role} | {target_industry} | Driving {kpis[0]} through data"
- [ ] About section: 3 sentences with quantified achievements
- [ ] Featured: Add 1 portfolio project screenshot

**3. Validation Checkpoint (Day 14):**

| Check | Target | Self-Assess |
|-------|--------|-------------|
| Resume ATS Score | ≥{target_ats}% | Run through our optimizer |
| Quantified bullets | ≥5 | Count them |
| {target_industry} keywords | ≥3 KPIs mentioned | {', '.join(kpis[:3])} |
| Ownership language | ≥3 instances | "Led", "Owned", "Drove" |

**If not met → Do not proceed to Phase 2.**

---

## ✅ PHASE 2: Market-Aligned Proof (Weeks 3–6)

### 🎯 Goal: Prove you can solve real {target_industry} problems

**🚫 No generic projects. Only high-signal artifacts.**

### MANDATORY Project 1: {target_industry} KPI Dashboard

**Why this matters:** Recruiters screen for domain relevance. This proves you understand {target_industry} metrics.

| Component | Requirement |
|-----------|-------------|
| **Dataset** | {target_industry} data (transactions, users, or operations) |
| **KPIs** | {kpis[0]}, {kpis[1]}, {kpis[2]} |
| **Tools** | SQL + {'Tableau/Power BI' if has_tableau else 'Python visualization'} |
| **Output 1** | Interactive dashboard |
| **Output 2** | 1-page executive summary with recommendations |

**Deliverables:**
- [ ] Dashboard live (Tableau Public or hosted)
- [ ] GitHub repo with SQL queries
- [ ] README with business context
- [ ] 1-page PDF: "Executive Summary: [KPI] Analysis"

**This alone beats 80% of portfolios.**

### Project 2 (If Time Permits): A/B Test Analysis

| Component | Requirement |
|-----------|-------------|
| **Scenario** | Product/feature test with conversion metrics |
| **Analysis** | Statistical significance, confidence intervals |
| **Tools** | SQL + Python (scipy.stats) |
| **Output** | Business recommendation with data backing |

**Deliverables:**
- [ ] Jupyter notebook with analysis
- [ ] Summary slide: "Test Results & Recommendation"

### 🚫 REMOVE These From Your Portfolio:
- Stock price prediction (everyone has this)
- Titanic/Iris/MNIST (too basic)
- Anything without business context

### Validation Checkpoint (Day 42):

| Check | Target |
|-------|--------|
| Dashboard complete | ✅ Live link |
| {target_industry} KPIs shown | ≥3 relevant metrics |
| Business narrative | Clear "so what" |
| GitHub documentation | README + queries |

---

## ✅ PHASE 3: Interview Conversion (Weeks 7–12)

### 🎯 Goal: Convert skills → interviews → offers

### Week 7-8: Application Engine

**Resume Variants:**
- [ ] Version A: {target_role} focus
- [ ] Version B: Product/Business Analyst focus (backup)

**Application Tracker:**

| Target | Quantity | Quality Bar |
|--------|----------|-------------|
| Applications sent | 20-30 | Only roles you're ≥70% qualified for |
| Tailored resumes | 100% | Each resume tweaked for JD keywords |
| Cover letters | Top 10 only | Only for dream companies |

**Daily Cadence:**
- 2-3 quality applications per day
- Track in spreadsheet: Company, Role, Date, Status, Follow-up

### Week 9-10: Interview Preparation

**STAR Stories Bank:**
Build 5 stories covering:
- [ ] Technical challenge you solved
- [ ] Business impact you drove
- [ ] Cross-functional collaboration
- [ ] Failure/learning experience
- [ ] Leadership/ownership moment

**{target_industry}-Specific Prep:**
- [ ] Research 3 target companies' recent news
- [ ] Prepare "Why {target_industry}?" narrative
- [ ] Practice explaining your dashboard project in 2 minutes

### Week 11-12: Optimization Loop

**Weekly Review:**
| Metric | Week 11 | Week 12 | Action if Low |
|--------|---------|---------|---------------|
| Applications sent | __ | __ | Increase volume |
| Response rate | __% | __% | Improve resume |
| Interview invites | __ | __ | Adjust targeting |

### Validation Checkpoint (Day 90):

| Outcome | Target | Actual |
|---------|--------|--------|
| Applications submitted | 40+ | ___ |
| Recruiter responses | ≥8 (20%) | ___ |
| Interviews scheduled | ≥3 | ___ |

**If interviews < 3 by Day 75:**
- Revisit resume (Phase 1)
- Add another project (Phase 2)
- Expand role/industry targeting

---

## 📊 Expected Outcomes (Based on Market Signals)

Following this execution plan is designed to:

| Outcome | Baseline | Expected | Improvement |
|---------|----------|----------|-------------|
| ATS Pass Rate | {current_ats}% | {target_ats}% | +{target_ats - current_ats}% |
| Recruiter Response | ~5% | 15-20% | 3-4x |
| Interview Invites | 1-2 | 5-8 | 3-4x |

**Methodology:** Improvements estimated based on:
- JD keyword frequency analysis
- Recruiter screening pattern research
- Portfolio differentiation benchmarks

*These are directional estimates, not guarantees. Actual results depend on market conditions, application quality, and role fit.*

---

## ⚠️ Critical Success Factors

**Do:**
- ✅ Focus on outputs, not inputs (artifacts > courses)
- ✅ Validate at each checkpoint before proceeding
- ✅ Track everything in a spreadsheet
- ✅ Iterate based on feedback

**Don't:**
- ❌ Skip Phase 1 (resume is gatekeeper)
- ❌ Build generic projects without business context
- ❌ Apply to 100 jobs without tracking results
- ❌ Wait until "ready" — start applying in Week 7

---

## 🎯 This Week's Action Items

**Today:**
1. [ ] Audit current resume for quantified bullets (count them)
2. [ ] List 3 {target_industry} KPIs you can speak to

**By End of Week:**
1. [ ] Resume has ≥5 quantified achievements
2. [ ] LinkedIn headline updated for {target_industry}
3. [ ] Dashboard project dataset identified

**Your north star:** Every action should increase your interview callback probability.
"""
        
        return output


# ============================================
# 8. RESPONSIBLE AI LAYER
# ============================================
class ResponsibleAIGuard:
    """
    Responsible AI safeguards and transparency.
    
    TOPIC: Responsible AI, AI Governance, Ethics
    
    Features:
    - Bias detection in recommendations
    - Transparency in AI reasoning
    - User control over AI outputs
    - Disclaimer generation
    """
    
    @staticmethod
    def add_transparency_disclaimer(content: str, content_type: str) -> str:
        """Add appropriate disclaimers to AI-generated content"""
        
        disclaimers = {
            "cover_letter": "\n\n---\n*🤖 AI-Generated: Review and personalize before sending. Verify all company details.*",
            "interview_prep": "\n\n---\n*🤖 AI-Generated: These are suggestions based on available data. Research the company directly for the most current information.*",
            "salary": "\n\n---\n*🤖 AI-Generated: Salary estimates are based on market data and may vary. Consult multiple sources and consider your specific circumstances.*",
            "analysis": "\n\n---\n*🤖 AI Analysis: Based on real-time market data. AI recommendations should supplement, not replace, your own judgment.*"
        }
        
        disclaimer = disclaimers.get(content_type, disclaimers["analysis"])
        return content + disclaimer
    
    @staticmethod
    def check_for_bias(recommendations: List[Dict]) -> Dict:
        """
        Check recommendations for potential biases.
        
        TOPIC: AI Ethics - Bias Detection
        """
        bias_report = {
            "geographic_concentration": False,
            "company_size_bias": False,
            "salary_range_bias": False,
            "warnings": []
        }
        
        # Check geographic concentration
        locations = [r.get('job', {}).get('location', '') for r in recommendations]
        location_counts = Counter(locations)
        if location_counts and max(location_counts.values()) > len(recommendations) * 0.7:
            bias_report["geographic_concentration"] = True
            bias_report["warnings"].append(
                "⚠️ Results concentrated in one location. Consider expanding geographic preferences."
            )
        
        return bias_report
    
    @staticmethod
    def get_reasoning_explanation(agent_trace: List[Dict]) -> str:
        """
        Generate human-readable explanation of AI reasoning.
        
        TOPIC: Explainable AI (XAI)
        """
        if not agent_trace:
            return "No reasoning trace available."
        
        explanation = "## 🔍 How I Analyzed Your Opportunities\n\n"
        
        for i, trace in enumerate(agent_trace[:3], 1):
            explanation += f"**{i}. {trace.get('job', 'Unknown')} at {trace.get('company', 'Unknown')}**\n"
            perception = trace.get('perception', {})
            explanation += f"- Company Momentum: {perception.get('company_momentum', 'N/A')}/100\n"
            explanation += f"- Skill Match: {perception.get('skill_overlap', 'N/A')}%\n"
            explanation += f"- Signals: {', '.join(perception.get('company_signals', ['None']))}\n\n"
        
        return explanation


# ============================================
# 9. TOP PERFORMER BENCHMARKS
# ============================================
"""
TOPIC: Domain-specific AI - Fintech Career Intelligence

This section contains curated knowledge about what makes
top 10% candidates stand out in each fintech field.
"""

TOP_PERFORMER_PROFILES = {
    "Data Analyst": {
        "must_have_skills": ["SQL", "Python", "Excel", "Tableau/Power BI", "Statistics"],
        "differentiator_skills": ["dbt", "Airflow", "Looker", "Mode Analytics", "Amplitude"],
        "standout_projects": [
            "Built automated KPI dashboard tracking $10M+ loan portfolio performance",
            "Created customer segmentation model that improved marketing ROI by 25%",
            "Developed fraud detection alerting system processing 100K+ daily transactions"
        ],
        "certifications": ["Google Data Analytics", "Tableau Desktop Specialist", "AWS Cloud Practitioner"],
        "experience_benchmark": "3-5 years with at least 1 year in fintech/finance",
        "what_top_10_percent_have": [
            "Experience with financial data (transactions, loans, payments)",
            "Built dashboards used by C-suite executives",
            "Quantified business impact ($X saved, Y% improvement)",
            "Experience with A/B testing in production",
            "SQL window functions and CTEs mastery"
        ]
    },
    "Strategy Analyst": {
        "must_have_skills": ["Excel", "PowerPoint", "SQL", "Financial Modeling", "Market Research"],
        "differentiator_skills": ["Python", "Tableau", "Competitive Intelligence Tools", "Gartner Access"],
        "standout_projects": [
            "Led market entry analysis for new product line generating $5M Year 1 revenue",
            "Built competitive intelligence framework tracking 20+ fintech competitors",
            "Developed pricing optimization model increasing margins by 15%"
        ],
        "certifications": ["CFA Level 1+", "Fintech Certificate (Wharton/MIT)", "Strategy Consulting Case Prep"],
        "experience_benchmark": "2-4 years in consulting, banking, or fintech strategy",
        "what_top_10_percent_have": [
            "Consulting or investment banking background",
            "Direct exposure to C-suite presentations",
            "M&A or due diligence experience",
            "Built financial models used for fundraising",
            "Published market research or thought leadership"
        ]
    },
    "Data Scientist": {
        "must_have_skills": ["Python", "SQL", "Machine Learning", "Statistics", "Pandas/NumPy"],
        "differentiator_skills": ["MLOps", "Feature Stores", "Real-time ML", "Causal Inference", "LLMs/NLP"],
        "standout_projects": [
            "Deployed credit risk model serving 1M+ decisions/month with <50ms latency",
            "Built recommendation engine increasing cross-sell conversion by 40%",
            "Created NLP pipeline extracting insights from 100K+ customer support tickets"
        ],
        "certifications": ["AWS ML Specialty", "Google ML Engineer", "Deep Learning Specialization"],
        "experience_benchmark": "3-6 years with production ML experience",
        "what_top_10_percent_have": [
            "Models running in production (not just notebooks)",
            "Experience with financial ML (credit, fraud, AML)",
            "A/B test design and causal inference skills",
            "MLOps pipeline experience (MLflow, Kubeflow, SageMaker)",
            "Published papers or Kaggle competition wins"
        ]
    },
    "Product Analyst": {
        "must_have_skills": ["SQL", "Product Analytics Tools", "A/B Testing", "Excel", "Data Visualization"],
        "differentiator_skills": ["Amplitude", "Mixpanel", "Python", "dbt", "Experimentation Platforms"],
        "standout_projects": [
            "Designed experimentation framework running 50+ A/B tests per quarter",
            "Built product health dashboard reducing churn by identifying at-risk users",
            "Led data analysis for feature launch reaching 500K+ users"
        ],
        "certifications": ["Product Analytics Certificate", "Amplitude/Mixpanel Certified", "Reforge Program"],
        "experience_benchmark": "2-4 years in product analytics or growth",
        "what_top_10_percent_have": [
            "Experience at high-growth startup (Series B+)",
            "Direct collaboration with product and engineering",
            "Designed and analyzed experiments end-to-end",
            "Funnel optimization with measurable conversion lift",
            "User segmentation for personalization"
        ]
    },
    "Financial Analyst": {
        "must_have_skills": ["Excel", "Financial Modeling", "Accounting", "PowerPoint", "SQL"],
        "differentiator_skills": ["Python", "Hyperion/Anaplan", "NetSuite", "Tableau", "VBA Macros"],
        "standout_projects": [
            "Built 3-statement financial model for Series C fundraise ($100M+)",
            "Created automated monthly close reporting reducing cycle by 3 days",
            "Developed unit economics model guiding $20M budget allocation"
        ],
        "certifications": ["CFA", "CPA", "Financial Modeling (FMVA)", "FP&A Certification"],
        "experience_benchmark": "3-5 years in FP&A, investment banking, or corporate finance",
        "what_top_10_percent_have": [
            "Investment banking or Big 4 background",
            "Built models used for board presentations",
            "Experience with IPO or M&A transactions",
            "Variance analysis presented to executives",
            "Budget ownership ($10M+)"
        ]
    },
    "Risk Analyst": {
        "must_have_skills": ["SQL", "Excel", "Risk Modeling", "Statistics", "Regulatory Knowledge"],
        "differentiator_skills": ["Python", "SAS", "Credit Scoring", "Basel III", "Stress Testing"],
        "standout_projects": [
            "Developed credit scorecard reducing default rate by 20%",
            "Built stress testing framework for Fed/OCC regulatory exams",
            "Created real-time fraud detection system with 95% precision"
        ],
        "certifications": ["FRM", "PRM", "CFA", "SAS Certified"],
        "experience_benchmark": "3-5 years in risk management at bank or fintech",
        "what_top_10_percent_have": [
            "Direct regulatory exam experience (Fed, OCC, CFPB)",
            "Built models approved by Model Risk Management",
            "Experience with CECL, Basel III, or Dodd-Frank",
            "Credit policy development",
            "Portfolio-level risk monitoring"
        ]
    },
    "Credit Analyst": {
        "must_have_skills": ["Financial Statement Analysis", "Excel", "Credit Risk", "Underwriting", "SQL"],
        "differentiator_skills": ["Python", "Credit Scoring Models", "LOS Systems", "Automated Decisioning"],
        "standout_projects": [
            "Underwrote $50M+ loan portfolio with <2% default rate",
            "Built automated credit decisioning rules reducing manual review by 60%",
            "Developed early warning system identifying at-risk accounts 30 days earlier"
        ],
        "certifications": ["Credit Risk Certification", "CFA", "Commercial Lending Certificate"],
        "experience_benchmark": "2-4 years in credit/underwriting at bank or lender",
        "what_top_10_percent_have": [
            "Portfolio performance tracking record",
            "Experience with automated decisioning systems",
            "Built or refined credit policy",
            "Workout/collections experience",
            "Multiple asset classes (consumer, SMB, commercial)"
        ]
    },
    "Operations Analyst": {
        "must_have_skills": ["Excel", "Process Mapping", "SQL", "Project Management", "Data Analysis"],
        "differentiator_skills": ["Python", "Six Sigma", "Automation Tools", "Salesforce", "Zendesk"],
        "standout_projects": [
            "Automated customer onboarding reducing processing time by 70%",
            "Built operations dashboard tracking SLAs across 5 teams",
            "Led process redesign saving $500K annually in operational costs"
        ],
        "certifications": ["Six Sigma Green/Black Belt", "PMP", "Lean Certification"],
        "experience_benchmark": "2-4 years in operations at fintech or financial services",
        "what_top_10_percent_have": [
            "Process automation experience",
            "Cross-functional project leadership",
            "Vendor management experience",
            "Compliance/audit coordination",
            "Quantified efficiency improvements"
        ]
    }
}

# Company-specific intelligence
COMPANY_BENCHMARKS = {
    "Stripe": {
        "culture": "Engineering-first, write great documentation, high bar",
        "interview_focus": ["System design", "SQL deep-dive", "Business cases"],
        "what_they_value": "Clear thinking, written communication, builder mentality",
        "standout_for_them": "Open source contributions, technical blog posts, payments domain knowledge"
    },
    "Plaid": {
        "culture": "Developer-focused, API obsessed, fintech infrastructure",
        "interview_focus": ["API design", "Data modeling", "Financial data knowledge"],
        "what_they_value": "Understanding of banking/financial data, API experience",
        "standout_for_them": "Banking integrations experience, developer tools background"
    },
    "Ramp": {
        "culture": "Speed, automation, expense management disruption",
        "interview_focus": ["SQL", "Product sense", "Growth mindset"],
        "what_they_value": "Moving fast, automation mindset, B2B SaaS experience",
        "standout_for_them": "Corporate card/expense domain, procurement experience"
    },
    "Chime": {
        "culture": "Consumer fintech, financial inclusion, member-first",
        "interview_focus": ["Consumer analytics", "A/B testing", "Growth"],
        "what_they_value": "Consumer empathy, growth experimentation, mobile-first",
        "standout_for_them": "Consumer banking experience, growth marketing analytics"
    },
    "Robinhood": {
        "culture": "Democratize finance, trading/brokerage, fast-paced",
        "interview_focus": ["Trading knowledge", "Real-time systems", "Compliance"],
        "what_they_value": "Capital markets knowledge, regulatory awareness",
        "standout_for_them": "Brokerage experience, Series 7/63, trading analytics"
    },
    "Intuit": {
        "culture": "Customer-obsessed, data-driven decisions, innovation at scale",
        "interview_focus": ["Customer empathy", "Data analysis", "Product sense", "A/B testing"],
        "what_they_value": "Small business understanding, experiment-driven mindset, customer impact",
        "standout_for_them": "QuickBooks/TurboTax domain knowledge, SMB experience, experimentation frameworks"
    },
    "SoFi": {
        "culture": "Member-first, full financial services, ambitious growth",
        "interview_focus": ["Financial products", "Cross-sell analytics", "Consumer lending"],
        "what_they_value": "Understanding of lending, investing, and banking products",
        "standout_for_them": "Multi-product fintech experience, member lifecycle analytics"
    },
    "Affirm": {
        "culture": "Honest finance, BNPL leader, engineering excellence",
        "interview_focus": ["Credit risk", "Machine learning", "Consumer behavior"],
        "what_they_value": "Understanding of credit, consumer lending, ML in production",
        "standout_for_them": "Credit modeling experience, underwriting analytics, merchant partnerships"
    },
    "Coinbase": {
        "culture": "Crypto-first, regulatory-forward, remote-first",
        "interview_focus": ["Crypto knowledge", "Compliance", "Security"],
        "what_they_value": "Blockchain understanding, regulatory awareness, security mindset",
        "standout_for_them": "Crypto/DeFi experience, trading analytics, AML/KYC knowledge"
    }
}

# ============================================
# 10. FIELD CONFIGURATION
# ============================================

# ============================================
# 9.5 APPLICATION TRACKER & NOTIFICATIONS
# ============================================
"""
TOPIC: AI for Good - Complete Job Seeker Journey

Features:
- Application tracking
- Email/Slack notifications
- Follow-up reminders
- Saved jobs export
"""

class ApplicationTracker:
    """
    Track job applications and generate follow-up reminders.
    """
    
    def __init__(self):
        self.applications = []
    
    def generate_tracker_template(self, jobs: List[Dict], ats_analyses: List[Dict]) -> str:
        """
        Generate a tracking spreadsheet template for the user.
        """
        output = """## 📋 Application Tracker Template

Copy this to a spreadsheet (Google Sheets/Excel) to track your applications:

| # | Company | Role | ATS Match | Status | Date Applied | Follow-up Date | Notes |
|---|---------|------|-----------|--------|--------------|----------------|-------|
"""
        for i, (job, ats) in enumerate(zip(jobs[:10], ats_analyses[:10]), 1):
            company = job.get('company', 'Unknown')
            title = job.get('title', 'Unknown')[:30]
            ats_score = ats.get('ats_score', 0)
            output += f"| {i} | {company} | {title} | {ats_score}% | ⏳ To Apply | | | |\n"
        
        return output
    
    def generate_followup_reminders(self, jobs: List[Dict]) -> str:
        """
        Generate follow-up email templates and reminder schedule.
        """
        output = """## ⏰ Follow-Up Reminder Schedule

**Best Practice:** Follow up 5-7 business days after applying if no response.

### 📧 Follow-Up Email Template:

```
Subject: Following Up - [Job Title] Application - [Your Name]

Dear [Hiring Manager/Recruiter],

I recently applied for the [Job Title] position at [Company] and wanted to 
express my continued interest in the role. 

With my background in [key skill 1] and [key skill 2], I believe I could 
make a meaningful contribution to your team, particularly in [specific area 
from job description].

I would welcome the opportunity to discuss how my experience aligns with 
your needs. Please let me know if you need any additional information.

Thank you for your time and consideration.

Best regards,
[Your Name]
[Phone] | [Email] | [LinkedIn]
```

### 📅 Suggested Follow-Up Schedule:

| Company | Role | Follow-Up Date | Action |
|---------|------|----------------|--------|
"""
        from datetime import datetime, timedelta
        today = datetime.now()
        
        for i, job in enumerate(jobs[:8], 1):
            followup_date = (today + timedelta(days=7*i)).strftime("%b %d")
            company = job.get('company', 'Unknown')
            title = job.get('title', 'Unknown')[:25]
            output += f"| {company} | {title} | {followup_date} | Send follow-up email |\n"
        
        return output


class NotificationManager:
    """
    Set up notifications via email, Slack, or Telegram.
    """
    
    @staticmethod
    def generate_email_summary(candidate: CandidateProfile, 
                               jobs: List[Dict], 
                               ats_analyses: List[Dict],
                               market_intel: MarketIntelligence) -> str:
        """
        Generate an email-ready summary to send to yourself.
        """
        output = """## 📧 Email Summary (Copy & Send to Yourself)

**Subject:** My Fintech Job Search Summary - [Date]

---

### 🎯 Top Job Matches

"""
        for i, (job, ats) in enumerate(zip(jobs[:5], ats_analyses[:5]), 1):
            output += f"""**{i}. {job.get('title')} at {job.get('company')}**
- ATS Match: {ats.get('ats_score', 0)}%
- Location: {job.get('location', 'N/A')}
- Salary: {job.get('salary_range', 'N/A')}
- Apply: {job.get('url', 'N/A')}

"""
        
        output += """### 📈 Market Insights

"""
        if market_intel.company_signals:
            for company, data in list(market_intel.company_signals.items())[:3]:
                output += f"- **{company}**: Momentum {data.get('momentum_score', 0)}/100\n"
        
        output += """
### ✅ Action Items This Week

1. [ ] Apply to top 3 matched jobs
2. [ ] Update resume with recommended keywords
3. [ ] Prepare for interviews using prep guide
4. [ ] Follow up on pending applications

---

*Generated by Fintech Career Intelligence Bot*
"""
        return output
    
    @staticmethod
    def generate_slack_webhook_message(jobs: List[Dict], ats_analyses: List[Dict]) -> str:
        """
        Generate a Slack-formatted message for webhook notifications.
        """
        output = """## 🔔 Slack/Discord Notification Setup

### Option 1: Slack Webhook

1. Go to: https://api.slack.com/messaging/webhooks
2. Create a webhook for your channel
3. Use this JSON payload:

```json
{
  "text": "🚀 New Fintech Job Matches!",
  "blocks": [
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*Top Job Matches:*"
      }
    },
"""
        for i, (job, ats) in enumerate(zip(jobs[:3], ats_analyses[:3]), 1):
            output += f"""    {{
      "type": "section", 
      "text": {{
        "type": "mrkdwn",
        "text": "{i}. *{job.get('title', 'N/A')}* at {job.get('company', 'N/A')}\\nATS: {ats.get('ats_score', 0)}% | <{job.get('url', '#')}|Apply>"
      }}
    }},
"""
        
        output += """  ]
}
```

### Option 2: Daily Email Digest

Set up a daily reminder:
1. Copy the "Email Summary" above
2. Schedule an email to yourself using Gmail's "Schedule Send"
3. Or use a service like Zapier to automate daily job alerts

### Option 3: Google Calendar Reminder

1. Create a recurring calendar event: "Check Job Matches"
2. Add the link to this tool in the event description
3. Set daily/weekly reminders
"""
        return output
    
    @staticmethod
    def generate_job_alert_services() -> str:
        """
        Generate list of job alert services to complement this tool.
        """
        return """## 🔔 Set Up Job Alerts (External Services)

Complement this tool with automated alerts:

### LinkedIn Job Alerts
1. Search for your target role on LinkedIn Jobs
2. Click "Set Alert" 
3. Choose Daily or Weekly digest

### Indeed Job Alerts  
1. Search on indeed.com
2. Enter your email at the bottom
3. Get daily matches

### Glassdoor Alerts
1. Set up at glassdoor.com/Job-Alerts
2. Filter by company rating, salary, etc.

### Company Career Pages
Set alerts directly on company websites:
- **Stripe:** stripe.com/jobs
- **Plaid:** plaid.com/careers
- **Ramp:** ramp.com/careers
- **Chime:** chime.com/careers
- **Robinhood:** robinhood.com/careers

### Pro Tip: Use Google Alerts
1. Go to google.com/alerts
2. Create alert: "[Company Name] hiring" OR "[Company Name] jobs"
3. Get notified when companies post new roles

---

*Run this tool weekly to get updated ATS scores and market intelligence!*
"""


class OfferComparator:
    """
    Compare multiple job offers.
    """
    
    @staticmethod
    def generate_offer_comparison_template() -> str:
        """
        Generate a template for comparing job offers.
        """
        return """## 💰 Offer Comparison Calculator

When you receive offers, use this template to compare:

| Factor | Weight | Offer 1 | Offer 2 | Offer 3 |
|--------|--------|---------|---------|---------|
| **Base Salary** | 30% | $____ | $____ | $____ |
| **Equity/RSUs** | 20% | $____ | $____ | $____ |
| **Bonus** | 10% | $____ | $____ | $____ |
| **401k Match** | 5% | ____% | ____% | ____% |
| **Healthcare** | 10% | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **PTO Days** | 5% | ____ | ____ | ____ |
| **Remote Policy** | 10% | ____ | ____ | ____ |
| **Growth Potential** | 10% | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### Total Compensation Calculator

```
Total Comp = Base + (Equity ÷ 4) + Bonus + (401k Match × Base × 0.06)

Offer 1: $______
Offer 2: $______
Offer 3: $______
```

### Non-Financial Factors

| Factor | Offer 1 | Offer 2 | Offer 3 |
|--------|---------|---------|---------|
| Commute Time | | | |
| Team/Manager | | | |
| Tech Stack | | | |
| Company Stage | | | |
| Learning Potential | | | |
| Work-Life Balance | | | |

### 🤖 AI Negotiation Tips

1. **Always negotiate** - 84% of employers expect it
2. **Get competing offers** - Leverage is key
3. **Negotiate beyond salary** - Title, equity, PTO, start date
4. **Use market data** - "Based on market rates for this role..."
5. **Be specific** - "I'm looking for $X to accept"
"""

# ============================================
# 11. INDUSTRY & FIELD CONFIGURATION
# ============================================

INDUSTRIES = {
    "FAANG / Big Tech": {
        "news_query": "Google hiring OR Meta jobs OR Amazon careers OR Apple hiring OR Netflix jobs OR Microsoft hiring OR tech layoffs OR big tech expansion",
        "job_query": "",  # Empty - just use role name
        "companies": [
            "Google", "Alphabet", "Meta", "Facebook", "Amazon", "Apple", "Netflix", 
            "Microsoft", "Nvidia", "Tesla", "Salesforce", "Adobe", "Oracle", "IBM",
            "Intel", "Cisco", "VMware", "Snowflake", "Databricks", "Palantir",
            "Uber", "Lyft", "Airbnb", "DoorDash", "Instacart", "Snap", "Twitter", "X",
            "LinkedIn", "Pinterest", "Reddit", "Discord", "Slack", "Zoom", "Dropbox"
        ]
    },
    "Fintech": {
        "news_query": "fintech startup funding OR fintech hiring OR fintech expansion OR digital banking",
        "job_query": "",  # Empty - just use role name
        "companies": [
            "Stripe", "Square", "Block", "PayPal", "Adyen", "Checkout", "Marqeta", "Toast",
            "Plaid", "Chime", "Robinhood", "SoFi", "Revolut", "N26", "Monzo", "Varo",
            "Affirm", "Klarna", "Upstart", "LendingClub", "Brex", "Ramp", "Mercury",
            "Coinbase", "Kraken", "Gemini", "Circle", "Ripple", "Chainalysis",
            "Gusto", "Rippling", "Deel", "Bill", "Melio", "Intuit", "ADP"
        ]
    },
    "Healthcare / Healthtech": {
        "news_query": "healthtech startup OR digital health funding OR healthcare AI OR telemedicine hiring",
        "job_query": "healthcare",
        "companies": [
            "Epic Systems", "Cerner", "Veeva", "IQVIA", "Optum", "CVS Health",
            "Teladoc", "Amwell", "Hims", "Ro", "Carbon Health", "One Medical",
            "Oscar Health", "Clover Health", "Bright Health", "Devoted Health",
            "Tempus", "Flatiron Health", "Butterfly Network", "Viz.ai",
            "Moderna", "Illumina", "23andMe", "Color Health"
        ]
    },
    "Consulting / Professional Services": {
        "news_query": "McKinsey OR Bain OR BCG OR Deloitte hiring OR consulting jobs OR management consulting",
        "job_query": "",  # Empty - just use role name
        "companies": [
            "McKinsey", "Bain", "BCG", "Boston Consulting Group", "Deloitte", "PwC", 
            "EY", "Ernst Young", "KPMG", "Accenture", "Booz Allen", "Oliver Wyman",
            "Roland Berger", "Strategy&", "LEK", "Kearney", "ZS Associates"
        ]
    },
    "E-Commerce / Retail Tech": {
        "news_query": "ecommerce startup OR retail tech OR D2C brand funding OR shopify",
        "job_query": "ecommerce",
        "companies": [
            "Amazon", "Shopify", "Etsy", "eBay", "Walmart", "Target", "Wayfair",
            "Chewy", "Instacart", "DoorDash", "Gopuff",
            "Warby Parker", "Allbirds", "Glossier", "Stitch Fix", "ThredUp"
        ]
    },
    "AI / Machine Learning": {
        "news_query": "AI startup funding OR machine learning hiring OR artificial intelligence company OR LLM startup",
        "job_query": "",  # Empty - just use role name for ML/AI roles
        "companies": [
            "OpenAI", "Anthropic", "Google DeepMind", "Cohere", "Hugging Face",
            "Scale AI", "Databricks", "DataRobot", "H2O.ai", "Weights & Biases",
            "Anyscale", "Modal", "Replicate", "Midjourney", "Stability AI", "Runway",
            "Jasper", "Writer", "Inflection", "Character.ai", "Perplexity"
        ]
    },
    "Startups (YC / General)": {
        "news_query": "startup funding OR series A OR YC startup OR venture capital",
        "job_query": "startup",
        "companies": [
            "Notion", "Figma", "Canva", "Airtable", "Coda", "Linear", "Vercel",
            "Retool", "Webflow", "Loom", "Calendly", "Miro", "ClickUp", "Asana"
        ]
    }
}

# Role configurations per industry - Use common job titles that return results
ROLES_BY_INDUSTRY = {
    "FAANG / Big Tech": [
        "Software Engineer", "Data Scientist", "Data Analyst", "Product Manager",
        "Machine Learning Engineer", "DevOps Engineer", "Frontend Developer",
        "Backend Developer", "Full Stack Developer", "UX Designer"
    ],
    "Fintech": [
        "Data Analyst", "Data Scientist", "Software Engineer", "Financial Analyst",
        "Risk Analyst", "Product Manager", "Business Analyst", "Compliance Analyst",
        "Quantitative Analyst", "Operations Analyst"
    ],
    "Healthcare / Healthtech": [
        "Data Analyst", "Data Scientist", "Software Engineer", "Product Manager",
        "Business Analyst", "Clinical Data Analyst", "Research Scientist", "Project Manager"
    ],
    "Consulting / Professional Services": [
        "Consultant", "Business Analyst", "Data Analyst", "Strategy Consultant",
        "Management Consultant", "Project Manager", "Financial Analyst"
    ],
    "E-Commerce / Retail Tech": [
        "Data Analyst", "Product Manager", "Software Engineer", "Marketing Analyst",
        "Business Analyst", "Supply Chain Analyst", "Category Manager", "Data Scientist"
    ],
    "AI / Machine Learning": [
        "Machine Learning Engineer", "Data Scientist", "Software Engineer", "Research Scientist",
        "AI Engineer", "Data Engineer", "MLOps Engineer", "NLP Engineer"
    ],
    "Startups (YC / General)": [
        "Software Engineer", "Data Analyst", "Product Manager", "Full Stack Developer",
        "Frontend Developer", "Backend Developer", "Data Scientist", "DevOps Engineer"
    ]
}

FINTECH_FIELDS = {
    "All Fintech": {
        "news_query": "fintech startup funding OR fintech hiring OR fintech expansion",
        "job_query": "fintech"
    },
    "Payments": {
        "news_query": "payments fintech OR digital payments startup OR payment processing funding",
        "job_query": "payments fintech"
    },
    "Lending": {
        "news_query": "fintech lending OR digital lending startup OR loan fintech funding",
        "job_query": "lending fintech"
    },
    "Neobanking": {
        "news_query": "neobank OR digital bank startup OR challenger bank funding",
        "job_query": "neobank digital bank"
    },
    "BNPL": {
        "news_query": "buy now pay later OR BNPL fintech OR installment payments startup",
        "job_query": "BNPL buy now pay later"
    },
    "Wealth Management": {
        "news_query": "wealthtech OR robo advisor OR digital wealth management funding",
        "job_query": "wealthtech wealth management fintech"
    },
    "Crypto/Blockchain": {
        "news_query": "crypto startup funding OR blockchain fintech OR cryptocurrency exchange",
        "job_query": "crypto blockchain fintech"
    }
}

SKILL_CATEGORIES = {
    "technical": [
        "python", "sql", "r", "tableau", "power bi", "excel", 
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
        "aws", "azure", "gcp", "spark", "hadoop", "airflow",
        "postgresql", "mysql", "mongodb", "snowflake", "dbt"
    ],
    "analytics": [
        "data analysis", "statistical modeling", "machine learning",
        "predictive modeling", "regression", "classification", "clustering",
        "a/b testing", "hypothesis testing", "forecasting", "time series"
    ],
    "business": [
        "strategy", "business intelligence", "kpi", "metrics",
        "stakeholder management", "presentations", "reporting",
        "project management", "product management", "operations"
    ],
    "finance": [
        "financial modeling", "valuation", "credit risk", "underwriting",
        "loan origination", "collections", "compliance", "audit",
        "budgeting", "unit economics", "p&l", "financial analysis"
    ]
}

SIGNAL_KEYWORDS = {
    "funding": ["raised", "funding", "series a", "series b", "series c", "seed", 
                "investment", "valuation", "million", "billion"],
    "expansion": ["expands", "expansion", "hiring", "opens office", "growth", 
                  "launch", "scaling", "hires", "employees"],
    "product": ["launched", "announces", "new product", "feature", "platform", 
                "introduces", "rolls out"],
    "partnership": ["partners", "partnership", "collaboration", "integration", 
                    "deal", "agreement"],
    "acquisition": ["acquires", "acquisition", "bought", "merger", "merged"]
}


# ============================================
# 10. DATA FETCHING FUNCTIONS
# ============================================
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from uploaded PDF file"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text


def extract_skills(text: str) -> Dict[str, List[str]]:
    """Extract skills from resume text"""
    # Ensure text is a string
    if not text or not isinstance(text, str):
        return {category: [] for category in SKILL_CATEGORIES}
    
    text_lower = text.lower()
    found_skills = {category: [] for category in SKILL_CATEGORIES}
    
    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if skill.lower() in text_lower:
                found_skills[category].append(skill)
    
    return found_skills


def extract_experience_years(text: str) -> int:
    """Estimate years of experience from resume"""
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
        r'experience\s*(?:of)?\s*(\d+)\+?\s*years?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    
    year_pattern = r'(20\d{2}|19\d{2})\s*[-–]\s*(20\d{2}|19\d{2}|present|current)'
    matches = re.findall(year_pattern, text.lower())
    
    if matches:
        total_years = 0
        for start, end in matches:
            start_year = int(start)
            end_year = 2025 if end in ['present', 'current'] else int(end)
            total_years += (end_year - start_year)
        return min(total_years, 30)
    
    return 0


def parse_resume(pdf_path: str) -> CandidateProfile:
    """Parse resume into structured CandidateProfile"""
    text = extract_text_from_pdf(pdf_path)
    skills = extract_skills(text)
    
    return CandidateProfile(
        raw_text=text,
        skills=skills,
        years_experience=extract_experience_years(text),
        education=[],  # Could be extracted with more parsing
        previous_companies=[],  # Could be extracted with NER
        career_trajectory=""  # Will be filled by LLM
    )


def fetch_news_from_api(query: str, days_back: int = 7) -> Tuple[List[Dict], str]:
    """Fetch real news from NewsAPI"""
    if not NEWS_API_KEY:
        return [], "❌ NEWS_API_KEY not set"
    
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 50,
        "apiKey": NEWS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if data.get("status") == "ok":
            return data.get("articles", []), ""
        return [], f"❌ NewsAPI Error: {data.get('message', 'Unknown')}"
    except Exception as e:
        return [], f"❌ NewsAPI request failed: {str(e)}"


def fetch_jobs_from_adzuna(role: str, field_query: str, location: str = "") -> Tuple[List[Dict], str]:
    """
    Fetch real jobs from Adzuna API with smart fallback logic.
    
    Strategy:
    1. Try specific query (role + industry)
    2. If no results, try just role
    3. If still no results, try broader search
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return [], "❌ Adzuna keys not set"
    
    url = f"https://api.adzuna.com/v1/api/jobs/us/search/1"
    
    # Try multiple queries in order of specificity
    queries_to_try = [
        role,  # Start with just the role (most reliable)
        f"{role} {field_query}" if field_query else role,  # Then add industry
    ]
    
    for query in queries_to_try:
        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_APP_KEY,
            "what": query,
            "results_per_page": 30
        }
        
        # Add location filter if provided
        if location and location.strip():
            params["where"] = location.strip()
        
        try:
            print(f"    🔍 Trying query: '{query}' {'in ' + location if location else ''}")
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            if "error" in data:
                print(f"    ⚠️ Query '{query}' returned error: {data.get('error')}")
                continue
            
            results = data.get("results", [])
            
            if not results:
                print(f"    ⚠️ Query '{query}' returned 0 results, trying next...")
                continue
            
            print(f"    ✅ Found {len(results)} jobs with query: '{query}'")
            
            jobs = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                    
                min_sal = result.get("salary_min")
                max_sal = result.get("salary_max")
                
                if min_sal and max_sal:
                    salary = f"${int(min_sal):,} - ${int(max_sal):,}"
                elif min_sal:
                    salary = f"${int(min_sal):,}+"
                else:
                    salary = "Not disclosed"
                
                # Safely extract company name
                company_data = result.get("company")
                if isinstance(company_data, dict):
                    company_name = company_data.get("display_name", "Unknown")
                else:
                    company_name = "Unknown"
                
                # Safely extract location
                location_data = result.get("location")
                if isinstance(location_data, dict):
                    location_name = location_data.get("display_name", "Not specified")
                else:
                    location_name = "Not specified"
                
                # Ensure description is a string
                description = result.get("description", "")
                if not isinstance(description, str):
                    description = str(description) if description else ""
                
                jobs.append({
                    "title": str(result.get("title", "No title") or "No title"),
                    "company": str(company_name or "Unknown"),
                    "location": str(location_name or "Not specified"),
                    "url": str(result.get("redirect_url", "") or ""),
                    "description": description[:500],
                    "salary_range": salary,
                    "posted": str(result.get("created", "")[:10] if result.get("created") else "Unknown"),
                    "requirements": [],
                    "query_used": query  # Track which query found this job
                })
            
            return jobs, ""
            
        except Exception as e:
            print(f"    ❌ Query '{query}' failed: {str(e)}")
            continue
    
    # If all queries failed
    return [], f"❌ No jobs found. Tried queries: {queries_to_try}"


def process_news_for_intelligence(articles: List[Dict], industry_companies: List[str] = None) -> MarketIntelligence:
    """Process news articles into structured market intelligence"""
    
    # Use industry-specific companies if provided, otherwise use default list
    if industry_companies:
        known_companies = industry_companies
    else:
        # Default expanded list of companies for fallback
        known_companies = [
            # FAANG / Big Tech
            "Google", "Alphabet", "Meta", "Facebook", "Amazon", "Apple", "Netflix", 
            "Microsoft", "Nvidia", "Tesla", "Salesforce", "Adobe", "Oracle",
            "Uber", "Lyft", "Airbnb", "DoorDash", "Snap", "LinkedIn", "Zoom",
            # Fintech
            "Stripe", "Square", "Block", "PayPal", "Adyen", "Marqeta", "Toast",
            "Plaid", "Chime", "Robinhood", "SoFi", "Revolut", "Mercury",
            "Affirm", "Klarna", "Upstart", "Brex", "Ramp", "Coinbase",
            "Gusto", "Rippling", "Deel", "Intuit", "ADP",
            # AI
            "OpenAI", "Anthropic", "Cohere", "Hugging Face", "Scale AI", "Databricks",
            # Healthcare
            "Epic", "Teladoc", "Oscar Health", "Tempus", "Moderna",
            # Consulting
            "McKinsey", "Bain", "BCG", "Deloitte", "Accenture"
        ]
    
    company_signals = {}
    processed_news = []
    funding_events = []
    
    # Handle case where articles might be None or not a list
    if not articles or not isinstance(articles, list):
        return MarketIntelligence(
            company_signals={},
            recent_news=[],
            funding_events=[],
            hiring_trends={},
            timestamp=datetime.now()
        )
    
    for article in articles:
        if not isinstance(article, dict):
            continue
        # Ensure title and description are strings (not bool/None)
        title = str(article.get('title') or '')
        description = str(article.get('description') or '')
        full_text = f"{title} {description}"
        
        # Find company
        article_company = None
        for company in known_companies:
            if company.lower() in full_text.lower():
                article_company = company
                break
        
        # Classify signal
        signal = "general"
        for signal_type, keywords in SIGNAL_KEYWORDS.items():
            if any(kw in full_text.lower() for kw in keywords):
                signal = signal_type
                break
        
        news_item = {
            "title": title,
            "company": article_company,
            "signal": signal,
            "url": article.get('url', ''),
            "date": article.get('publishedAt', '')[:10] if article.get('publishedAt') else ''
        }
        processed_news.append(news_item)
        
        if signal == "funding":
            funding_events.append(news_item)
        
        # Aggregate company signals
        if article_company:
            if article_company not in company_signals:
                company_signals[article_company] = {
                    'signals': [],
                    'news_count': 0,
                    'is_hiring': False,
                    'latest_news': [],
                    'momentum_score': 0
                }
            
            company_signals[article_company]['signals'].append(signal)
            company_signals[article_company]['news_count'] += 1
            company_signals[article_company]['is_hiring'] = (
                company_signals[article_company]['is_hiring'] or 
                'hiring' in full_text.lower()
            )
            
            if len(company_signals[article_company]['latest_news']) < 3:
                company_signals[article_company]['latest_news'].append(news_item)
    
    # Calculate momentum scores
    score_weights = {"funding": 30, "expansion": 25, "product": 20, 
                     "partnership": 15, "acquisition": 20, "general": 5}
    
    for company in company_signals:
        signals = company_signals[company]['signals']
        score = sum(score_weights.get(s, 5) for s in signals)
        company_signals[company]['momentum_score'] = min(score, 100)
    
    return MarketIntelligence(
        company_signals=company_signals,
        recent_news=processed_news,
        funding_events=funding_events,
        hiring_trends={},
        timestamp=datetime.now()
    )


# ============================================
# 11. MAIN PROCESSING FUNCTION
# ============================================
def process_ai_career_intelligence(resume_file, target_industry, target_role, target_location,
                                   generate_cover_letter, generate_interview_prep,
                                   job_selection):
    """
    Main processing function with full AI capabilities.
    Now supports multiple industries!
    """
    import time
    start_time = time.time()
    
    # Validate inputs
    if resume_file is None:
        return "❌ **Please upload your resume (PDF format)**", "", "", "", "", "", "", ""
    
    # Convert job_selection to integer index (0-based)
    job_index = int(job_selection) - 1
    
    print("\n" + "="*60)
    print("🚀 STARTING ANALYSIS")
    print("="*60)
    
    # Initialize AI components
    print("⚙️  [1/7] Initializing AI components...")
    llm = LLMClient()
    rag = RAGEngine(llm)
    agent = CareerAgent(llm, rag)
    content_gen = ContentGenerator(rag)
    responsible_ai = ResponsibleAIGuard()
    
    # Get industry configuration
    industry_config = INDUSTRIES.get(target_industry, INDUSTRIES["Fintech"])
    
    # Step 1: Parse resume
    print(f"📄 [2/7] Parsing resume for {target_industry} analysis...")
    candidate = parse_resume(resume_file.name)
    
    if not candidate.raw_text:
        return "❌ **Could not read resume.**", "", "", "", "", "", "", ""
    
    print(f"    ✓ Found {candidate.years_experience} years experience, {sum(len(v) for v in candidate.skills.values())} skills")
    
    # Step 2: Fetch market data
    print(f"📰 [3/7] Fetching {target_industry} market intelligence...")
    articles, news_error = fetch_news_from_api(industry_config["news_query"])
    
    if news_error:
        print(f"    ⚠️ News fetch issue: {news_error}")
        market_intel = MarketIntelligence({}, [], [], {}, datetime.now())
    else:
        # Pass industry-specific companies for better detection
        market_intel = process_news_for_intelligence(articles, industry_config.get("companies", []))
        print(f"    ✓ Found {len(articles)} articles, {len(market_intel.company_signals)} company signals")
    
    # Step 3: Fetch jobs - Use JUST the role for best results, industry query as fallback
    location_filter = target_location if target_location and target_location != "Any Location" else ""
    print(f"💼 [4/7] Fetching {target_role} opportunities...")
    jobs, job_error = fetch_jobs_from_adzuna(target_role, industry_config.get('job_query', ''), location_filter)
    
    if job_error:
        print(f"    ⚠️ {job_error}")
    else:
        print(f"    ✓ Found {len(jobs)} job listings")
    
    # Step 4: Agent analysis of opportunities
    print(f"🤖 [5/7] AI Agent analyzing {min(len(jobs), 10)} opportunities...")
    analyzed_opportunities = []
    for i, job in enumerate(jobs[:10], 1):
        print(f"    Analyzing job {i}/{min(len(jobs), 10)}: {job.get('title', 'Unknown')[:40]}...")
        analysis = agent.analyze_opportunity(job, candidate, market_intel)
        # Add ATS analysis
        ats_analysis = content_gen.generate_ats_analysis(job, candidate, market_intel)
        analysis['ats_analysis'] = ats_analysis
        analyzed_opportunities.append(analysis)
    
    # Sort by ATS score (primary) and skill overlap (secondary)
    analyzed_opportunities.sort(
        key=lambda x: (x['ats_analysis']['ats_score'], x['perception']['skill_overlap']), 
        reverse=True
    )
    
    print(f"📊 [6/7] Building analysis outputs...")
    
    # ============================================
    # BUILD OUTPUTS
    # ============================================
    
    # Output 1: AI-Enhanced Resume Analysis - MENTOR-LEVEL INSIGHTS
    skills = candidate.skills
    total_skills = sum(len(v) for v in skills.values())
    
    # Calculate role fit scores with RATIONALE
    role_fit_data = {
        "Data Analyst": {
            "required": ["sql", "python", "excel", "tableau", "power bi", "data analysis"],
            "strong_signal": ["sql", "tableau", "stakeholder"],
            "weak_signal": ["mlops", "production"],
        },
        "Data Scientist": {
            "required": ["python", "machine learning", "statistics", "pandas", "scikit-learn", "tensorflow"],
            "strong_signal": ["machine learning", "tensorflow", "modeling"],
            "weak_signal": ["mlops", "deployment", "production"],
        },
        "Product Analyst": {
            "required": ["sql", "a/b testing", "analytics", "metrics", "excel", "experimentation"],
            "strong_signal": ["a/b testing", "experimentation", "product"],
            "weak_signal": ["hypothesis", "causal"],
        },
        "Software Engineer": {
            "required": ["python", "java", "javascript", "git", "aws", "docker"],
            "strong_signal": ["docker", "kubernetes", "system design"],
            "weak_signal": ["analytics", "bi"],
        },
        "Machine Learning Engineer": {
            "required": ["python", "tensorflow", "pytorch", "mlops", "aws", "docker"],
            "strong_signal": ["mlops", "deployment", "production"],
            "weak_signal": ["excel", "tableau"],
        },
        "Strategy Consultant": {
            "required": ["excel", "powerpoint", "strategy", "financial modeling", "presentations"],
            "strong_signal": ["strategy", "presentations", "stakeholder"],
            "weak_signal": ["python", "machine learning"],
        }
    }
    
    all_candidate_skills = []
    all_candidate_skills_str = ""
    for skill_list in skills.values():
        all_candidate_skills.extend([s.lower() for s in skill_list])
    all_candidate_skills_str = ' '.join(all_candidate_skills)
    
    role_fit_scores = {}
    role_rationales = {}
    
    for role, data in role_fit_data.items():
        matched = sum(1 for r in data["required"] if any(r in s for s in all_candidate_skills))
        score = min(round(matched / len(data["required"]) * 100), 100)
        role_fit_scores[role] = score
        
        # Generate rationale
        strengths = [s for s in data["strong_signal"] if any(s in sk for sk in all_candidate_skills)]
        weaknesses = [s for s in data["weak_signal"] if not any(s in sk for sk in all_candidate_skills)]
        
        if strengths:
            rationale = f"Strong {', '.join(strengths[:2])}"
        else:
            rationale = "Baseline skills present"
        
        if weaknesses and score < 100:
            rationale += f"; lighter {weaknesses[0]} signals"
        
        role_rationales[role] = rationale
    
    sorted_roles = sorted(role_fit_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Estimate salary with CONTEXT
    base_salary = 70000
    exp_bonus = min(candidate.years_experience * 8000, 80000)
    skill_bonus = min(total_skills * 1500, 30000)
    estimated_salary_low = base_salary + exp_bonus + skill_bonus - 15000
    estimated_salary_high = base_salary + exp_bonus + skill_bonus + 20000
    
    # Determine candidate tier
    if candidate.years_experience >= 8 and total_skills >= 20:
        tier = "🏆 **Top 10%**"
        tier_level = "Senior/Lead"
        overqualified_for = "Junior / entry-level analyst roles"
        competitive_for = "Senior IC roles, Staff positions"
        stretch_into = "Lead / Manager / Principal roles"
        hire_verdict = "✅ **Strong Hire** for senior analytics roles"
        hire_condition = "Clarify scope and business impact of recent projects"
    elif candidate.years_experience >= 5 and total_skills >= 15:
        tier = "🥈 **Top 25%**"
        tier_level = "Mid-Senior"
        overqualified_for = "Entry-level roles"
        competitive_for = "Mid-level to Senior IC roles"
        stretch_into = "Senior / Lead roles"
        hire_verdict = "✅ **Hire** with development potential"
        hire_condition = "Show ownership and measurable outcomes"
    elif candidate.years_experience >= 2 and total_skills >= 10:
        tier = "🥉 **Top 50%**"
        tier_level = "Mid-level"
        overqualified_for = "Internships"
        competitive_for = "Mid-level IC roles"
        stretch_into = "Senior IC roles"
        hire_verdict = "🟡 **Conditional Hire** — needs stronger signal"
        hire_condition = "Add quantified impact and modern tools"
    else:
        tier = "📈 **Building**"
        tier_level = "Entry/Junior"
        overqualified_for = "N/A"
        competitive_for = "Entry-level, Associate roles"
        stretch_into = "Mid-level roles with strong portfolio"
        hire_verdict = "🟡 **Potential** — focus on portfolio"
        hire_condition = "Build 2-3 strong portfolio projects"
    
    # Find high-momentum company for recommendation
    recommended_company = None
    if market_intel.company_signals:
        sorted_companies = sorted(
            market_intel.company_signals.items(),
            key=lambda x: x[1].get('momentum_score', 0),
            reverse=True
        )
        if sorted_companies:
            recommended_company = sorted_companies[0][0]
            recommended_momentum = sorted_companies[0][1].get('momentum_score', 0)
    
    # Detect resume weaknesses (Why You Might Be Getting Rejected)
    rejection_reasons = []
    
    # Check for outcome language
    outcome_words = ["impact", "drove", "increased", "decreased", "saved", "generated", "improved"]
    has_outcomes = any(w in all_candidate_skills_str for w in outcome_words)
    if not has_outcomes:
        rejection_reasons.append("Resume emphasizes **tools more than business outcomes** — add metrics like 'Drove $X revenue' or 'Reduced Y by Z%'")
    
    # Check for ownership language
    ownership_words = ["led", "owned", "managed", "built", "designed", "architected"]
    has_ownership = any(w in all_candidate_skills_str for w in ownership_words)
    if not has_ownership:
        rejection_reasons.append("Lacks **ownership language** — add 'Led', 'Owned', 'Drove' to show initiative")
    
    # Check for scale indicators
    scale_words = ["scale", "million", "thousand", "enterprise", "production"]
    has_scale = any(w in all_candidate_skills_str for w in scale_words)
    if not has_scale:
        rejection_reasons.append("**Scope is unclear** — add team size, data scale, or user impact numbers")
    
    # Check for modern tools
    modern_tools = ["dbt", "airflow", "spark", "snowflake", "databricks"]
    has_modern = any(t in all_candidate_skills_str for t in modern_tools)
    if not has_modern:
        rejection_reasons.append("Missing **modern data stack** signals — consider adding dbt, Airflow, or Snowflake")
    
    resume_output = f"""## 📄 AI-Enhanced Resume Analysis for {target_industry}

---

### 🎯 Your Candidate Profile

| Metric | Your Score |
|--------|------------|
| **Experience Level** | {candidate.years_experience}+ years ({tier_level}) |
| **Skills Detected** | {total_skills} skills |
| **Candidate Tier** | {tier} |

---

### 💰 Estimated Salary Range

**${estimated_salary_low:,} — ${estimated_salary_high:,}** per year

*Assumes: US-based, {tier_level} IC roles at mid-to-large companies, base salary only (excludes equity/bonus).*

---

### 🎯 Best-Fit Roles (with Rationale)

| Role | Fit | Why This Score |
|------|-----|----------------|
| **{sorted_roles[0][0]}** | {sorted_roles[0][1]}% ✅ | {role_rationales[sorted_roles[0][0]]} |
| **{sorted_roles[1][0]}** | {sorted_roles[1][1]}% | {role_rationales[sorted_roles[1][0]]} |
| **{sorted_roles[2][0]}** | {sorted_roles[2][1]}% | {role_rationales[sorted_roles[2][0]]} |

"""
    
    # Add High-Signal Opportunity
    if recommended_company:
        resume_output += f"""---

### 🔥 High-Signal Opportunity (Based on Market Data)

**{recommended_company}** — Momentum Score: {recommended_momentum}/100

This company is actively growing based on recent news signals. Your background in {', '.join(skills['technical'][:3]) if skills['technical'] else 'data'} aligns with their typical hiring profile.

"""
    
    # Add Resume-to-Job Delta View
    target_role_key = sorted_roles[0][0]  # Use best-fit role
    resume_output += f"""---

### 📊 Gap Analysis: You vs. {target_role_key} (Senior)

| Skill Area | You | Market Expectation | Gap |
|------------|-----|-------------------|-----|
| **SQL/Analytics** | {"✅ Strong" if any("sql" in s for s in all_candidate_skills) else "⚠️ Weak"} | Strong | {"—" if any("sql" in s for s in all_candidate_skills) else "🔴"} |
| **Python** | {"✅ Strong" if any("python" in s for s in all_candidate_skills) else "⚠️ Weak"} | Strong | {"—" if any("python" in s for s in all_candidate_skills) else "🔴"} |
| **Business Impact** | {"✅ Strong" if has_outcomes else "⚠️ Medium"} | Strong | {"—" if has_outcomes else "🟡"} |
| **Modern Data Stack** | {"✅ Strong" if has_modern else "⚠️ Weak"} | Medium-Strong | {"—" if has_modern else "🟡"} |
| **Ownership/Leadership** | {"✅ Strong" if has_ownership else "⚠️ Weak"} | Strong for Senior | {"—" if has_ownership else "🔴"} |

**Top 2 upgrades to move from {sorted_roles[0][1]}% → 95% fit:**
1. {"Add quantified business impact (revenue, cost savings, efficiency gains)" if not has_outcomes else "Add modern data stack tools (dbt, Airflow)"}
2. {"Use ownership language: 'Led', 'Owned', 'Drove'" if not has_ownership else "Clarify scope: team size, data scale"}

---

### ❌ Why You Might Be Getting Rejected (Simulated Recruiter View)

"""
    
    if rejection_reasons:
        for reason in rejection_reasons[:4]:
            resume_output += f"- {reason}\n"
    else:
        resume_output += "✅ Your resume has strong signals — focus on targeting the right companies.\n"
    
    resume_output += f"""
---

### 🧭 Confidence Calibration

| Level | Roles |
|-------|-------|
| **You are OVERQUALIFIED for:** | {overqualified_for} |
| **You are COMPETITIVE for:** | {competitive_for} |
| **You should STRETCH into:** | {stretch_into} |

*Don't under-apply. Your experience qualifies you for more than you think.*

---

### 🗣️ How Recruiters Interpret Your Resume

| What You Wrote | What Recruiters Think |
|----------------|----------------------|
| "{candidate.years_experience}+ years experience" | Expect **leadership & ownership** — are you showing it? |
| "Python, SQL" | **Baseline**, not differentiator — everyone has this |
| "Machine Learning" | Do you **ship models** or just experiment? |
| "Data Analysis" | **Generic** — what decisions did you drive? |

---

### 🎯 Would I Hire You?

{hire_verdict}

**One condition:** {hire_condition}

---

### 📋 Skills Inventory

**Technical ({len(skills['technical'])}):** {', '.join(skills['technical']) if skills['technical'] else 'None detected'}

**Analytics ({len(skills['analytics'])}):** {', '.join(skills['analytics']) if skills['analytics'] else 'None detected'}

**Business ({len(skills['business'])}):** {', '.join(skills['business']) if skills['business'] else 'None detected'}

**Finance ({len(skills['finance'])}):** {', '.join(skills['finance']) if skills['finance'] else 'None detected'}

---

### 🤖 AI Career Trajectory Analysis
"""
    
    # Generate AI career insight
    context = rag.retrieve_context("career analysis", market_intel, candidate)
    career_insight = rag.augmented_generate(
        f"In 2-3 sentences, give specific, actionable advice for this candidate targeting {target_role} in {target_industry}. They have {candidate.years_experience} years experience with skills in {', '.join(skills['technical'][:5])}. Focus on: 1) Their strongest selling point, 2) One specific company that would value their background, 3) One concrete next step.",
        context,
        "career_advice"
    )
    resume_output += career_insight
    resume_output = responsible_ai.add_transparency_disclaimer(resume_output, "analysis")
    
    # Output 2: Market Intelligence - Make it actionable!
    market_output = f"## 📈 AI-Powered Market Intelligence: {target_industry}\n\n"
    
    # Explain WHAT this is and WHY it matters
    market_output += """### 💡 What is Market Intelligence?

**Purpose:** Identifies which companies are **growing, hiring, and have momentum** — so you can prioritize applications to companies most likely to hire.

**How it helps you:**
- 🟢 **High momentum (60+)** = Company is growing fast, APPLY NOW
- 🟡 **Medium momentum (40-59)** = Worth watching
- 🔴 **Low momentum (<40)** = Less urgent, may be slower to hire

---

"""
    
    if market_intel.company_signals:
        market_output += f"### 📊 Company Momentum Scores\n"
        market_output += f"*Based on {len(articles)} recent news articles*\n\n"
        market_output += "| Company | Momentum | Signals | What This Means |\n"
        market_output += "|---------|----------|---------|------------------|\n"
        
        sorted_companies = sorted(
            market_intel.company_signals.items(),
            key=lambda x: x[1]['momentum_score'],
            reverse=True
        )
        
        for company, data in sorted_companies[:10]:
            momentum = data['momentum_score']
            if momentum >= 60:
                emoji = "🟢"
                meaning = "Hot! Prioritize applications"
            elif momentum >= 40:
                emoji = "🟡"
                meaning = "Active, worth applying"
            else:
                emoji = "🔴"
                meaning = "Quiet recently"
            
            hiring = "✅ Hiring!" if data['is_hiring'] else "—"
            signals = ', '.join(set(data['signals']))[:30]
            market_output += f"| {company} | {emoji} {momentum}/100 | {signals} | {meaning} |\n"
        
        # Connect to job matches - DEDUPLICATE companies
        market_output += "\n### 🔗 How This Connects to Your Jobs\n\n"
        if analyzed_opportunities:
            seen_companies = set()  # Track companies we've already shown
            for opp in analyzed_opportunities[:8]:
                job_company = opp['job'].get('company', '')
                
                # Skip if we've already shown this company
                if job_company in seen_companies:
                    continue
                seen_companies.add(job_company)
                
                momentum = opp['perception']['company_momentum']
                ats = opp['ats_analysis']['ats_score']
                
                if job_company in market_intel.company_signals:
                    signals = market_intel.company_signals[job_company].get('signals', [])
                    market_output += f"- **{job_company}** (Job Match): Momentum {momentum}/100, ATS {ats}% — "
                    if 'funding' in signals:
                        market_output += "💰 Recently funded = likely hiring!\n"
                    elif 'expansion' in signals:
                        market_output += "📈 Expanding = new roles opening!\n"
                    else:
                        market_output += f"Signals: {', '.join(signals)}\n"
                else:
                    market_output += f"- **{job_company}** (Job Match): ATS {ats}% — No recent news (may still be hiring)\n"
    else:
        market_output += f"""### ⚠️ Limited Market Data

Only a few companies detected in recent news. This can happen when:
- News API returned few articles today
- Selected industry is niche
- Try selecting a different industry

**Don't worry** — your job matches still work! Market intelligence just adds extra context.
"""
    
    # Actionable recommendations
    market_output += "\n### 🎯 Recommended Actions\n\n"
    if market_intel.company_signals:
        high_momentum = [c for c, d in market_intel.company_signals.items() if d.get('momentum_score', 0) >= 60]
        if high_momentum:
            market_output += f"**Priority Applications:** {', '.join(high_momentum)} — these companies have strong momentum\n\n"
        
        hiring_companies = [c for c, d in market_intel.company_signals.items() if d.get('is_hiring', False)]
        if hiring_companies:
            market_output += f"**Confirmed Hiring:** {', '.join(hiring_companies)}\n\n"
    
    market_output += "**General Strategy:** Apply to high-momentum companies first — they're more likely to respond quickly.\n"
    
    # Output 3: AI-Analyzed Job Matches with ATS Scores
    location_str = f" in {target_location}" if target_location and target_location != "Any Location" else ""
    jobs_output = f"## 🎯 AI-Analyzed Opportunities: {target_role} ({target_industry}){location_str}\n\n"
    
    if not analyzed_opportunities:
        jobs_output += f"""### ⚠️ No Jobs Found

**What happened:** The job search for "{target_role}" didn't return results.

**Try these fixes:**
1. Select **"Any Location"** instead of a specific city
2. Try a more common role like "Data Analyst" or "Software Engineer"
3. Check terminal for detailed error messages

**Technical note:** We use Adzuna API (job aggregator). Very specific searches may return empty results.
"""
    elif analyzed_opportunities:
        # Show how many jobs found
        jobs_output += f"*Found {len(analyzed_opportunities)} opportunities*\n\n"
        for i, opp in enumerate(analyzed_opportunities[:8], 1):
            job = opp['job']
            perception = opp['perception']
            ats = opp['ats_analysis']
            
            ats_score = ats['ats_score']
            
            # ATS Score emoji and label
            if ats_score >= 80:
                ats_emoji = "🟢"
                ats_label = "STRONG MATCH"
            elif ats_score >= 60:
                ats_emoji = "🟡"
                ats_label = "GOOD MATCH"
            elif ats_score >= 40:
                ats_emoji = "🟠"
                ats_label = "PARTIAL MATCH"
            else:
                ats_emoji = "🔴"
                ats_label = "WEAK MATCH"
            
            jobs_output += f"""### {i}. {job['title']}
**Company:** {job['company']} | **Location:** {job['location']} | **Salary:** {job['salary_range']}

---

#### {ats_emoji} ATS Score: {ats_score}% | Recruiter Score: {ats.get('recruiter_score', ats_score)}%

| Metric | Status |
|--------|--------|
| **Skills Matched** | {len(ats['matched_skills'])}/{len(ats['matched_skills']) + len(ats['missing_skills'])} |
| **Experience Required** | {ats['required_experience']}+ years |
| **Your Experience** | {ats['candidate_experience']} years {'✅' if ats['experience_match'] else '⚠️'} |
| **Company Momentum** | {perception['company_momentum']}/100 |

"""
            
            # Show matched skills
            if ats['matched_skills']:
                jobs_output += f"**✅ Skills You Have:** {', '.join(ats['matched_skills'])}\n\n"
            
            # Show missing skills (GAP ANALYSIS)
            if ats['missing_skills']:
                jobs_output += f"**❌ Skills Gap:** {', '.join(ats['missing_skills'])}\n\n"
                jobs_output += f"**💡 To Improve Match:** Add projects/experience with: **{ats['missing_skills'][0]}**"
                if len(ats['missing_skills']) > 1:
                    jobs_output += f", **{ats['missing_skills'][1]}**"
                jobs_output += "\n\n"
            else:
                jobs_output += "**✅ No skill gaps detected!**\n\n"
            
            # AI Analysis
            jobs_output += f"**🤖 AI Analysis:** {opp['ai_analysis']}\n\n"
            
            jobs_output += f"🔗 **[Apply Here]({job['url']})**\n\n"
            jobs_output += "---\n\n"
    
    jobs_output = responsible_ai.add_transparency_disclaimer(jobs_output, "analysis")
    
    # Output 4: Resume Optimizer - COMPLETELY REBUILT
    recommendations_output = f"## 📊 Resume Optimizer - Recruiter Simulation\n\n"
    
    if analyzed_opportunities:
        selected_index = min(job_index, len(analyzed_opportunities) - 1)
        selected_opp = analyzed_opportunities[selected_index]
        selected_job = selected_opp['job']
        ats = selected_opp['ats_analysis']
        
        recommendations_output += f"""### 🎯 Optimizing for: {selected_job['title']} at {selected_job['company']}

*Analyzing Job #{selected_index + 1} — Use the slider above to optimize for different jobs*

---

### 📊 Dual Score Analysis (ATS + Recruiter)

| Score Type | Result | What This Means |
|------------|--------|-----------------|
| **ATS Pass Probability** | {"🟢" if ats['ats_score'] >= 70 else "🟡" if ats['ats_score'] >= 50 else "🔴"} **{ats['ats_score']}%** | {"✅ Will pass automated filters" if ats['ats_score'] >= 70 else "⚠️ May get filtered out" if ats['ats_score'] >= 50 else "❌ Likely to be auto-rejected"} |
| **Recruiter Shortlist Probability** | {"🟢" if ats['recruiter_score'] >= 70 else "🟡" if ats['recruiter_score'] >= 50 else "🔴"} **{ats['recruiter_score']}%** | {"✅ Strong candidate for interview" if ats['recruiter_score'] >= 70 else "⚠️ May lose to stronger candidates" if ats['recruiter_score'] >= 50 else "❌ Unlikely to be shortlisted"} |

"""
        
        # Explain score difference
        if ats['ats_score'] > ats['recruiter_score'] + 10:
            recommendations_output += f"**⚠️ Gap Alert:** You'll pass ATS filters but may struggle during human review. Focus on seniority signals and impact metrics.\n\n"
        elif ats['recruiter_score'] > ats['ats_score'] + 10:
            recommendations_output += f"**💡 Insight:** Your experience is strong but keyword optimization is needed to pass ATS first.\n\n"
        
        recommendations_output += "---\n\n"
        
        # Weighted Skill Breakdown
        recommendations_output += f"### 🔥 Weighted Skill Analysis (Role: {ats.get('detected_role', 'analyst').title()})\n\n"
        recommendations_output += "| Skill | Importance | Weight | Status |\n"
        recommendations_output += "|-------|------------|--------|--------|\n"
        
        for skill_data in ats.get('skill_breakdown', [])[:10]:  # Top 10 skills
            recommendations_output += f"| {skill_data['skill']} | {skill_data['importance_emoji']} {skill_data['importance'].title()} | {skill_data['weight']}pts | {skill_data['status']} |\n"
        
        recommendations_output += f"\n**Skill Score:** {ats.get('skill_score', 0)}% weighted match\n\n"
        
        recommendations_output += "---\n\n"
        
        # Seniority Signals
        recommendations_output += f"### 🧠 Seniority Signals (Expected: {ats.get('expected_seniority', 'mid').title()} Level)\n\n"
        
        seniority_signals = ats.get('seniority_signals', {})
        
        recommendations_output += "| Signal Type | Score | Keywords Found | Status |\n"
        recommendations_output += "|-------------|-------|----------------|--------|\n"
        
        for signal_type, data in seniority_signals.items():
            score = data.get('score', 0)
            found = data.get('found', [])
            status = "✅ Strong" if score >= 50 else ("⚠️ Weak" if score > 0 else "❌ Missing")
            found_str = ', '.join(found[:3]) if found else "None"
            recommendations_output += f"| {signal_type.replace('_', ' ').title()} | {score}/100 | {found_str} | {status} |\n"
        
        recommendations_output += f"\n**Overall Seniority Score:** {ats.get('seniority_score', 0)}/100\n\n"
        
        if ats.get('expected_seniority') == 'senior' and ats.get('seniority_score', 0) < 40:
            recommendations_output += "**⚠️ Warning:** This is a senior role but your resume lacks leadership language. Add words like 'led', 'owned', 'mentored', 'strategic'.\n\n"
        
        recommendations_output += "---\n\n"
        
        # Impact Score
        recommendations_output += "### 📈 Business Impact Score\n\n"
        
        impact_components = ats.get('impact_components', {})
        metrics_found = ats.get('metrics_found', {})
        
        recommendations_output += f"**Overall Impact Score: {ats.get('impact_score', 0)}/100**\n\n"
        
        recommendations_output += "| Component | Score | Evidence Found |\n"
        recommendations_output += "|-----------|-------|----------------|\n"
        recommendations_output += f"| Metrics Present | {impact_components.get('metrics_present', 0)}/40 | {len(metrics_found.get('percentages', [])) + len(metrics_found.get('dollars', []))} instances |\n"
        recommendations_output += f"| Financial Impact | {impact_components.get('financial_impact', 0)}/30 | {', '.join(metrics_found.get('dollars', [])[:2]) or 'None found'} |\n"
        recommendations_output += f"| Scale Indicators | {impact_components.get('scale_indicators', 0)}/30 | {', '.join(metrics_found.get('scale', [])[:2]) or 'None found'} |\n"
        
        if ats.get('impact_score', 0) < 40:
            recommendations_output += "\n**🔴 Impact Too Low:** Add quantified achievements like:\n"
            recommendations_output += "- 'Reduced processing time by **40%**'\n"
            recommendations_output += "- 'Drove **$2M** in cost savings'\n"
            recommendations_output += "- 'Managed pipeline of **1M+** daily transactions'\n\n"
        
        recommendations_output += "---\n\n"
        
        # Experience Match
        recommendations_output += "### 📅 Experience Match\n\n"
        recommendations_output += f"| Requirement | Status |\n"
        recommendations_output += f"|-------------|--------|\n"
        recommendations_output += f"| Required Experience | {ats['required_experience']}+ years |\n"
        recommendations_output += f"| Your Experience | {ats['candidate_experience']} years |\n"
        
        exp_gap = ats.get('experience_gap', 0)
        exp_status = "✅ Meets requirement" if ats['experience_match'] else f"❌ Gap of {exp_gap} years"
        recommendations_output += f"| Match Status | {exp_status} |\n\n"
        
        recommendations_output += "---\n\n"
        
        # Why You Might Get Rejected
        recommendations_output += "### ❌ Why You Might Get Rejected (Recruiter Simulation)\n\n"
        
        rejection_reasons = ats.get('rejection_reasons', [])
        if rejection_reasons:
            for reason in rejection_reasons:
                recommendations_output += f"- {reason}\n"
        else:
            recommendations_output += "✅ No major red flags detected. Strong candidate!\n"
        
        recommendations_output += "\n---\n\n"
        
        # Actionable Improvements
        recommendations_output += "### 🚀 Top 3 Improvements to Increase Your Score\n\n"
        
        improvements = []
        
        # Check for critical missing skills
        critical_missing = [s for s in ats.get('skill_breakdown', []) if s['importance'] == 'critical' and not s['has_skill']]
        if critical_missing:
            improvements.append(f"**Add critical skill:** {critical_missing[0]['skill']} — This alone could add +15-25% to your score")
        
        # Check seniority signals
        if ats.get('seniority_score', 0) < 50:
            improvements.append("**Add ownership language:** Replace 'worked on' with 'led', 'owned', 'drove' — Recruiters scan for these")
        
        # Check impact
        if ats.get('impact_score', 0) < 40:
            improvements.append("**Quantify your impact:** Add at least 3 metrics (%, $, scale) to your bullet points")
        
        # Check experience
        if not ats.get('experience_match', True):
            improvements.append(f"**Address experience gap:** Highlight projects that demonstrate {ats.get('expected_seniority', 'senior')}-level work")
        
        # Default improvements if none found
        if not improvements:
            improvements = [
                "**Tailor keywords:** Mirror the exact phrases from the job description",
                "**Add recent tech:** Include modern tools like dbt, Airflow if you have them",
                "**Show progression:** Highlight career growth and increasing responsibility"
            ]
        
        for i, improvement in enumerate(improvements[:3], 1):
            recommendations_output += f"{i}. {improvement}\n"
        
        recommendations_output += "\n"
        recommendations_output = responsible_ai.add_transparency_disclaimer(recommendations_output, "analysis")
    else:
        recommendations_output += "*No jobs to analyze. Run the analysis first.*\n"
    
    # Output 5: Role Readiness Assessment (Honest, Defensible)
    benchmark_output = f"## 📊 Role Readiness Assessment: {target_role}\n\n"
    print("📊 Generating role readiness assessment...")
    
    selected_job = analyzed_opportunities[min(job_index, len(analyzed_opportunities) - 1)]['job'] if analyzed_opportunities else None
    benchmark_output += content_gen.generate_top_performer_benchmark(target_role, candidate, selected_job)
    benchmark_output = responsible_ai.add_transparency_disclaimer(benchmark_output, "analysis")
    
    # Output 6: 90-Day Execution Plan (Outcome-Focused)
    roadmap_output = f"## 🚀 90-Day Interview Conversion Plan: {target_role} in {target_industry}\n\n"
    print("🚀 Generating 90-day execution plan...")
    
    # Get ATS analysis from selected job for baseline metrics
    selected_ats = analyzed_opportunities[min(job_index, len(analyzed_opportunities) - 1)]['ats_analysis'] if analyzed_opportunities else None
    roadmap_output += content_gen.generate_career_roadmap(target_role, candidate, target_industry, selected_ats)
    roadmap_output = responsible_ai.add_transparency_disclaimer(roadmap_output, "analysis")
    
    # Output 7: Generated Cover Letters (MULTIPLE - for top jobs)
    cover_letter_output = ""
    if generate_cover_letter and analyzed_opportunities:
        cover_letter_output = "## ✉️ AI-Generated Cover Letters\n\n"
        cover_letter_output += "*Select the job number with the slider to see that cover letter first*\n\n"
        
        # Generate for selected job first, then next 2 jobs
        selected_index = min(job_index, len(analyzed_opportunities) - 1)
        jobs_to_generate = [selected_index]
        
        # Add 2 more jobs (if available)
        for i in range(len(analyzed_opportunities)):
            if i != selected_index and len(jobs_to_generate) < 3:
                jobs_to_generate.append(i)
        
        for idx in jobs_to_generate:
            selected_job = analyzed_opportunities[idx]['job']
            ats_score = analyzed_opportunities[idx]['ats_analysis']['ats_score']
            
            cover_letter_output += f"### 📧 Cover Letter #{idx + 1}: {selected_job['title']} at {selected_job['company']}\n"
            cover_letter_output += f"*ATS Match: {ats_score}%*\n\n"
            cover_letter_output += content_gen.generate_cover_letter(selected_job, candidate, market_intel)
            cover_letter_output += "\n\n---\n\n"
        
        cover_letter_output = responsible_ai.add_transparency_disclaimer(cover_letter_output, "cover_letter")
    
    # Output 8: Interview Prep (MULTIPLE - for top jobs)
    interview_output = ""
    if generate_interview_prep and analyzed_opportunities:
        interview_output = "## 🎤 AI Interview Preparation Guides\n\n"
        
        # Generate for selected job first, then 1 more
        selected_index = min(job_index, len(analyzed_opportunities) - 1)
        jobs_to_generate = [selected_index]
        
        # Add 1 more job (if available)
        for i in range(len(analyzed_opportunities)):
            if i != selected_index and len(jobs_to_generate) < 2:
                jobs_to_generate.append(i)
        
        for idx in jobs_to_generate:
            selected_job = analyzed_opportunities[idx]['job']
            ats_score = analyzed_opportunities[idx]['ats_analysis']['ats_score']
            
            interview_output += f"### 🎯 Interview Prep #{idx + 1}: {selected_job['title']} at {selected_job['company']}\n"
            interview_output += f"*ATS Match: {ats_score}%*\n\n"
            interview_output += content_gen.generate_interview_prep(selected_job, candidate, market_intel)
            interview_output += "\n\n---\n\n"
        
        interview_output = responsible_ai.add_transparency_disclaimer(interview_output, "interview_prep")
    
    # Completion message
    elapsed = time.time() - start_time
    print(f"✅ [7/7] Analysis complete!")
    print(f"="*60)
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print(f"📊 Jobs analyzed: {len(analyzed_opportunities)}")
    print(f"="*60 + "\n")
    
    return resume_output, market_output, jobs_output, recommendations_output, benchmark_output, roadmap_output, cover_letter_output, interview_output


# ============================================
# 12. GRADIO INTERFACE
# ============================================
def create_gradio_interface():
    """Create the enhanced Gradio interface"""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Fintech Career AI") as demo:
        
        gr.Markdown("""
        # 🚀 Career Intelligence Bot v2.0
        ### AI-Powered Career Platform for Any Industry
        
        **Your One-Stop Job Search Solution:**
        - 🏢 **Multi-Industry** — Fintech, FAANG, Healthcare, Consulting & more
        - 🧠 **AI Analysis** — LLM-powered resume & job matching
        - 📊 **ATS Scoring** — Know your match percentage
        - 🏆 **Benchmarking** — Compare to top 10% candidates
        - 🗺️ **90-Day Roadmap** — Personalized action plan
        - ✉️ **Cover Letters** — AI-generated for multiple jobs
        - 🎤 **Interview Prep** — Company-specific preparation
        
        *Built by Neelima Verma | MS Data Science, Pace University*
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                resume_file = gr.File(
                    label="📄 Upload Resume (PDF)",
                    file_types=[".pdf"]
                )
                
                gr.Markdown("### 🎯 Target Industry & Role")
                target_industry = gr.Dropdown(
                    label="🏢 Target Industry",
                    choices=list(INDUSTRIES.keys()),
                    value="Fintech"
                )
                target_role = gr.Dropdown(
                    label="🎯 Target Role",
                    choices=ROLES_BY_INDUSTRY["Fintech"],
                    value="Data Analyst"
                )
                target_location = gr.Dropdown(
                    label="📍 Preferred Location",
                    choices=[
                        "Any Location",
                        "New York", 
                        "San Francisco",
                        "Los Angeles",
                        "Chicago",
                        "Boston",
                        "Seattle",
                        "Austin",
                        "Denver",
                        "Atlanta",
                        "Miami",
                        "Washington DC",
                        "Remote"
                    ],
                    value="Any Location"
                )
                
                gr.Markdown("### 🤖 AI Features")
                gen_cover_letter = gr.Checkbox(
                    label="✉️ Generate Cover Letter",
                    value=True
                )
                gen_interview = gr.Checkbox(
                    label="🎤 Generate Interview Prep Guide",
                    value=True
                )
                job_selection = gr.Slider(
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=1,
                    label="📝 Start generating from Job # (generates for this job + 2 more)"
                )
                
                submit_btn = gr.Button("🚀 Analyze with AI", variant="primary")
        
        with gr.Tabs():
            with gr.TabItem("📄 Resume Analysis"):
                resume_output = gr.Markdown()
            with gr.TabItem("📈 Market Intelligence"):
                market_output = gr.Markdown()
            with gr.TabItem("🎯 AI Job Matches"):
                jobs_output = gr.Markdown()
            with gr.TabItem("📊 Resume Optimizer"):
                recommendations_output = gr.Markdown()
            with gr.TabItem("📊 Role Readiness"):
                benchmark_output = gr.Markdown()
            with gr.TabItem("🚀 90-Day Plan"):
                roadmap_output = gr.Markdown()
            with gr.TabItem("✉️ Cover Letters"):
                cover_output = gr.Markdown()
            with gr.TabItem("🎤 Interview Prep"):
                interview_output = gr.Markdown()
        
        # Dynamic role updating based on industry
        def update_roles(industry):
            roles = ROLES_BY_INDUSTRY.get(industry, ROLES_BY_INDUSTRY["Fintech"])
            return gr.Dropdown(choices=roles, value=roles[0])
        
        target_industry.change(
            fn=update_roles,
            inputs=[target_industry],
            outputs=[target_role]
        )
        
        # API Status
        with gr.Accordion("⚙️ API Configuration Status", open=False):
            api_status = f"""
            - **Groq API (FREE!):** {'✅ Configured' if GROQ_API_KEY else '❌ Not set - Get FREE key at console.groq.com/keys'}
            - **NewsAPI:** {'✅ Configured' if NEWS_API_KEY else '❌ Not set'}
            - **Adzuna API:** {'✅ Configured' if ADZUNA_APP_ID else '❌ Not set'}
            
            *Groq provides FREE access to Llama 3.3 70B - no credit card required!*
            """
            gr.Markdown(api_status)
        
        submit_btn.click(
            fn=process_ai_career_intelligence,
            inputs=[resume_file, target_industry, target_role, target_location, gen_cover_letter, gen_interview, job_selection],
            outputs=[resume_output, market_output, jobs_output, recommendations_output, benchmark_output, roadmap_output, cover_output, interview_output]
        )
    
    return demo


# ============================================
# 13. MAIN ENTRY POINT
# ============================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("🚀 CAREER INTELLIGENCE BOT v2.0 - AI POWERED")
    print("="*60)
    print("\nSUPPORTED INDUSTRIES:")
    for industry in INDUSTRIES.keys():
        print(f"  🏢 {industry}")
    print("\nTOPICS DEMONSTRATED:")
    print("  ✅ Large Language Models (LLMs) - Llama 3.3 70B via Groq")
    print("  ✅ Generative AI Applications")
    print("  ✅ Retrieval-Augmented Generation (RAG)")
    print("  ✅ Agentic Design & Reasoning")
    print("  ✅ Prompt Engineering")
    print("  ✅ Multi-Industry AI (FAANG, Fintech, Healthcare, etc.)")
    print("  ✅ Responsible AI & Transparency")
    print("  ✅ AI for Good (Career Accessibility)")
    print("\nAPI STATUS:")
    print(f"  {'✅' if GROQ_API_KEY else '❌'} Groq API (FREE! Get key at console.groq.com/keys)")
    print(f"  {'✅' if NEWS_API_KEY else '❌'} NewsAPI")
    print(f"  {'✅' if ADZUNA_APP_ID else '❌'} Adzuna API")
    print("\n" + "="*60 + "\n")
    
    demo = create_gradio_interface()
    demo.launch()
