import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

FILLER_WORDS = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally', 'i mean']

LAYER1_THRESHOLDS = {
    "min_word_count": 30,
    "max_silence_ratio": 0.40,
    "min_focus_score": 85
}

WPM_RANGES = {
    "slow": (0, 100),
    "good": (100, 160),
    "fast": (160, 999)
}

CANDIDATE_TOKENS = {"thinhbeo", "thanhbusy"}

ADMIN_TOKENS = {"luandeptrai","hongraphay"}

FIXED_QUESTIONS = [
    "Please introduce yourself and briefly describe your background.",
    "Why are you interested in working as a Data Analyst?"
]

QUESTION_POOL = [
    # Data Cleaning & Preparation 
    "How would you handle missing or inconsistent data in a dataset?",
    "Describe the steps you usually take to clean a messy dataset.",
    "What techniques do you use to detect outliers?",
    
    # SQL Skills 
    "What SQL functions or commands do you use most often, and why?",
    "How would you find duplicate records in a table using SQL?",
    "Explain the difference between INNER JOIN and LEFT JOIN.",
    
    # Business Analysis 
    "How do you determine which metrics or KPIs matter for a business problem?",
    "What steps do you follow when starting a new analysis project?",
    "Explain a situation where your analysis influenced a business decision.",
    "How do you validate whether your findings are reliable?",
    
    # Visualization 
    "How do you decide which chart type is appropriate for the data?",
    "Describe a dashboard you built and what decisions it helped support.",
    
    # Statistical Thinking 
    "Explain the difference between correlation and causation.",
    "How would you explain p-value to someone without a statistics background?",
    "You find a strong correlation in the data. What steps do you take before presenting it?"
]

VALID_TOKENS = CANDIDATE_TOKENS.union(ADMIN_TOKENS)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_MIME_TYPES = {"video/webm", "video/mp4"}
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


