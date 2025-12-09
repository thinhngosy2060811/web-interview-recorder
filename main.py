from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import pytz
import sys 
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import asyncio
import json
import re
import logging
import uvicorn
import subprocess
import random
import whisper  
from typing import Optional
import statistics 
import google.generativeai as genai 
from fastapi import BackgroundTasks 
from typing import Dict, List, Optional, Set
from dotenv import load_dotenv
import os 

# Cấu hình AI
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Key của bạn
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
# Token dành cho ứng viên 
CANDIDATE_TOKENS = {"thinhbeo", "thanhbusy"}
# Token dành cho người chấm
ADMIN_TOKENS = {"luandeptrai","hongraphay"}
# --- CẤU HÌNH NGÂN HÀNG CÂU HỎI ---
FIXED_QUESTIONS = [
    "Please introduce yourself and briefly describe your background.",
    "Why are you interested in working as a Data Analyst?"
]

QUESTION_POOL = [
    # Data Cleaning & Preparation (3 câu)
    "How would you handle missing or inconsistent data in a dataset?",
    "Describe the steps you usually take to clean a messy dataset.",
    "What techniques do you use to detect outliers?",
    
    # SQL Skills (3 câu)
    "What SQL functions or commands do you use most often, and why?",
    "How would you find duplicate records in a table using SQL?",
    "Explain the difference between INNER JOIN and LEFT JOIN.",
    
    # Business Analysis (4 câu)
    "How do you determine which metrics or KPIs matter for a business problem?",
    "What steps do you follow when starting a new analysis project?",
    "Explain a situation where your analysis influenced a business decision.",
    "How do you validate whether your findings are reliable?",
    
    # Visualization (2 câu)
    "How do you decide which chart type is appropriate for the data?",
    "Describe a dashboard you built and what decisions it helped support.",
    
    # Statistical Thinking (3 câu)
    "Explain the difference between correlation and causation.",
    "How would you explain p-value to someone without a statistics background?",
    "You find a strong correlation in the data. What steps do you take before presenting it?"
]

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Load Whisper Model ---
try:
    logger.info("Loading Whisper model (small)... This may take a while on first run.")
    # Model 'small' cân bằng giữa tốc độ và độ chính xác (khoảng 244MB)
    WHISPER_MODEL = whisper.load_model("small")
    logger.info("✅ Whisper model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    logger.warning("⚠️ Transcription feature will be disabled.")
    WHISPER_MODEL = None

# --- Configuration ---
app = FastAPI(title="Web Interview Recorder (Integrated)", version="2.0")
# Khởi tạo gemini
gemini_model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # ✅ THÊM SAFETY SETTINGS ĐỂ TRÁNH BỊ BLOCK
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        gemini_model = genai.GenerativeModel(
            'gemini-2.5-flash',
            safety_settings=safety_settings
        )
        
        logger.info("✅ Gemini 2.5 Flash initialized successfully")
        
        # ✅ TEST CONNECTION VỚI SAFETY SETTINGS
        try:
            test_response = gemini_model.generate_content(
                "Respond with only: OK",
                generation_config={"max_output_tokens": 100}
            )
            
            # ✅ KIỂM TRA FINISH REASON
            if hasattr(test_response, 'candidates') and test_response.candidates:
                finish_reason = test_response.candidates[0].finish_reason     
                if finish_reason == 1:  # STOP = success
                    logger.info(f"✅ Gemini test successful: {test_response.text[:50]}")
                elif finish_reason == 3:  # SAFETY
                    logger.warning("⚠️ Gemini test blocked by safety filter, but model is working")
                    logger.warning("This is OK - safety filter will be less strict for actual content")
                else:
                    logger.warning(f"⚠️ Gemini test finished with reason: {finish_reason}")
            else:
                logger.warning("⚠️ Cannot check finish_reason, but model initialized")
                
        except Exception as test_error:
            # Nếu test fail nhưng model đã init, vẫn có thể dùng được
            logger.warning(f"⚠️ Gemini test failed but model initialized: {test_error}")
            logger.info("✅ Model will still work for actual interview content")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini: {e}")
        logger.error("Vui lòng kiểm tra:")
        logger.error("1. API key có đúng không?")
        logger.error("2. Đã bật Gemini API trong Google Cloud Console chưa?")
        logger.error("3. Có billing account chưa?")
        gemini_model = None
else:
    logger.error("❌ Cannot initialize Gemini - No API key provided")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
# Mở quyền truy cập vào thư mục uploads để xem video MP4
app.mount("/uploads", StaticFiles(directory=BASE_DIR / "uploads"), name="uploads")

VALID_TOKENS = CANDIDATE_TOKENS.union(ADMIN_TOKENS)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_MIME_TYPES = {"video/webm", "video/mp4"}
BANGKOK_TZ = pytz.timezone('Asia/Bangkok')

active_sessions = {}
metadata_locks = {}

# --- Pydantic Models ---
class TokenRequest(BaseModel):
    token: str

class SessionStartRequest(BaseModel):
    token: str
    userName: str

class SessionFinishRequest(BaseModel):
    token: str
    folder: str
    questionsCount: int

# --- Helper Functions ---
def sanitize_username(username: str) -> str:
    safe_name = re.sub(r'[^\w\s-]', '', username)
    safe_name = re.sub(r'\s+', '_', safe_name)
    safe_name = safe_name.strip('_')
    return safe_name.lower()[:50]

def get_bangkok_timestamp() -> str:
    return datetime.now(BANGKOK_TZ).isoformat()

def generate_folder_name(username: str) -> str:
    now = datetime.now(BANGKOK_TZ)
    sanitized = sanitize_username(username)
    return f"{now.strftime('%d_%m_%Y_%H_%M')}_{sanitized}"

def verify_video_by_signature(file_path: Path) -> bool:
    try:
        with file_path.open('rb') as f:
            header = f.read(32)
            if header[:4] == b'\x1a\x45\xdf\xa3': return True # WebM
            if b'ftyp' in header[:12]: return True # MP4
            if header[:4] == b'RIFF' and header[8:12] == b'AVI ': return True # AVI
            if b'moov' in header or b'mdat' in header: return True # MOV
            return False
    except Exception as e:
        logger.error(f"Error reading file signature: {e}")
        return False

def convert_to_mp4(input_path: Path):
    """Dùng FFmpeg convert WebM sang MP4"""
    try:
        output_path = input_path.with_suffix(".mp4")
        command = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", str(output_path)
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"✅ Converted to MP4: {output_path.name}")
        return output_path.name
    except Exception as e:
        logger.error(f"❌ FFmpeg conversion failed: {e}")
        return None

async def transcribe_video_whisper(video_path: Path, question_index: int) -> Optional[str]:
    """Chuyển đổi Video sang Text dùng Whisper"""
    if WHISPER_MODEL is None:
        return None
    
    try:
        # 1. Extract audio
        audio_path = video_path.with_suffix('.wav')
        subprocess.run([
            'ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', str(audio_path), '-y', '-loglevel', 'error'
        ], check=False, timeout=60)
        
        # 2. Transcribe (Chạy trong thread riêng để không block server)
        logger.info(f"🎤 Transcribing Q{question_index}...")
        whisper_result = await asyncio.to_thread(
            WHISPER_MODEL.transcribe,
            str(audio_path),
            language='en',
            fp16=False
        )
        
        # 3. Format Transcript
        transcript_text = f"=== TRANSCRIPT Q{question_index} ===\n"
        transcript_text += f"Time: {get_bangkok_timestamp()}\n\n"
        transcript_text += whisper_result['text'].strip() + "\n\n"
        transcript_text += "--- TIMESTAMPS ---\n"
        
        for segment in whisper_result.get('segments', []):
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            transcript_text += f"[{start//60:02.0f}:{start%60:05.2f} -> {end//60:02.0f}:{end%60:05.2f}] {text}\n"
            
        # 4. Save to file
        transcript_file = video_path.parent / f"Q{question_index}_transcript.txt"
        transcript_file.write_text(transcript_text, encoding='utf-8')
        logger.info(f"📝 Transcript saved: {transcript_file.name}")
        
        # Cleanup
        audio_path.unlink(missing_ok=True)
        return whisper_result['text'].strip()

    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        return None

async def create_metadata(folder_path: Path, username: str, questions_list: list) -> dict:
    questions_dict = [
        {"index": i + 1, "text": question}
        for i, question in enumerate(questions_list)
    ]
    metadata = {
        "userName": username,
        "sessionStartTime": get_bangkok_timestamp(),
        "timeZone": "Asia/Bangkok",
        "interviewQuestions": questions_dict,
        "questions": [],
        "questionsCount": 0,
        "sessionEnded": False,
        "sessionEndTime": None
    }
    meta_file = folder_path / "meta.json"
    async with asyncio.Lock():
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata

async def update_metadata(folder_path: Path, question_data: dict = None, finalize: bool = False, questions_count: int = None):
    meta_file = folder_path / "meta.json"
    if not meta_file.exists(): raise HTTPException(404, "Metadata file not found")
    
    folder_key = str(folder_path)
    if folder_key not in metadata_locks:
        metadata_locks[folder_key] = asyncio.Lock()
    
    async with metadata_locks[folder_key]:
        with meta_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        if question_data:
            # Kiểm tra xem đã có câu hỏi này chưa, nếu có thì update (cho trường hợp update transcript sau)
            existing_idx = next((i for i, q in enumerate(metadata["questions"]) if q["index"] == question_data["index"]), -1)
            if existing_idx != -1:
                # Merge data mới vào data cũ
                metadata["questions"][existing_idx].update(question_data)
            else:
                metadata["questions"].append(question_data)
        
        if finalize:
            metadata["sessionEnded"] = True
            metadata["sessionEndTime"] = get_bangkok_timestamp()
            if questions_count is not None:
                metadata["questionsCount"] = questions_count
            
            # Đếm số transcript đã tạo
            transcript_count = len(list(folder_path.glob("*_transcript.txt")))
            metadata["transcriptsGenerated"] = transcript_count
        
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

def calculate_filler_density(transcript_text: str) -> float:
    """
    Tính mật độ từ đệm (Filler Density)
    Returns: Percentage (0-100)
    """
    text_lower = transcript_text.lower()
    words = text_lower.split()
    
    if len(words) == 0:
        return 0.0
    
    filler_count = sum(1 for word in words if word in FILLER_WORDS)
    
    # Check multi-word fillers
    for filler in ['you know', 'i mean']:
        filler_count += text_lower.count(filler)
    
    density = (filler_count / len(words)) * 100
    return round(density, 2)


def calculate_silence_ratio(segments: List[Dict]) -> tuple:
    """
    Tính tỷ lệ im lặng (Silence Ratio)
    Returns: (silence_ratio_percentage, total_pause_time, num_pauses)
    """
    if not segments or len(segments) < 2:
        return 0.0, 0.0, 0
    
    total_pause_time = 0.0
    num_pauses = 0
    
    for i in range(1, len(segments)):
        gap = segments[i]['start'] - segments[i-1]['end']
        if gap > 1.5:  # Ngưỡng 1.5 giây
            total_pause_time += gap
            num_pauses += 1
    
    total_duration = segments[-1]['end'] - segments[0]['start']
    
    if total_duration == 0:
        return 0.0, 0.0, 0
    
    silence_ratio = (total_pause_time / total_duration) * 100
    return round(silence_ratio, 2), round(total_pause_time, 2), num_pauses


def calculate_speaking_rate(word_count: int, duration_seconds: float) -> tuple:
    """
    Tính tốc độ nói (Speaking Rate)
    Returns: (wpm, category)
    """
    if duration_seconds == 0:
        return 0, "unknown"
    
    duration_minutes = duration_seconds / 60
    wpm = word_count / duration_minutes
    
    if wpm < WPM_RANGES["slow"][1]:
        category = "slow"
    elif wpm < WPM_RANGES["good"][1]:
        category = "good"
    else:
        category = "fast"
    
    return round(wpm, 1), category


def analyze_layer1_metrics(transcript_data: Dict, focus_score: int) -> Dict:
    """
    Phân tích Layer 1 - Quantitative Metrics
    
    Args:
        transcript_data: Dict chứa 'text' và 'segments' từ Whisper
        focus_score: Focus score từ Frontend (0-100)
    
    Returns:
        Dict chứa tất cả metrics và decision
    """
    text = transcript_data.get('text', '')
    segments = transcript_data.get('segments', [])
    
    # 1. Word count
    words = text.split()
    word_count = len(words)
    
    # 2. Filler density
    filler_density = calculate_filler_density(text)
    
    # 3. Silence ratio
    silence_ratio, pause_time, num_pauses = calculate_silence_ratio(segments)
    
    # 4. Speaking rate
    duration = segments[-1]['end'] if segments else 0
    wpm, wpm_category = calculate_speaking_rate(word_count, duration)
    
    # 5. Decision
    is_bad = (
        word_count < LAYER1_THRESHOLDS["min_word_count"] or
        silence_ratio > LAYER1_THRESHOLDS["max_silence_ratio"] * 100 or
        focus_score < LAYER1_THRESHOLDS["min_focus_score"]
    )
    
    result = {
        "word_count": word_count,
        "filler_density_percent": filler_density,
        "silence_ratio_percent": silence_ratio,
        "total_pause_seconds": pause_time,
        "num_pauses": num_pauses,
        "speaking_rate_wpm": wpm,
        "wpm_category": wpm_category,
        "focus_score": focus_score,
        "duration_seconds": round(duration, 2),
        "flagged_as_bad": is_bad,
        "flag_reasons": []
    }
    
    # Ghi lý do flag
    if word_count < LAYER1_THRESHOLDS["min_word_count"]:
        result["flag_reasons"].append(f"Too short ({word_count} words < {LAYER1_THRESHOLDS['min_word_count']})")
    
    if silence_ratio > LAYER1_THRESHOLDS["max_silence_ratio"] * 100:
        result["flag_reasons"].append(f"Too much silence ({silence_ratio}% > {LAYER1_THRESHOLDS['max_silence_ratio']*100}%)")
    
    if focus_score < LAYER1_THRESHOLDS["min_focus_score"]:
        result["flag_reasons"].append(f"Low focus ({focus_score}% < {LAYER1_THRESHOLDS['min_focus_score']}%)")
    
    return result

# ============== LAYER 2: AI SEMANTIC ANALYSIS (GEMINI) ==============

async def analyze_layer2_ai(
    question_text: str,
    transcript_text: str,
    layer1_metrics: Dict
) -> Dict:
    """
    Phân tích Layer 2 - AI Semantic Analysis using Gemini 2.5 Flash
    
    Args:
        question_text: Câu hỏi được hỏi
        transcript_text: Transcript đầy đủ
        layer1_metrics: Kết quả từ Layer 1
    
    Returns:
        Dict chứa priority, reason, và AI analysis
    """
    
    # ============== VALIDATE KEY VÀ MODEL ==============
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY not available")
        return {
            "priority": "UNKNOWN",
            "reason": "There no have API key",
            "ai_available": False,
            "error": "GEMINI_API_KEY not set"
        }
    
    if not gemini_model:
        logger.error("❌ Gemini model not initialized")
        return {
            "priority": "UNKNOWN",
            "reason": "AI model is not generate",
            "ai_available": False,
            "error": "Gemini model is None"
        }
    
    try:
        # ✅ THÊM SAFETY SETTINGS
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            safety_settings=safety_settings
        )
        
        logger.info("🤖 Calling Gemini for Layer 2 analysis...")
        
        # ============== CONSTRUCT PROMPT ==============
        prompt = f"""You are an expert HR interviewer evaluating a Data Analyst candidate's video interview response.

**QUESTION ASKED:**
{question_text}

**CANDIDATE'S TRANSCRIPT:**
{transcript_text}

**QUANTITATIVE METRICS (Layer 1):**
- Word Count: {layer1_metrics['word_count']}
- Filler Density: {layer1_metrics['filler_density_percent']}%
- Silence Ratio: {layer1_metrics['silence_ratio_percent']}%
- Speaking Rate: {layer1_metrics['speaking_rate_wpm']} WPM ({layer1_metrics['wpm_category']})
- Focus Score: {layer1_metrics['focus_score']}%
- Flagged as Bad: {layer1_metrics['flagged_as_bad']}

**YOUR TASK:**
Evaluate the candidate's response and assign a priority level for HR review.
You must answer briefly but fully and to the point

**EVALUATION CRITERIA:**
1. **Content Relevance**: Does the answer address the question directly?
2. **Communication Skills**: Clear structure, fluency, confidence
3. **Professionalism**: Appropriate tone and demeanor
4. **Technical Depth** (if applicable): Shows understanding of concepts

**OUTPUT FORMAT (JSON ONLY):**
{{
  "priority": "HIGH" | "MEDIUM" | "LOW",
  "reason": "Brief 1-2 sentence explanation in English for HR",
  "content_score": 0-10,
  "communication_score": 0-10,
  "overall_impression": "positive" | "neutral" | "negative"
}}

**PRIORITY GUIDELINES:**
- HIGH: Strong candidate, clear answers, good communication, relevant experience
- MEDIUM: Acceptable but has some weaknesses, needs closer review
- LOW: Poor answer quality, irrelevant content, or major communication issues

IMPORTANT: Respond with ONLY the JSON object. No markdown, no extra text."""

        # ============== CALL GEMINI API ==============
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=generation_config
        )
        
        # ============== CHECK FINISH REASON ==============
        if hasattr(response, 'candidates') and response.candidates:
            finish_reason = response.candidates[0].finish_reason
            
            if finish_reason == 3:  # SAFETY blocked
                logger.warning("⚠️ Response blocked by safety filter")
                return {
                    "priority": "MEDIUM",
                    "reason": "Không thể phân tích do bộ lọc an toàn",
                    "ai_available": False,
                    "error": "SAFETY_BLOCKED"
                }
            elif finish_reason != 1:  # Not STOP
                logger.warning(f"⚠️ Unexpected finish_reason: {finish_reason}")
        
        # ============== PARSE RESPONSE ==============
        response_text = response.text.strip()
        logger.info(f"📥 Raw Gemini response (first 200 chars): {response_text[:200]}")
        
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        ai_result = json.loads(response_text)
        
        # ============== VALIDATE REQUIRED FIELDS ==============
        required_fields = ["priority", "reason"]
        for field in required_fields:
            if field not in ai_result:
                raise ValueError(f"Missing required field: {field}")
        
        # Set defaults for optional fields
        ai_result.setdefault("content_score", 5)
        ai_result.setdefault("communication_score", 5)
        ai_result.setdefault("overall_impression", "neutral")
        
        logger.info(f"✅ AI Analysis completed - Priority: {ai_result.get('priority', 'UNKNOWN')}")
        
        return {
            **ai_result,
            "ai_available": True,
            "model_used": "gemini-2.5-flash",
            "tokens_used": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse Gemini response: {e}")
        logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'No response'}")
        return {
            "priority": "MEDIUM",
            "reason": "Lỗi phân tích AI (JSON parse error), cần review thủ công",
            "ai_available": False,
            "error": f"JSON parse error: {str(e)}",
            "raw_response": response_text[:500] if 'response_text' in locals() else None
        }
    
    except Exception as e:
        logger.error(f"❌ Gemini API error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return {
            "priority": "MEDIUM",
            "reason": "Lỗi hệ thống AI, cần review thủ công",
            "ai_available": False,
            "error": str(e)
        }
        
async def calculate_final_ranking(folder_path: Path):
    """
    Tính toán ranking tổng hợp sau khi hoàn thành tất cả câu hỏi
    """
    meta_file = folder_path / "meta.json"
    
    if not meta_file.exists():
        logger.error("❌ Metadata file not found for final ranking")
        return
    
    try:
        with meta_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        questions = metadata.get("questions", [])
        
        if not questions:
            logger.warning("⚠️ No questions data for ranking")
            return
        
        # ============== COLLECT METRICS ==============
        all_metrics = {
            "word_counts": [],
            "filler_densities": [],
            "silence_ratios": [],
            "wpms": [],
            "focus_scores": [],
            "priorities": [],
            "content_scores": [],
            "communication_scores": []
        }
        
        valid_questions = 0
        
        for q in questions:
            metrics = q.get("metrics", {})
            ai_eval = q.get("ai_evaluation", {})
            
            if metrics:
                all_metrics["word_counts"].append(metrics.get("word_count", 0))
                all_metrics["filler_densities"].append(metrics.get("filler_density_percent", 0))
                all_metrics["silence_ratios"].append(metrics.get("silence_ratio_percent", 0))
                all_metrics["wpms"].append(metrics.get("speaking_rate_wpm", 0))
                all_metrics["focus_scores"].append(metrics.get("focus_score", 0))
                valid_questions += 1
            
            if ai_eval and ai_eval.get("ai_available"):
                priority = ai_eval.get("priority", "MEDIUM")
                all_metrics["priorities"].append(priority)
                all_metrics["content_scores"].append(ai_eval.get("content_score", 5))
                all_metrics["communication_scores"].append(ai_eval.get("communication_score", 5))
        
        # ============== CALCULATE AVERAGES ==============
        def safe_mean(lst):
            return round(statistics.mean(lst), 2) if lst else 0
        
        summary = {
            "total_questions_analyzed": valid_questions,
            "avg_word_count": safe_mean(all_metrics["word_counts"]),
            "avg_filler_density_percent": safe_mean(all_metrics["filler_densities"]),
            "avg_silence_ratio_percent": safe_mean(all_metrics["silence_ratios"]),
            "avg_wpm": safe_mean(all_metrics["wpms"]),
            "avg_focus_score": safe_mean(all_metrics["focus_scores"]),
            "avg_content_score": safe_mean(all_metrics["content_scores"]),
            "avg_communication_score": safe_mean(all_metrics["communication_scores"])
        }
        
        # ============== DETERMINE FINAL PRIORITY ==============
        priorities = all_metrics["priorities"]
        
        if not priorities:
            logger.warning("⚠️ No AI priorities available for final ranking")
            summary["final_priority"] = "UNKNOWN"
            summary["final_reason"] = "Không có dữ liệu AI để đánh giá"
        else:
            high_count = priorities.count("HIGH")
            low_count = priorities.count("LOW")
            
            if high_count >= len(priorities) * 0.6:
                final_priority = "HIGH"
                final_reason = "Ứng viên xuất sắc - Nhiều câu trả lời chất lượng cao"
            elif low_count >= len(priorities) * 0.5:
                final_priority = "LOW"
                final_reason = "Ứng viên yếu - Nhiều câu trả lời kém chất lượng"
            else:
                final_priority = "MEDIUM"
                final_reason = "Ứng viên trung bình - Cần xem xét kỹ hơn"
            
            summary["final_priority"] = final_priority
            summary["final_reason"] = final_reason
            summary["priority_distribution"] = {
                "HIGH": high_count,
                "MEDIUM": priorities.count("MEDIUM"),
                "LOW": low_count
            }
        
        # ============== GEMINI OVERALL SUMMARY ==============
        if GEMINI_API_KEY and gemini_model:
            try:
                # ✅ THÊM SAFETY SETTINGS
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
                
                model = genai.GenerativeModel(
                    'gemini-2.5-flash',
                    safety_settings=safety_settings
                )
                
                logger.info("🤖 Generating overall AI summary with Gemini...")
                
                # Ghép tất cả transcript lại
                full_transcript = ""
                for i, q in enumerate(questions, 1):
                    transcript_file = folder_path / q.get("transcriptFile", "")
                    if transcript_file.exists():
                        content = transcript_file.read_text(encoding='utf-8')
                        text_only = content.split("--- TIMESTAMPS ---")[0].strip()
                        full_transcript += f"\n\n--- QUESTION {i} ---\n{text_only}"
                
                # Truncate nếu quá dài (Gemini có limit context)
                max_transcript_length = 3000
                if len(full_transcript) > max_transcript_length:
                    full_transcript = full_transcript[:max_transcript_length] + "\n\n[...truncated...]"
                
                # Prompt cho overall summary
                overall_prompt = f"""You are reviewing a complete Data Analyst interview. Provide a concise overall assessment.

**INTERVIEW STATISTICS:**
{json.dumps(summary, indent=2, ensure_ascii=False)}

**FULL TRANSCRIPT (SAMPLE):**
{full_transcript}

**YOUR TASK:**
Provide a brief overall assessment (2-3 sentences in English) for HR to decide if they should watch the videos.

**OUTPUT FORMAT (JSON ONLY):**
{{
  "overall_summary": "2-3 sentence summary in English",
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "recommendation": "RECOMMEND" | "NEUTRAL" | "NOT_RECOMMEND"
}}

Respond with ONLY the JSON object. No markdown, no extra text."""

                # ✅ GỌI GEMINI
                response = await asyncio.to_thread(
                    model.generate_content,
                    overall_prompt,
                    generation_config={
                        "temperature": 0.3, 
                        "max_output_tokens": 2048,
                        "top_p": 0.95
                    }
                )
                
                # ✅ KIỂM TRA FINISH REASON
                if hasattr(response, 'candidates') and response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    
                    if finish_reason == 3:  # SAFETY blocked
                        logger.warning("⚠️ Overall summary blocked by safety filter")
                        summary["overall_ai_summary"] = {
                            "error": "SAFETY_BLOCKED",
                            "message": "Nội dung bị chặn bởi bộ lọc an toàn"
                        }
                        # Continue to save other data
                    elif finish_reason == 1:  # SUCCESS
                        response_text = response.text.strip()
                        logger.info(f"📥 Gemini overall response (first 200 chars): {response_text[:200]}")
                        
                        # Clean markdown code blocks
                        if "```json" in response_text:
                            response_text = response_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in response_text:
                            response_text = response_text.split("```")[1].split("```")[0].strip()
                        
                        # Parse JSON
                        overall_analysis = json.loads(response_text)
                        
                        # Validate
                        required_fields = ["overall_summary", "strengths", "weaknesses", "recommendation"]
                        for field in required_fields:
                            if field not in overall_analysis:
                                logger.warning(f"⚠️ Missing field in AI response: {field}")
                                overall_analysis[field] = "N/A" if field in ["overall_summary", "recommendation"] else []
                        
                        summary["overall_ai_summary"] = overall_analysis
                        logger.info(f"✅ Overall AI summary generated - Recommendation: {overall_analysis.get('recommendation', 'N/A')}")
                    else:
                        logger.warning(f"⚠️ Unexpected finish_reason: {finish_reason}")
                        summary["overall_ai_summary"] = {
                            "error": f"Unexpected finish_reason: {finish_reason}"
                        }
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parse error in overall summary: {e}")
                logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
                summary["overall_ai_summary"] = {
                    "error": "JSON parse error",
                    "raw_response": response_text[:500] if 'response_text' in locals() else None
                }
            
            except Exception as e:
                logger.error(f"❌ Error generating overall summary: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                summary["overall_ai_summary"] = {
                    "error": str(e)
                }
        else:
            logger.warning("⚠️ Gemini not available for final ranking")
            logger.warning(f"API Key exists: {bool(GEMINI_API_KEY)}, Model exists: {bool(gemini_model)}")
            summary["overall_ai_summary"] = {
                "error": "Gemini API not configured",
                "details": f"API_KEY={'SET' if GEMINI_API_KEY else 'MISSING'}, MODEL={'INITIALIZED' if gemini_model else 'NOT_INITIALIZED'}"
            }
        
        # ============== MERGE ALL TRANSCRIPTS ==============
        full_transcript_content = "=== FULL INTERVIEW TRANSCRIPT ===\n\n"
        full_transcript_content += f"Candidate: {metadata.get('userName')}\n"
        full_transcript_content += f"Session: {metadata.get('sessionStartTime')}\n\n"
        
        for i, q in enumerate(questions, 1):
            transcript_file = folder_path / q.get("transcriptFile", "")
            if transcript_file.exists():
                full_transcript_content += f"\n{'='*60}\n"
                full_transcript_content += transcript_file.read_text(encoding='utf-8') + "\n"
        
        full_transcript_file = folder_path / "FULL_TRANSCRIPT.txt"
        full_transcript_file.write_text(full_transcript_content, encoding='utf-8')
        logger.info(f"📄 Full transcript saved: {full_transcript_file.name}")
        
        # ============== UPDATE METADATA ==============
        metadata["final_ranking_summary"] = summary
        metadata["final_ranking_calculated_at"] = get_bangkok_timestamp()
        
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Final ranking calculated - Priority: {summary.get('final_priority', 'UNKNOWN')}")
        logger.info(f"📊 Summary: {json.dumps(summary, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        logger.error(f"❌ Error calculating final ranking: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
async def background_transcribe(folder_path: Path, video_path: Path, question_index: int, question_text: str, focus_score: int):
    """
    Hàm chạy background để:
    1. Transcribe video
    2. Phân tích Layer 1 (Quantitative)
    3. Phân tích Layer 2 (AI Semantic)
    4. Cập nhật metadata
    """
    try:
        logger.info(f"🎤 [Background] Starting analysis for Q{question_index}...")
        
        if WHISPER_MODEL is None:
            logger.warning(f"⚠️ Whisper model not loaded, skipping Q{question_index}")
            return
        
        # ============== STEP 1: TRANSCRIBE ==============
        audio_path = video_path.with_suffix('.wav')
        subprocess.run([
            'ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', str(audio_path), '-y', '-loglevel', 'error'
        ], check=False, timeout=60)
        
        logger.info(f"🎤 Transcribing Q{question_index}...")
        whisper_result = await asyncio.to_thread(
            WHISPER_MODEL.transcribe,
            str(audio_path),
            language='en',
            fp16=False
        )
        
        transcript_text = whisper_result['text'].strip()
        segments = whisper_result.get('segments', [])
        
        # Save transcript
        transcript_content = f"=== TRANSCRIPT Q{question_index} ===\n"
        transcript_content += f"Question: {question_text}\n"
        transcript_content += f"Time: {get_bangkok_timestamp()}\n\n"
        transcript_content += transcript_text + "\n\n"
        transcript_content += "--- TIMESTAMPS ---\n"
        
        for segment in segments:
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            transcript_content += f"[{start//60:02.0f}:{start%60:05.2f} -> {end//60:02.0f}:{end%60:05.2f}] {text}\n"
        
        transcript_file = video_path.parent / f"Q{question_index}_transcript.txt"
        transcript_file.write_text(transcript_content, encoding='utf-8')
        logger.info(f"📄 Transcript saved: {transcript_file.name}")
        
        # ============== STEP 2: LAYER 1 ANALYSIS ==============
        logger.info(f"📊 Running Layer 1 analysis for Q{question_index}...")
        
        layer1_metrics = analyze_layer1_metrics(
            transcript_data={'text': transcript_text, 'segments': segments},
            focus_score=focus_score
        )
        
        logger.info(f"✅ Layer 1 completed - Flagged: {layer1_metrics['flagged_as_bad']}")
        
        # ============== STEP 3: LAYER 2 AI ANALYSIS ==============
        logger.info(f"🤖 Running Layer 2 AI analysis for Q{question_index}...")
        
        layer2_result = await analyze_layer2_ai(
            question_text=question_text,
            transcript_text=transcript_text,
            layer1_metrics=layer1_metrics
        )
        
        logger.info(f"✅ Layer 2 completed - Priority: {layer2_result.get('priority', 'UNKNOWN')}")
        
        # ============== STEP 4: UPDATE METADATA ==============
        analysis_summary = {
            "transcriptionStatus": "completed",
            "transcriptFile": f"Q{question_index}_transcript.txt",
            "analyzed_at": get_bangkok_timestamp(),
    
        # Layer 1 metrics 
            "metrics": {
                "word_count": layer1_metrics["word_count"],
                "focus_score": layer1_metrics["focus_score"],
                "speaking_rate_wpm": layer1_metrics["speaking_rate_wpm"],
                "wpm_category": layer1_metrics["wpm_category"],
                "silence_ratio_percent": layer1_metrics["silence_ratio_percent"],
                "total_pause_seconds": layer1_metrics["total_pause_seconds"],
                "num_pauses": layer1_metrics["num_pauses"],
                "filler_density_percent": layer1_metrics["filler_density_percent"],
                "duration_seconds": layer1_metrics["duration_seconds"],
                "flagged_as_bad": layer1_metrics["flagged_as_bad"],
                "flag_reasons": layer1_metrics["flag_reasons"]
            },
    
            # Layer 2 AI evaluation
            "ai_evaluation": {
                "priority": layer2_result.get("priority", "UNKNOWN"),
                "reason": layer2_result.get("reason", "Không có đánh giá"),
                "content_score": layer2_result.get("content_score", 0),
                "communication_score": layer2_result.get("communication_score", 0),
                "overall_impression": layer2_result.get("overall_impression", "neutral"),
                "ai_available": layer2_result.get("ai_available", False)
            }
        }

        await update_metadata(folder_path, question_data={
            "index": question_index,
            **analysis_summary  # Merge tất cả fields vào question data
        })

        logger.info(f"✅ [Background] Full analysis completed for Q{question_index}")

        # Cleanup
        audio_path.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"❌ [Background] Analysis error Q{question_index}: {e}")
        try:
            await update_metadata(folder_path, question_data={
                "index": question_index,
                "transcriptionStatus": "failed",
                "analysisError": str(e)
            })
        except:
            pass
# --- Routes ---

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE_DIR / "static" / "index_first.html").read_text(encoding="utf-8")

@app.post("/api/verify-token")
async def verify_token(request: TokenRequest):
    if request.token in ADMIN_TOKENS:
        return {"ok": True, "role": "evaluator"}
    elif request.token in CANDIDATE_TOKENS:
        return {"ok": True, "role": "candidate"}
    else:
        raise HTTPException(status_code=401, detail="Invalid Token")

@app.post("/api/session/start")
async def session_start(request: SessionStartRequest):
    if request.token not in VALID_TOKENS: raise HTTPException(401, "Invalid token")
    if not request.userName or len(request.userName.strip()) == 0: raise HTTPException(400, "Username empty")
    
    folder_name = generate_folder_name(request.userName)
    folder_path = UPLOAD_DIR / folder_name
    
    if folder_path.exists():
        counter = 1
        while folder_path.exists():
            folder_name = f"{generate_folder_name(request.userName)}_{counter}"
            folder_path = UPLOAD_DIR / folder_name
            counter += 1
    
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Random câu hỏi
    random_questions = random.sample(QUESTION_POOL, 3)
    selected_questions = FIXED_QUESTIONS + random_questions
    await create_metadata(folder_path, request.userName, selected_questions)
    
    active_sessions[folder_name] = {
        "token": request.token,
        "started_at": datetime.now(BANGKOK_TZ),
        "questions": selected_questions,
        "uploads": set()
    }
    
    return {
        "ok": True,
        "folder": folder_name,
        "questions": selected_questions # Trả về list câu hỏi cho Frontend
    }

@app.post("/api/upload-one")
async def upload_one(
    background_tasks: BackgroundTasks,
    token: str = Form(...),
    folder: str = Form(...),
    questionIndex: int = Form(...),
    video: UploadFile = File(...),
    analysisData: str = Form(...)
):
    """
    FLOW: Upload -> Save WebM -> Verify -> Convert MP4 -> Save Meta -> Transcribe -> Update Meta
    """
    logger.info(f"Upload request - Folder: {folder}, Question: {questionIndex}")
    
    # 1. Validation
    if token not in VALID_TOKENS: raise HTTPException(401, "Invalid token")
    folder_path = UPLOAD_DIR / folder
    if not folder_path.exists(): raise HTTPException(404, "Session not found")
    if folder not in active_sessions: raise HTTPException(400, "Session inactive")
    if active_sessions[folder]["token"] != token: raise HTTPException(401, "Token mismatch")
    
    meta_file = folder_path / "meta.json"
    with meta_file.open("r", encoding="utf-8") as f:
        if json.load(f).get("sessionEnded", False):
            raise HTTPException(400, "Session finished")
    
    # 2. Save Video
    filename = f"Q{questionIndex}.webm"
    dest_path = folder_path / filename
    file_size = 0
    
    try:
        with dest_path.open("wb") as buffer:
            chunk_size = 1024 * 1024
            while chunk := await video.read(chunk_size):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(413, "File too large")
                buffer.write(chunk)
        
        if not verify_video_by_signature(dest_path):
            dest_path.unlink(missing_ok=True)
            raise HTTPException(415, "Invalid video format")
            
        # 3. Convert to MP4
        mp4_filename = convert_to_mp4(dest_path)

        #3.5 Lấy nội dung câu hỏi 
        question_text = "Unknown question"
        try:
            # Đọc lại file meta để lấy text câu hỏi
            if meta_file.exists():
                with meta_file.open("r", encoding="utf-8") as f:
                    meta_temp = json.load(f)
                    # Tìm câu hỏi có index tương ứng
                    q_def = next((q for q in meta_temp.get("interviewQuestions", []) if q["index"] == questionIndex), None)
                    if q_def:
                        question_text = q_def["text"]
        except Exception as e:
            logger.error(f"Failed to fetch question text: {e}")
        # 4. Initial Metadata (Pending Transcription)
        try: ai_metrics = json.loads(analysisData)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid analysisData JSON: {e}")
            ai_metrics = {}

        question_data = {
            "index": questionIndex,
            "text": question_text,
            "uploadedAt": get_bangkok_timestamp(),
            "filename": filename,
            "mp4_filename": mp4_filename,
            "size": file_size,
            "aiAnalysis": ai_metrics,
            "transcriptionStatus": "pending" # Đánh dấu đang xử lý
        }
        
        await update_metadata(folder_path, question_data=question_data)
        active_sessions[folder]["uploads"].add(questionIndex)
        
        # 5. Run Whisper Transcription (Async)
        if WHISPER_MODEL:
            with meta_file.open("r") as f:
                meta = json.load(f)
                question_text = next(
                    (q["text"] for q in meta.get("interviewQuestions", []) if q["index"] == questionIndex),
                "Unknown question"
                 )
        
            try: ai_metrics = json.loads(analysisData)
            except: ai_metrics = {}
            focus_score = ai_metrics.get("focusScore", 0)

            background_tasks.add_task(
                background_transcribe,  # ← TÊN HÀM MỚI
                folder_path, 
                dest_path, 
                questionIndex,
                question_text,
                focus_score
            )
            transcription_status = "processing"
        else:
            transcription_status = "disabled"

        # SỬA return statement:
        return {
            "ok": True,
            "savedAs": filename,
            "convertedTo": mp4_filename,
            "transcription": transcription_status,  # ← SỬA CHỖ NÀY
            "size": file_size
        }

    except HTTPException: raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        dest_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/api/session/finish")
async def session_finish(request: SessionFinishRequest,bg_tasks: BackgroundTasks ):
    if request.folder in active_sessions and active_sessions[request.folder]["token"] != request.token:
        raise HTTPException(401, "Token mismatch")
        
    folder_path = UPLOAD_DIR / request.folder
    bg_tasks.add_task(calculate_final_ranking, folder_path)
    await update_metadata(folder_path, finalize=True, questions_count=request.questionsCount)
    
    if request.folder in active_sessions:
        del active_sessions[request.folder]
    
    return {"ok": True}

# Endpoint để xem transcript (Optional)
@app.get("/api/transcript/{folder}/{question_index}")
async def get_transcript(folder: str, question_index: int, token: str):
    if token not in VALID_TOKENS: raise HTTPException(401, "Invalid token")
    
    transcript_file = UPLOAD_DIR / folder / f"Q{question_index}_transcript.txt"
    if not transcript_file.exists():
        raise HTTPException(404, "Transcript not found")
        
    return {
        "ok": True,
        "content": transcript_file.read_text(encoding='utf-8')
    }
@app.get("/api/admin/candidates")
async def get_candidates(token: str):
    # Check quyền
    if token not in ADMIN_TOKENS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    results = []
    # Quét thư mục uploads lấy danh sách
    if UPLOAD_DIR.exists():
        for folder in UPLOAD_DIR.iterdir():
            if folder.is_dir():
                # Lấy tên folder làm dữ liệu hiển thị tạm
                meta_file = folder / "meta.json"
                if not meta_file.exists():
                    continue
                try:
                    with meta_file.open("r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    parts = folder.name.split("_")
                    if len(parts) >= 5:
                        time_str = f"{parts[0]}/{parts[1]}/{parts[2]} {parts[3]}:{parts[4]}"
                    else:
                        time_str = "Unknown"
                    # Lấy điểm focus của câu đầu tiên (tạm thời để test UI)
                    qs = metadata.get("questions", [])
                    final_summary = metadata.get("final_ranking_summary", {})
                    ai_note = "No data yet"
                    priority_num = 2
                    if final_summary and "overall_ai_summary" in final_summary:
                        overall = final_summary["overall_ai_summary"]
                        if "overall_summary" in overall:
                            # Lấy đoạn văn tổng kết
                            full_text = overall["overall_summary"]
                            # Cắt ngắn nếu dài quá 20 từ để bảng đỡ bị vỡ
                            words = full_text.split()
                            ai_note = " ".join(words[:30]) + "..." if len(words) > 30 else full_text
                        
                        # Set Priority tổng
                        prio = final_summary.get("final_priority", "MEDIUM")
                        if prio == "HIGH": priority_num = 1
                        elif prio == "LOW": priority_num = 3
                    elif qs:
                        # Lọc ra các câu đã được AI chấm
                        evaluated_qs = [q for q in qs if q.get("ai_evaluation", {}).get("ai_available")]
                        
                        if evaluated_qs:
                            # Tìm xem có câu nào bị LOW (Yếu) không để cảnh báo ngay
                            bad_q = next((q for q in evaluated_qs if q["ai_evaluation"].get("priority") == "LOW"), None)
                            
                            target_q = bad_q if bad_q else evaluated_qs[-1]
                            
                            # --- ĐÂY LÀ DÒNG QUAN TRỌNG NHẤT ---
                            # Lấy trực tiếp trường "reason" từ JSON
                            ai_note = target_q["ai_evaluation"].get("reason", "AI processed but no reason provided")
                            
                            # Set priority theo câu đó
                            prio = target_q["ai_evaluation"].get("priority", "MEDIUM")
                            if prio == "HIGH": priority_num = 1
                            elif prio == "LOW": priority_num = 3
                        else:
                            ai_note = "Waiting for AI..."
                    # Tính focus trung bình
                    avg_focus = 0
                    if qs:
                        focus_scores = [q.get("aiAnalysis", {}).get("focusScore", 0) for q in qs]
                        avg_focus = sum(focus_scores) / len(focus_scores) if focus_scores else 0 
                    folder_url = f"/uploads/{folder.name}"
                    results.append({
                        "name": metadata.get("userName","Unknown"), 
                        "time": time_str,
                        "priority": priority_num,
                        "note": ai_note,
                        "folderUrl": folder_url,
                        "folder": folder.name,
                        "focus": round(avg_focus,1)
                    })
                    
                except Exception as e:
                    logger.error(f"Error reading {folder.name}: {e}")
                    continue
    
    results_sorted = sorted(results, key=lambda x: x.get("priority", 2))
    return {"candidates": results_sorted}
     
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)