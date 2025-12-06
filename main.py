from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import pytz
import asyncio
import json
import re
import logging
import uvicorn
import whisper 
import subprocess 
import asyncio  
from typing import Optional 
import random

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
#2: log
try:
    logger.info("Loading Whisper model (small)... This may take 1-2 minutes on first run")
    logger.info("Model will be downloaded (~244MB) if not exists")
    
    # Load model "small" - balance giữa accuracy và speed
    # Options: "tiny" (39MB), "base" (74MB), "small" (244MB), "medium" (769MB), "large" (1550MB)
    WHISPER_MODEL = whisper.load_model("small")
    
    logger.info("Whisper model loaded successfully!")
    logger.info(f"Model info: small (244MB, ~90% accuracy)")
    
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    logger.error("Transcription will be disabled. Install: pip install openai-whisper")
    WHISPER_MODEL = None

# --- Configuration ---
app = FastAPI(title="Web Interview Recorder", version="1.0")

# MỤC ĐÍCH: Cho phép frontend gọi API từ domain khác (CORS)
# - allow_origins=["*"]: Cho phép mọi domain (dev only, prod nên chỉ định cụ thể)
# - Cần thiết để HTML có thể gọi API
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

# MỤC ĐÍCH: Serve các file tĩnh (HTML, CSS, JS)
# - /static route sẽ map đến thư mục static/
# - Không serve uploads để bảo mật
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# --- Configuration ---
# MỤC ĐÍCH: Định nghĩa các token hợp lệ (trong thực tế nên dùng database)
VALID_TOKENS = {"Thịnh", "Hồng", "Thành", "Luân"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_MIME_TYPES = {"video/webm", "video/mp4"}
BANGKOK_TZ = pytz.timezone('Asia/Bangkok')

# MỤC ĐÍCH: Theo dõi các session đang active trong memory
# - Key: tên folder, Value: thông tin session (token, thời gian, uploads)
# - Để kiểm tra session còn active không và ngăn upload sau khi finish
active_sessions = {}

# MỤC ĐÍCH: Lock để tránh race condition khi nhiều request cập nhật metadata cùng lúc
metadata_locks = {}

# --- Pydantic Models ---
# MỤC ĐÍCH: Validate dữ liệu input từ client
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
    """
    MỤC ĐÍCH: Làm sạch username để tránh directory traversal attack
    - Xóa ký tự đặc biệt nguy hiểm (/, \, .., etc)
    - Thay space bằng underscore
    - Lowercase và giới hạn 50 ký tự
    """
    safe_name = re.sub(r'[^\w\s-]', '', username)
    safe_name = re.sub(r'\s+', '_', safe_name)
    safe_name = safe_name.strip('_')
    return safe_name.lower()[:50]

def get_bangkok_timestamp() -> str:
    """
    MỤC ĐÍCH: Lấy timestamp theo timezone Asia/Bangkok (ISO 8601 format)
    - Theo yêu cầu project phải dùng Bangkok timezone
    """
    return datetime.now(BANGKOK_TZ).isoformat()

def generate_folder_name(username: str) -> str:
    """
    MỤC ĐÍCH: Tạo tên folder theo format DD_MM_YYYY_HH_mm_ten_user
    - Theo yêu cầu project, timezone Asia/Bangkok
    """
    now = datetime.now(BANGKOK_TZ)
    sanitized = sanitize_username(username)
    return f"{now.strftime('%d_%m_%Y_%H_%M')}_{sanitized}"

def verify_video_by_signature(file_path: Path) -> bool:
    """
    MỤC ĐÍCH: Verify file thực sự là video bằng cách check magic bytes (file signature)
    - Không dùng python-magic vì khó cài trên Windows
    - Check 32 bytes đầu tiên của file
    - Hỗ trợ: WebM, MP4, AVI, MOV
    """
    try:
        with file_path.open('rb') as f:
            header = f.read(32)
            
            # WebM: magic bytes \x1a\x45\xdf\xa3
            if header[:4] == b'\x1a\x45\xdf\xa3':
                logger.info(f"Detected WebM file: {file_path.name}")
                return True
            
            # MP4: có 'ftyp' trong 12 bytes đầu
            if b'ftyp' in header[:12]:
                logger.info(f"Detected MP4/M4V file: {file_path.name}")
                return True
            
            # AVI: bắt đầu bằng 'RIFF' và có 'AVI ' ở byte 8-11
            if header[:4] == b'RIFF' and header[8:12] == b'AVI ':
                logger.info(f"Detected AVI file: {file_path.name}")
                return True
            
            # MOV/QuickTime: có 'moov' hoặc 'mdat' atom
            if b'moov' in header or b'mdat' in header:
                logger.info(f"Detected QuickTime file: {file_path.name}")
                return True
            
            logger.warning(f"Unknown file signature: {header[:16].hex()}")
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
            language='vi',
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
    """
    MỤC ĐÍCH: Cập nhật metadata file sau mỗi upload hoặc khi finish
    - Thêm thông tin question sau mỗi upload thành công
    - Set sessionEnded = True khi finish
    - Dùng lock để tránh 2 request cùng ghi file đè lên nhau
    """
    meta_file = folder_path / "meta.json"
    
    if not meta_file.exists():
        logger.error(f"Metadata file not found: {meta_file}")
        raise HTTPException(status_code=404, detail="Metadata file not found")
    
    # MỤC ĐÍCH: Mỗi folder có 1 lock riêng để tránh race condition
    folder_key = str(folder_path)
    if folder_key not in metadata_locks:
        metadata_locks[folder_key] = asyncio.Lock()
    
    async with metadata_locks[folder_key]:
        with meta_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # CASE 1: Thêm question data sau mỗi upload
        if question_data:
            metadata["questions"].append(question_data)
            logger.info(f"➕ Added question {question_data['index']} to metadata: {folder_path.name}")
        
        # CASE 2: Finalize session (gọi từ /api/session/finish)
        if finalize:
            metadata["sessionEnded"] = True
            metadata["sessionEndTime"] = get_bangkok_timestamp()
            
            if questions_count is not None:
                metadata["questionsCount"] = questions_count
            
            # ===== MỚI THÊM: Count số transcript files =====
            # Mục đích: Kiểm tra có bao nhiêu transcript đã được generate
            # Pattern: Q*_transcript.txt (VD: Q1_transcript.txt, Q2_transcript.txt)
            transcript_files = list(folder_path.glob("*_transcript.txt"))
            transcript_count = len(transcript_files)
            
            metadata["transcriptsGenerated"] = transcript_count
            
            logger.info(f"🏁 Finalized session: {folder_path.name}")
            logger.info(f"📊 Questions answered: {questions_count}")
            logger.info(f"📝 Transcripts generated: {transcript_count}/{questions_count}")
            
            # WARNING: Nếu không có transcript nào
            if transcript_count == 0:
                logger.warning(f"⚠️  No transcripts generated for this session!")
        
        # GHI lại vào file
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

async def background_transcribe(folder_path: Path, video_path: Path, question_index: int):
    """
    Hàm chạy background để transcribe video
    Chạy SONG SONG với response trả về client
    """
    try:
        logger.info(f"🎤 [Background] Starting transcription for Q{question_index}...")
        
        if WHISPER_MODEL is None:
            logger.warning(f"⚠️ Whisper model not loaded, skipping Q{question_index}")
            return
        
        # Chạy Whisper
        transcript_text = await transcribe_video_whisper(video_path, question_index)
        
        if transcript_text:
            # Update metadata khi xong
            await update_metadata(folder_path, question_data={
                "index": question_index,
                "transcriptionStatus": "completed",
                "transcriptFile": f"Q{question_index}_transcript.txt"
            })
            logger.info(f"✅ [Background] Transcription completed for Q{question_index}")
        else:
            # Đánh dấu failed
            await update_metadata(folder_path, question_data={
                "index": question_index,
                "transcriptionStatus": "failed"
            })
            logger.error(f"❌ [Background] Transcription failed for Q{question_index}")
            
    except Exception as e:
        logger.error(f"❌ [Background] Transcription error Q{question_index}: {e}")
        # Đánh dấu failed trong metadata
        try:
            await update_metadata(folder_path, question_data={
                "index": question_index,
                "transcriptionStatus": "failed",
                "transcriptionError": str(e)
            })
        except:
            pass


# --- Home Page ---
@app.get("/", response_class=HTMLResponse)
def home():
    """
    MỤC ĐÍCH: Serve trang HTML chính
    - Đọc file static/index.html và trả về
    - HTML này chứa code getUserMedia để xin quyền camera/mic
    """
    html = (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)

# --- API Endpoints ---

@app.post("/api/verify-token")
async def verify_token(request: TokenRequest):
    """
    MỤC ĐÍCH: Verify token có hợp lệ không
    - Bước đầu tiên trước khi cho phép start session
    - Server-side validation (không tin client)
    """
    logger.info(f"Token verification attempt: {request.token[:4]}...")
    
    if request.token not in VALID_TOKENS:
        logger.warning(f"Invalid token attempt: {request.token[:4]}...")
        raise HTTPException(status_code=401, detail="Invalid token")
    
    logger.info("Token verified successfully")
    return {"ok": True}

@app.post("/api/session/start")
async def session_start(request: SessionStartRequest):
    """
    MỤC ĐÍCH: Bắt đầu 1 session phỏng vấn mới
    - Verify token
    - Tạo folder theo format DD_MM_YYYY_HH_mm_ten_user
    - Tạo file meta.json
    - Track session trong active_sessions
    """
    logger.info(f"Session start request - Token: {request.token[:4]}..., User: {request.userName}")
    
    if request.token not in VALID_TOKENS:
        logger.warning("Invalid token for session start")
        raise HTTPException(status_code=401, detail="Invalid token")
    
    if not request.userName or len(request.userName.strip()) == 0:
        logger.warning("Empty username provided")
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    
    folder_name = generate_folder_name(request.userName)
    folder_path = UPLOAD_DIR / folder_name
    
    # MỤC ĐÍCH: Tránh trùng tên folder (nếu cùng phút có 2 người cùng tên)
    if folder_path.exists():
        counter = 1
        while folder_path.exists():
            folder_name = f"{generate_folder_name(request.userName)}_{counter}"
            folder_path = UPLOAD_DIR / folder_name
            counter += 1
    
    folder_path.mkdir(parents=True, exist_ok=True)
    random_questions = random.sample(QUESTION_POOL, 3)
    selected_questions = FIXED_QUESTIONS + random_questions
    
    await create_metadata(folder_path, request.userName, selected_questions)
    
    # MỤC ĐÍCH: Track session để check khi upload và finish
    active_sessions[folder_name] = {
        "token": request.token,
        "started_at": datetime.now(BANGKOK_TZ),
        "uploads": set()
    }
    
    logger.info(f"Session started successfully: {folder_name}")
    
    return {
        "ok": True,
        "folder": folder_name,
        "questions": selected_questions
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
    with meta_file.open("r") as f:
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

        # 4. Initial Metadata (Pending Transcription)
        try: ai_metrics = json.loads(analysisData)
        except: ai_metrics = {}

        question_data = {
            "index": questionIndex,
            "uploadedAt": get_bangkok_timestamp(),
            "filename": filename,
            "mp4_filename": mp4_filename,
            "size": file_size,
            "aiAnalysis": ai_metrics,
            "transcriptionStatus": "pending" # Đánh dấu đang xử lý
        }
        
        await update_metadata(folder_path, question_data=question_data)
        active_sessions[folder]["uploads"].add(questionIndex)
        
        # 5. ✅ ĐẨY TRANSCRIPTION VÀO BACKGROUND
        # Trả response NGAY cho user, không chờ transcribe
        if WHISPER_MODEL:
            background_tasks.add_task(
                background_transcribe, 
                folder_path, 
                dest_path, 
                questionIndex
            )
            transcription_status = "processing"  # Đang xử lý background
        else:
            transcription_status = "disabled"  # Whisper không khả dụng
        
        logger.info(f"✅ Upload successful: {filename} - Transcription queued in background")
        
        return {
            "ok": True,
            "savedAs": filename,
            "convertedTo": mp4_filename,
            "transcription": transcription_status,  # "processing" hoặc "disabled"
            "size": file_size
        }

    except HTTPException: raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        dest_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Upload failed: {str(e)}")

    
@app.post("/api/session/finish")
async def session_finish(request: SessionFinishRequest):
    """
    MỤC ĐÍCH: Kết thúc session phỏng vấn
    - Set sessionEnded = True trong metadata
    - Update questionsCount
    - Remove khỏi active_sessions
    - Sau khi finish thì không cho upload nữa
    """
    logger.info(f"Session finish request - Folder: {request.folder}")
    
    if request.token not in VALID_TOKENS:
        logger.warning("Invalid token for session finish")
        raise HTTPException(status_code=401, detail="Invalid token")
    
    folder_path = UPLOAD_DIR / request.folder
    if not folder_path.exists():
        logger.error(f"Session folder not found: {request.folder}")
        raise HTTPException(status_code=404, detail="Session folder not found")
    
    # MỤC ĐÍCH: Verify token khớp với session
    if request.folder in active_sessions:
        if active_sessions[request.folder]["token"] != request.token:
            logger.warning("Token mismatch for session finish")
            raise HTTPException(status_code=401, detail="Token does not match session")
    
    await update_metadata(folder_path, finalize=True, questions_count=request.questionsCount)
    
    # MỤC ĐÍCH: Remove khỏi active sessions để ngăn upload sau finish
    if request.folder in active_sessions:
        del active_sessions[request.folder]
    
    logger.info(f"Session finished successfully: {request.folder}")
    
    return {"ok": True}

@app.get("/api/sessions")
async def list_sessions(token: str):
    """
    MỤC ĐÍCH: Debug endpoint - list tất cả sessions
    - Yêu cầu token để bảo mật
    - Duyệt qua thư mục uploads và đọc meta.json
    """
    if token not in VALID_TOKENS:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    sessions = []
    for folder in UPLOAD_DIR.iterdir():
        if folder.is_dir():
            meta_file = folder / "meta.json"
            if meta_file.exists():
                with meta_file.open("r") as f:
                    metadata = json.load(f)
                sessions.append({
                    "folder": folder.name,
                    "userName": metadata.get("userName"),
                    "sessionStartTime": metadata.get("sessionStartTime"),
                    "questionsCount": len(metadata.get("questions", [])),
                    "sessionEnded": metadata.get("sessionEnded", False)
                })
    
    return {"count": len(sessions), "sessions": sessions}



if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="server.key", ssl_certfile="server.crt")
    
    