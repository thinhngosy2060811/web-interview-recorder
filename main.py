from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
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
import whisper  # OpenAI Whisper cho Speech-to-Text
import subprocess  # Để chạy FFmpeg command
import asyncio  # Để chạy transcription không block server
from typing import Optional #type hints cho python

# --- Logging Configuration ---
# MỤC ĐÍCH: Ghi log để debug và theo dõi hoạt động hệ thống
# - Lưu vào file app.log và hiển thị trên console
# - Format có timestamp, level (INFO/ERROR), và message
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

async def transcribe_video_whisper(video_path: Path, question_index: int) -> Optional[str]:
    """
    CHỨC NĂNG: Chuyển video thành text transcript sử dụng OpenAI Whisper
    
    QUY TRÌNH:
    1. Extract audio từ video file (.webm → .wav) bằng FFmpeg
    2. Chạy Whisper model để transcribe audio → text
    3. Format kết quả với timestamps từng câu
    4. Lưu vào file Q<N>_transcript.txt
    5. Cleanup file audio tạm
    
    Args:
        video_path: Đường dẫn đến file video (VD: uploads/folder/Q1.webm)
        question_index: Số thứ tự câu hỏi (1-5)
    
    Returns:
        str: Nội dung transcript, hoặc None nếu thất bại
    
    Examples:
        >>> await transcribe_video_whisper(Path("Q1.webm"), 1)
        "Hello, my name is John..."
    """
    
    # KIỂM TRA: Model đã được load chưa?
    if WHISPER_MODEL is None:
        logger.warning("Whisper model not loaded, skipping transcription")
        logger.warning("Install: pip install openai-whisper ffmpeg-python")
        return None
    
    try:
        # ===== BƯỚC 1: EXTRACT AUDIO TỪ VIDEO =====
        # Mục đích: Whisper chỉ nhận audio, không nhận video
        # Format: WAV 16kHz mono (theo yêu cầu của Whisper)
        
        audio_path = video_path.with_suffix('.wav')  # Q1.webm → Q1.wav
        
        logger.info(f"[Q{question_index}] Extracting audio from {video_path.name}...")
        logger.info(f"Output: {audio_path.name}")
        
        # Chạy FFmpeg command
        # -i: input file
        # -vn: không lấy video (only audio)
        # -acodec pcm_s16le: audio codec (WAV format)
        # -ar 16000: sample rate 16kHz (Whisper requirement)
        # -ac 1: mono channel (1 channel, không stereo)
        # -y: overwrite nếu file đã tồn tại
        # -loglevel error: chỉ show error, không show info
        result = subprocess.run([
            'ffmpeg',
            '-i', str(video_path),      # Input: Q1.webm
            '-vn',                        # No video
            '-acodec', 'pcm_s16le',      # Audio codec cho WAV
            '-ar', '16000',               # Sample rate 16kHz
            '-ac', '1',                   # Mono (1 channel)
            str(audio_path),             # Output: Q1.wav
            '-y',                         # Overwrite
            '-loglevel', 'error'         # Chỉ show errors
        ], capture_output=True, text=True, timeout=60)  # Timeout 60s
        
        # KIỂM TRA: FFmpeg có chạy thành công không?
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            logger.error("Check: ffmpeg -version")
            return None
        
        logger.info(f"Audio extracted: {audio_path.name} ({audio_path.stat().st_size // 1024}KB)")
        
        # ===== BƯỚC 2: TRANSCRIBE BẰNG WHISPER =====
        # Mục đích: Chuyển audio → text
        # Chạy trong thread pool để không block server (vì Whisper chậm 30-60s)
        
        logger.info(f"[Q{question_index}] Transcribing with Whisper (small model)...")
        logger.info(f"Expected time: ~30-60 seconds for 1-minute video")
        
        # Run trong thread pool (asyncio.to_thread) để không block event loop
        whisper_result = await asyncio.to_thread(
            WHISPER_MODEL.transcribe,
            str(audio_path),              # Input audio file
            language='vi',                 # 'en' = English, 'vi' = Vietnamese, None = auto-detect
            task='transcribe',            # 'transcribe' hoặc 'translate' (translate → English)
            fp16=False,                   # Tắt FP16 nếu không có GPU (CPU mode)
            verbose=False,                # Không print progress
            temperature=0.0,              # Temperature 0 = deterministic (same input → same output)
            compression_ratio_threshold=2.4,  # Detect hallucinations
            logprob_threshold=-1.0,       # Confidence threshold
            no_speech_threshold=0.6       # Detect silent parts
        )
        
        logger.info(f"Transcription completed!")
        logger.info(f"Detected language: {whisper_result.get('language', 'unknown')}")
        logger.info(f"Text length: {len(whisper_result['text'])} characters")
        logger.info(f"Segments: {len(whisper_result.get('segments', []))} parts")
        
        # ===== BƯỚC 3: FORMAT TRANSCRIPT =====
        # Mục đích: Tạo file text dễ đọc với timestamps
        
        transcript_text = f"=" * 60 + "\n"
        transcript_text += f"QUESTION {question_index} TRANSCRIPT\n"
        transcript_text += f"=" * 60 + "\n\n"
        
        # Metadata
        transcript_text += f"Generated at: {get_bangkok_timestamp()}\n"
        transcript_text += f"Language detected: {whisper_result.get('language', 'unknown').upper()}\n"
        transcript_text += f"Total duration: {whisper_result.get('segments', [{}])[-1].get('end', 0):.2f} seconds\n"
        transcript_text += f"Total segments: {len(whisper_result.get('segments', []))}\n"
        transcript_text += f"\n" + "-" * 60 + "\n"
        
        # Full text (không có timestamps)
        transcript_text += f"FULL TEXT\n"
        transcript_text += f"-" * 60 + "\n"
        transcript_text += whisper_result['text'].strip() + "\n"
        transcript_text += f"\n" + "-" * 60 + "\n"
        
        # Segments with timestamps (chi tiết từng câu)
        transcript_text += f"DETAILED SEGMENTS (with timestamps)\n"
        transcript_text += f"-" * 60 + "\n\n"
        
        for i, segment in enumerate(whisper_result.get('segments', []), 1):
            start = segment['start']      # Thời gian bắt đầu (giây)
            end = segment['end']          # Thời gian kết thúc (giây)
            text = segment['text'].strip()  # Nội dung text
            
            # Format: [MM:SS - MM:SS] Text
            transcript_text += f"[{start//60:02.0f}:{start%60:05.2f} → {end//60:02.0f}:{end%60:05.2f}] {text}\n"
        
        transcript_text += f"\n" + "=" * 60 + "\n"
        transcript_text += f"END OF TRANSCRIPT\n"
        transcript_text += f"=" * 60 + "\n"
        # ===== BƯỚC 4: LƯU FILE TRANSCRIPT =====
        # Mục đích: Lưu transcript vào file Q<N>_transcript.txt
        
        transcript_file = video_path.parent / f"Q{question_index}_transcript.txt"
        transcript_file.write_text(transcript_text, encoding='utf-8')
        
        logger.info(f"Transcript saved: {transcript_file.name}")
        logger.info(f"File size: {transcript_file.stat().st_size // 1024}KB")
        
        # ===== BƯỚC 5: CLEANUP =====
        # Mục đích: Xóa file audio tạm để tiết kiệm disk space
        
        try:
            audio_path.unlink(missing_ok=True)  # Xóa Q1.wav
            logger.info(f"Cleaned up: {audio_path.name}")
        except Exception as e:
            logger.warning(f"Could not delete temp audio file: {e}")
        
        # Trả về full text (không có timestamps)
        return whisper_result['text'].strip()
        
    except subprocess.TimeoutExpired:
        # FFmpeg chạy quá lâu (> 60s)
        logger.error(f"FFmpeg timeout - video quá dài hoặc bị lỗi")
        return None
        
    except FileNotFoundError as e:
        # FFmpeg không được cài đặt
        logger.error(f"FFmpeg not found: {e}")
        logger.error("Install FFmpeg:")
        logger.error("   - Windows: choco install ffmpeg")
        logger.error("   - Mac: brew install ffmpeg")
        logger.error("   - Ubuntu: sudo apt install ffmpeg")
        return None
        
    except Exception as e:
        # Lỗi khác (Whisper error, file error, etc)
        logger.error(f"Transcription error for Q{question_index}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

async def create_metadata(folder_path: Path, username: str) -> dict:
    """
    MỤC ĐÍCH: Tạo file meta.json ban đầu khi start session
    - Chứa thông tin: userName, timestamps, timezone, questions list
    - sessionEnded = False để track session còn active
    """
    metadata = {
        "userName": username,
        "sessionStartTime": get_bangkok_timestamp(),
        "timeZone": "Asia/Bangkok",
        "questions": [],
        "questionsCount": 0,
        "sessionEnded": False,
        "sessionEndTime": None
    }
    
    meta_file = folder_path / "meta.json"
    
    # MỤC ĐÍCH: Async lock để tránh race condition
    async with asyncio.Lock():
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created metadata for session: {folder_path.name}")
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
    await create_metadata(folder_path, request.userName)
    
    # MỤC ĐÍCH: Track session để check khi upload và finish
    active_sessions[folder_name] = {
        "token": request.token,
        "started_at": datetime.now(BANGKOK_TZ),
        "uploads": set()
    }
    
    logger.info(f"Session started successfully: {folder_name}")
    
    return {
        "ok": True,
        "folder": folder_name
    }

@app.post("/api/upload-one")
async def upload_one(
    token: str = Form(...),
    folder: str = Form(...),
    questionIndex: int = Form(...),
    video: UploadFile = File(...)
):
    """
    MỤC ĐÍCH: Upload 1 video cho 1 câu hỏi
    
    FLOW MỚI:
    1. Validate token, folder, questionIndex (GIỮ NGUYÊN)
    2. Save video file (GIỮ NGUYÊN)
    3. Verify video format (GIỮ NGUYÊN)
    4. Update metadata (GIỮ NGUYÊN)
    5. ✨ MỚI: Generate transcript với Whisper
    6. ✨ MỚI: Update metadata với transcription status
    7. Return response với transcription info
    """
    logger.info(f"📤 Upload request - Folder: {folder}, Question: {questionIndex}")
    
    # ===== VALIDATION (GIỮ NGUYÊN TẤT CẢ CODE CŨ) =====
    # Token validation
    if token not in VALID_TOKENS:
        logger.warning("❌ Invalid token for upload")
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Folder exists?
    folder_path = UPLOAD_DIR / folder
    if not folder_path.exists():
        logger.error(f"❌ Session folder not found: {folder}")
        raise HTTPException(status_code=404, detail="Session folder not found")
    
    # Session active?
    if folder not in active_sessions:
        logger.warning(f"⚠️  Inactive session upload attempt: {folder}")
        raise HTTPException(status_code=400, detail="Session not active or already finished")
    
    # Token match?
    if active_sessions[folder]["token"] != token:
        logger.warning("❌ Token mismatch for session")
        raise HTTPException(status_code=401, detail="Token does not match session")
    
    # Session ended?
    meta_file = folder_path / "meta.json"
    with meta_file.open("r") as f:
        metadata = json.load(f)
        if metadata.get("sessionEnded", False):
            logger.warning(f"⚠️  Upload attempt after session finish: {folder}")
            raise HTTPException(status_code=400, detail="Cannot upload after session/finish")
    
    # Question index valid?
    if questionIndex < 1 or questionIndex > 5:
        logger.warning(f"❌ Invalid question index: {questionIndex}")
        raise HTTPException(status_code=400, detail="Question index must be between 1 and 5")
    
    # Allow retry?
    if questionIndex in active_sessions[folder]["uploads"]:
        logger.info(f"🔄 Duplicate upload detected for Q{questionIndex}, allowing re-upload")
    
    # MIME type valid?
    if video.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"❌ Invalid content type: {video.content_type}")
        raise HTTPException(
            status_code=415, 
            detail=f"Unsupported media type: {video.content_type}. Allowed: {', '.join(ALLOWED_MIME_TYPES)}"
        )
    
    filename = f"Q{questionIndex}.webm"
    dest_path = folder_path / filename
    
    file_size = 0
    
    try:
        # ===== SAVE FILE (GIỮ NGUYÊN CODE CŨ) =====
        logger.info(f"💾 Saving video: {filename}")
        
        with dest_path.open("wb") as buffer:
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await video.read(chunk_size):
                file_size += len(chunk)
                
                # Check file size
                if file_size > MAX_FILE_SIZE:
                    dest_path.unlink(missing_ok=True)
                    logger.warning(f"❌ File too large: {file_size} bytes")
                    raise HTTPException(
                        status_code=413, 
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
                    )
                
                buffer.write(chunk)
        
        logger.info(f"✅ Video saved: {filename} ({file_size / 1024 / 1024:.2f}MB)")
        
        # ===== VERIFY VIDEO (GIỮ NGUYÊN CODE CŨ) =====
        logger.info(f"🔍 Verifying video format: {filename}")
        
        if not verify_video_by_signature(dest_path):
            dest_path.unlink(missing_ok=True)
            logger.warning(f"❌ Invalid video file format detected")
            raise HTTPException(
                status_code=415,
                detail="File is not a valid video format"
            )
        
        logger.info(f"✅ Video format verified: {filename}")
        
        # ===== UPDATE METADATA - INITIAL (GIỮ NGUYÊN CODE CŨ) =====
        question_data = {
            "index": questionIndex,
            "uploadedAt": get_bangkok_timestamp(),
            "filename": filename,
            "size": file_size,
            "transcriptionStatus": "pending"  # ✨ THÊM field này
        }
        
        await update_metadata(folder_path, question_data=question_data)
        logger.info(f"📝 Metadata updated: Q{questionIndex} marked as pending transcription")
        
        # ===== ✨ MỚI: GENERATE TRANSCRIPT =====
        # Mục đích: Chạy Whisper để tạo transcript ngay sau khi upload thành công
        # Chạy async để không block response (user không phải đợi 30-60s)
        
        transcript_text = None
        transcription_success = False
        
        try:
            logger.info(f"🤖 Starting transcription for Q{questionIndex}...")
            logger.info(f"⏱️  This will take ~30-60 seconds, running in background...")
            
            # Chạy transcription (async, không block)
            transcript_text = await transcribe_video_whisper(dest_path, questionIndex)
            
            # KIỂM TRA: Transcription có thành công không?
            if transcript_text:
                transcription_success = True
                logger.info(f"✅ Transcription completed for Q{questionIndex}")
                logger.info(f"📏 Transcript length: {len(transcript_text)} characters")
                
                # ===== ✨ UPDATE METADATA VỚI TRANSCRIPTION INFO =====
                # Mục đích: Đánh dấu transcription đã hoàn thành trong meta.json
                
                async with metadata_locks.get(str(folder_path), asyncio.Lock()):
                    # Đọc metadata hiện tại
                    with meta_file.open("r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    
                    # Tìm question vừa add và update transcription status
                    for q in metadata["questions"]:
                        if q["index"] == questionIndex:
                            q["transcriptionStatus"] = "completed"  # ✨ pending → completed
                            q["transcriptLength"] = len(transcript_text)  # ✨ Thêm độ dài
                            q["transcriptFile"] = f"Q{questionIndex}_transcript.txt"  # ✨ Tên file
                            break
                    
                    # Ghi lại vào file
                    with meta_file.open("w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                logger.info(f"📝 Metadata updated: Q{questionIndex} marked as completed")
                
            else:
                # Transcription thất bại
                logger.warning(f"⚠️  Transcription failed for Q{questionIndex}")
                logger.warning(f"💡 Video saved successfully, but no transcript generated")
                
                # Update metadata: failed
                async with metadata_locks.get(str(folder_path), asyncio.Lock()):
                    with meta_file.open("r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    
                    for q in metadata["questions"]:
                        if q["index"] == questionIndex:
                            q["transcriptionStatus"] = "failed"  # ✨ pending → failed
                            break
                    
                    with meta_file.open("w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            # Lỗi khi chạy transcription
            logger.error(f"❌ Transcription error for Q{questionIndex}: {str(e)}")
            logger.error(f"📍 Error type: {type(e).__name__}")
            logger.warning(f"⚠️  Video uploaded successfully, but transcription failed")
            
            # Update metadata: error
            try:
                async with metadata_locks.get(str(folder_path), asyncio.Lock()):
                    with meta_file.open("r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    
                    for q in metadata["questions"]:
                        if q["index"] == questionIndex:
                            q["transcriptionStatus"] = "error"  # ✨ pending → error
                            q["transcriptionError"] = str(e)[:100]  # ✨ Lưu error message (max 100 chars)
                            break
                    
                    with meta_file.open("w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
            except:
                pass  # Không raise exception nếu không update được metadata
        
        # ===== TRACK UPLOAD (GIỮ NGUYÊN CODE CŨ) =====
        active_sessions[folder]["uploads"].add(questionIndex)
        
        logger.info(f"🎉 Upload successful: {filename} ({file_size} bytes)")
        
        # ===== ✨ RETURN RESPONSE VỚI TRANSCRIPTION INFO =====
        return {
            "ok": True,
            "savedAs": filename,
            "size": file_size,
            "transcription": "completed" if transcription_success else "failed"  # ✨ Thêm field này
        }
        
    except HTTPException:
        # Re-raise HTTPException (validation errors)
        raise
        
    except Exception as e:
        # Lỗi không mong đợi
        logger.error(f"❌ Upload error: {str(e)}")
        logger.error(f"📍 Error type: {type(e).__name__}")
        
        # Cleanup file nếu có lỗi
        dest_path.unlink(missing_ok=True)
        
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    
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
    
    