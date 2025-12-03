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
VALID_TOKENS = {"demo123", "test456", "student2024"}
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
    return f"{now.strftime('%d_%m_%Y_%H_%M')}_ten_{sanitized}"

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
        
        if question_data:
            metadata["questions"].append(question_data)
            logger.info(f"Added question {question_data['index']} to metadata: {folder_path.name}")
        
        if finalize:
            metadata["sessionEnded"] = True
            metadata["sessionEndTime"] = get_bangkok_timestamp()
            if questions_count is not None:
                metadata["questionsCount"] = questions_count
            logger.info(f"Finalized session: {folder_path.name}")
        
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
    html = (BASE_DIR / "static" / "index(1).html").read_text(encoding="utf-8")
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
    MỤC ĐÍCH: Upload 1 video cho 1 câu hỏi (per-question upload)
    - Verify token và session còn active
    - Check file size (max 50MB)
    - Check MIME type (video/webm, video/mp4)
    - Verify bằng magic bytes
    - Lưu file với tên Q1.webm, Q2.webm, ...
    - Update metadata
    - Cho phép retry (overwrite file cũ)
    """
    logger.info(f"Upload request - Folder: {folder}, Question: {questionIndex}")
    
    if token not in VALID_TOKENS:
        logger.warning("Invalid token for upload")
        raise HTTPException(status_code=401, detail="Invalid token")
    
    folder_path = UPLOAD_DIR / folder
    if not folder_path.exists():
        logger.error(f"Session folder not found: {folder}")
        raise HTTPException(status_code=404, detail="Session folder not found")
    
    # MỤC ĐÍCH: Kiểm tra session còn active không
    if folder not in active_sessions:
        logger.warning(f"Inactive session upload attempt: {folder}")
        raise HTTPException(status_code=400, detail="Session not active or already finished")
    
    # MỤC ĐÍCH: Verify token khớp với token đã dùng start session
    if active_sessions[folder]["token"] != token:
        logger.warning("Token mismatch for session")
        raise HTTPException(status_code=401, detail="Token does not match session")
    
    # MỤC ĐÍCH: Ngăn upload sau khi đã gọi finish
    meta_file = folder_path / "meta.json"
    with meta_file.open("r") as f:
        metadata = json.load(f)
        if metadata.get("sessionEnded", False):
            logger.warning(f"Upload attempt after session finish: {folder}")
            raise HTTPException(status_code=400, detail="Cannot upload after session/finish")
    
    # MỤC ĐÍCH: Validate questionIndex từ 1-5 theo yêu cầu project
    if questionIndex < 1 or questionIndex > 5:
        logger.warning(f"Invalid question index: {questionIndex}")
        raise HTTPException(status_code=400, detail="Question index must be between 1 and 5")
    
    # MỤC ĐÍCH: Cho phép upload lại cùng câu hỏi (retry mechanism)
    if questionIndex in active_sessions[folder]["uploads"]:
        logger.info(f"Duplicate upload detected for Q{questionIndex}, allowing re-upload")
    
    # MỤC ĐÍCH: Check MIME type từ header (advisory only, vẫn check magic bytes sau)
    if video.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"Invalid content type: {video.content_type}")
        raise HTTPException(
            status_code=415, 
            detail=f"Unsupported media type: {video.content_type}. Allowed: {', '.join(ALLOWED_MIME_TYPES)}"
        )
    
    filename = f"Q{questionIndex}.webm"
    dest_path = folder_path / filename
    
    file_size = 0
    try:
        # MỤC ĐÍCH: Lưu file theo chunk để handle file lớn và check size
        with dest_path.open("wb") as buffer:
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await video.read(chunk_size):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    dest_path.unlink(missing_ok=True)  # Xóa file nếu quá lớn
                    logger.warning(f"File too large: {file_size} bytes")
                    raise HTTPException(
                        status_code=413, 
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
                    )
                buffer.write(chunk)
        
        # MỤC ĐÍCH: Verify file thực sự là video bằng magic bytes
        # Không tin content-type từ client vì có thể fake
        if not verify_video_by_signature(dest_path):
            dest_path.unlink(missing_ok=True)
            logger.warning(f"Invalid video file format detected")
            raise HTTPException(
                status_code=415,
                detail="File is not a valid video format"
            )
        
        # MỤC ĐÍCH: Update metadata với thông tin question vừa upload
        question_data = {
            "index": questionIndex,
            "uploadedAt": get_bangkok_timestamp(),
            "filename": filename,
            "size": file_size
        }
        await update_metadata(folder_path, question_data=question_data)
        
        # MỤC ĐÍCH: Track question đã upload để support retry
        active_sessions[folder]["uploads"].add(questionIndex)
        
        logger.info(f"Upload successful: {filename} ({file_size} bytes)")
        
        return {
            "ok": True,
            "savedAs": filename,
            "size": file_size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        dest_path.unlink(missing_ok=True)  # Cleanup on error
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
    # MỤC ĐÍCH: Chạy server khi execute file trực tiếp
    # - host="0.0.0.0" để bạn bè vào được (thay vì 127.0.0.1)
    # - port=8000
    uvicorn.run(app, host="0.0.0.0", port=8000)