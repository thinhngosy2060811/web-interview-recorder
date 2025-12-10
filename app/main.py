import sys
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# --- FIX LỖI TIẾNG VIỆT TRÊN WINDOWS ---
# Phải đặt cái này ở đầu để tránh lỗi khi in log
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Import các biến cấu hình
from app.config import BASE_DIR, UPLOAD_DIR
from app.routers import router
from app.services.ai_service import init_ai_models

# 1. Cấu hình Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 2. Khởi tạo App
app = FastAPI(title="Web Interview Recorder (Integrated)", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Khởi tạo AI
init_ai_models()

# 4. Mount thư mục tĩnh
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# 5. Kết nối Router
app.include_router(router)

# 6. Trang chủ
@app.get("/", response_class=HTMLResponse)
def home():
    index_file = BASE_DIR / "static" / "index_first.html" 
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>File index_first.html not found</h1>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)