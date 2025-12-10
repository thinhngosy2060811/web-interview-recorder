import json
import subprocess
import asyncio
import logging
from pathlib import Path
from fastapi import HTTPException
from app.config import UPLOAD_DIR
from app.utils import get_bangkok_timestamp
from app.database import get_metadata_lock 

logger = logging.getLogger(__name__)

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
    
    async with get_metadata_lock(folder_key):
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
