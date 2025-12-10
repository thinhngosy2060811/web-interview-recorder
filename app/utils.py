import re
from datetime import datetime
import pytz

BANGKOK_TZ = pytz.timezone('Asia/Bangkok')

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
