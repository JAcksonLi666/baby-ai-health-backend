import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
MODEL_CACHE_DIR = DATA_DIR / "models"

for directory in [UPLOAD_DIR, DATA_DIR, VECTOR_DB_DIR, MODEL_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text2vec-base-chinese")

CHROMA_COLLECTION_NAME = "baby_health_records"

CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "")
CLOUD_API_BASE = os.getenv("CLOUD_API_BASE", "https://api.deepseek.com")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

MAX_UPLOAD_SIZE = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.pdf', '.bmp', '.tiff'}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
