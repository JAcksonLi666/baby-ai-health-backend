from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from datetime import datetime
from typing import Dict

from config import LOG_LEVEL
from routes.health import router as health_router
from routes.records import router as records_router
from routes.daily import router as daily_router
from routes.ai import router as ai_router

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VERSION = "1.5.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("宝宝健康档案 AI 服务启动中...")
    logger.info(f"服务版本: {VERSION}")
    
    import threading
    from vector_db import vector_db_service
    
    def preload_models():
        try:
            logger.info("开始后台预加载embedding模型...")
            vector_db_service._init_embedding_model()
            logger.info("后台预加载完成")
        except Exception as e:
            logger.warning(f"预加载失败，但服务仍可运行: {str(e)}")

    threading.Thread(target=preload_models, daemon=True).start()

    yield
    logger.info("宝宝健康档案 AI 服务已关闭")


app = FastAPI(
    title="宝宝健康档案 AI 服务",
    description="本地化婴幼儿健康档案管理、化验单识别与智能问答系统",
    version=VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Rate Limiter ====================
_rate_limit_store: Dict[str, list] = {}
_RATE_LIMIT_WINDOW = 60
_RATE_LIMIT_MAX_REQUESTS = 60
_RATE_LIMIT_AI_MAX = 10


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    path = request.url.path
    if any(ai_path in path for ai_path in ["/ask", "/api/lab-report", "/api/symptom", "/api/chat"]):
        max_requests = _RATE_LIMIT_AI_MAX
    else:
        max_requests = _RATE_LIMIT_MAX_REQUESTS

    if client_ip not in _rate_limit_store:
        _rate_limit_store[client_ip] = []

    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < _RATE_LIMIT_WINDOW
    ]

    if len(_rate_limit_store[client_ip]) >= max_requests:
        return JSONResponse(
            status_code=429,
            content={"detail": "请求过于频繁，请稍后再试"}
        )

    _rate_limit_store[client_ip].append(now)
    response = await call_next(request)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务内部错误，请稍后重试"}
    )

# ==================== 注册路由 ====================
app.include_router(health_router)
app.include_router(records_router)
app.include_router(daily_router)
app.include_router(ai_router)


if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动服务，版本: {VERSION}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
