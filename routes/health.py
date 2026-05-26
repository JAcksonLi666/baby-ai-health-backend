from fastapi import APIRouter, HTTPException
from datetime import datetime
from vector_db import vector_db_service
from knowledge_base import knowledge_service

router = APIRouter(tags=["健康检查"])

VERSION = "1.5.0"


@router.get("/")
async def root():
    """服务健康检查"""
    return {
        "status": "running",
        "service": "baby-ai-health-backend",
        "version": VERSION,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health")
async def health_check():
    """快速健康检查"""
    try:
        db_stats = vector_db_service.get_collection_stats()
        kb_status = knowledge_service.get_status()
        return {
            "status": "healthy",
            "version": VERSION,
            "services": {
                "chroma_db": {
                    "status": "online",
                    "total_records": db_stats.get("total_records", 0)
                },
                "embedding": {
                    "status": "ready" if vector_db_service.embedding_initialized else "loading"
                },
                "knowledge_base": {
                    "status": "ready" if kb_status.get("ready") else "not_ready",
                    "total_entries": kb_status.get("total_entries", 0)
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")
