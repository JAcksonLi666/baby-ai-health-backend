from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import logging
import uuid
import os
import json
from pathlib import Path
from datetime import datetime
import shutil

from config import UPLOAD_DIR, MAX_UPLOAD_SIZE, ALLOWED_EXTENSIONS, LOG_LEVEL
from models import UploadResponse, AskRequest, AskResponse, ErrorResponse
from ocr_service import ocr_service
from vector_db import vector_db_service
from rag_service import rag_service

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("宝宝健康档案 AI 服务启动中...")
    logger.info(f"上传目录: {UPLOAD_DIR}")
    logger.info(f"最大文件大小: {MAX_UPLOAD_SIZE / (1024*1024):.1f} MB")

    import threading
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
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["健康检查"])
async def root():
    """服务健康检查"""
    return {
        "status": "running",
        "service": "baby-ai-health-backend",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["健康检查"])
async def health_check():
    """快速健康检查"""
    try:
        db_stats = vector_db_service.get_collection_stats()

        return {
            "status": "healthy",
            "services": {
                "chroma_db": {
                    "status": "online",
                    "total_records": db_stats.get("total_records", 0)
                },
                "embedding": {
                    "status": "ready" if vector_db_service.embedding_initialized else "loading"
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse, tags=["文件上传"])
async def upload_file(
    file: UploadFile = File(...),
    record_date: str = Query(None, description="记录日期 (YYYY-MM-DD)"),
    record_type: str = Query("general", description="记录类型: blood_test, urine_test, general")
):
    """上传化验单图片或 PDF，进行 OCR 识别和向量化存储"""

    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}，支持的类型: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if file.size and file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE / (1024*1024):.1f} MB"
        )

    file_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    filename = f"{file_id}{file_ext}"
    file_path = UPLOAD_DIR / filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"文件已保存: {file_path}")

        if file_ext == '.pdf':
            extracted_text = ocr_service.extract_text_from_pdf(str(file_path))
        else:
            extracted_text = ocr_service.extract_text_from_image(str(file_path))
        if not extracted_text:
            return UploadResponse(
                success=False,
                file_id=file_id,
                filename=filename,
                message="未能从文件中提取到文字，请确保图片清晰"
            )

        desensitized_text = ocr_service.desensitize_text(extracted_text)

        metrics = ocr_service.parse_health_metrics(desensitized_text)

        metadata = {
            "type": record_type,
            "filename": filename,
            "original_filename": file.filename,
            "file_size": file.size or 0,
            "upload_time": datetime.now().isoformat(),
            "record_date": record_date or datetime.now().strftime("%Y-%m-%d"),
            "metrics_count": len(metrics)
        }
        storage_text = f"""日期：{metadata['record_date']}
类型：{record_type}
指标：{', '.join([m['name'] for m in metrics]) if metrics else 'N/A'}
内容：{desensitized_text}
"""
        vector_db_service.add_record(
            record_id=file_id,
            text=storage_text,
            metadata=metadata,
            date=metadata['record_date']
        )
        return UploadResponse(
            success=True,
            file_id=file_id,
            filename=filename,
            message="上传成功，已完成 OCR 识别和向量化存储",
            extracted_text=desensitized_text
        )
    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse, tags=["智能问答"])
async def ask_question(request: AskRequest):
    """基于历史健康档案的智能问答"""

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        result = rag_service.answer_question(
            question=request.question,
            top_k=request.top_k,
            use_cloud=request.use_cloud
        )
        if result.get("success"):
            return AskResponse(
                success=True,
                answer=result["answer"],
                sources=result.get("sources", []),
                model_used=result.get("model_used", "unknown"),
                cloud_used=result.get("cloud_used", False)
            )
        else:
            return AskResponse(
                success=False,
                answer="",
                sources=[],
                model_used=result.get("model_used", "unknown"),
                cloud_used=request.use_cloud
            )
    except Exception as e:
        logger.error(f"问答处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_stream_response(question: str, top_k: int = 3, use_cloud: bool = False):
    """生成流式响应"""
    try:
        result_gen = rag_service.answer_question_stream(
            question=question,
            top_k=top_k,
            use_cloud=use_cloud
        )

        sources = []
        model_used = None
        context_count = 0

        for token in result_gen:
            if token is None:
                continue

            if token.startswith("错误:"):
                yield f"data: {json.dumps({'error': token}, ensure_ascii=False)}\n\n"
                continue

            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"流式生成失败: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"


@app.get("/ask/stream", tags=["智能问答"])
async def ask_question_stream(
    question: str = Query(..., description="问题内容"),
    top_k: int = Query(3, ge=1, le=10, description="返回的参考档案数量"),
    use_cloud: bool = Query(False, description="是否使用云端模型")
):
    """基于历史健康档案的智能问答（流式输出）"""

    if not question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    return StreamingResponse(
        generate_stream_response(
            question=question,
            top_k=top_k,
            use_cloud=use_cloud
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/records", tags=["档案管理"])
async def get_records(limit: int = Query(50, ge=1, le=100)):
    """获取所有健康档案记录"""
    try:
        records = vector_db_service.get_all_records(limit=limit)
        stats = vector_db_service.get_collection_stats()
        return {
            "success": True,
            "records": [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "metadata": r["metadata"]
                }
                for r in records
            ],
            "total": stats.get("total_records", 0),
            "showing": len(records)
        }
    except Exception as e:
        logger.error(f"获取记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/record/{record_id}", tags=["档案管理"])
async def get_record(record_id: str):
    """获取指定健康档案详情"""
    try:
        record = vector_db_service.get_record(record_id)
        if record:
            return {
                "success": True,
                "record": record
            }
        else:
            raise HTTPException(status_code=404, detail="记录不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取记录详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/record/{record_id}", tags=["档案管理"])
async def delete_record(record_id: str):
    """删除指定健康档案"""
    try:
        success = vector_db_service.delete_record(record_id)
        if success:
            return {
                "success": True,
                "message": f"记录 {record_id} 已删除"
            }
        else:
            raise HTTPException(status_code=500, detail="删除失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-trend", tags=["健康分析"])
async def analyze_health_trend(
    metric_name: str = Query(..., description="要分析的健康指标名称"),
    time_range: str = Query("all", description="时间范围: all, 1month, 3months, 6months")
):
    """分析特定健康指标的历史趋势"""
    try:
        result = rag_service.analyze_health_trend(metric_name, time_range)
        if result.get("success"):
            return result
        else:
            return JSONResponse(
                status_code=404,
                content=result
            )
    except Exception as e:
        logger.error(f"趋势分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", tags=["模型管理"])
async def get_available_models():
    """获取可用的 AI 模型列表"""
    try:
        ollama_status = rag_service.llm.check_ollama_health()
        if ollama_status:
            models = rag_service.llm.get_available_models()
            return {
                "success": True,
                "ollama_online": True,
                "models": models,
                "default_model": rag_service.llm.default_model
            }
        else:
            return {
                "success": False,
                "ollama_online": False,
                "models": [],
                "message": "Ollama 服务未启动，请先运行 `ollama serve`"
            }
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}")
        return {
            "success": False,
            "ollama_online": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
