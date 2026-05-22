from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import logging
import uuid
import os
import json
import re
import time
from pathlib import Path
from datetime import datetime, date
import shutil
from typing import Dict

from config import UPLOAD_DIR, MAX_UPLOAD_SIZE, ALLOWED_EXTENSIONS, LOG_LEVEL
from models import UploadResponse, AskRequest, AskResponse, ErrorResponse
from models import (
    SleepRecordCreate, SleepRecordUpdate, SleepRecordResponse,
    DiaperRecordCreate, DiaperRecordUpdate, DiaperRecordResponse,
    CryRecordCreate, CryRecordUpdate, CryRecordResponse,
    FeedingRecordCreate, FeedingRecordUpdate, FeedingRecordResponse,
    GrowthRecordCreate, GrowthRecordUpdate, GrowthRecordResponse,
    ReminderRecordCreate, ReminderRecordUpdate, ReminderRecordResponse,
    TodaySummaryResponse,
    LabReportParseRequest, LabReportResponse,
)
from ocr_service import ocr_service
from vector_db import vector_db_service
from rag_service import rag_service
from daily_records import sleep_service, diaper_service, cry_service, feeding_service, growth_service, reminder_service
from knowledge_base import knowledge_service
from growth_standards import get_growth_standard, calculate_percentile, AGE_GROUPS
from lab_report_parser import lab_report_parser
from symptom_checker import symptom_checker
from chat_history import chat_history_service
from models import (
    SymptomCheckRequest, SymptomAnalysis,
    ChatSessionCreate, ChatMessageCreate, ChatSessionResponse,
    KnowledgeEntryCreate,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VERSION = "1.4.0"


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
_rate_limit_store: Dict[str, list] = {}  # {client_ip: [timestamp, ...]}
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX_REQUESTS = 60  # per window per IP
_RATE_LIMIT_AI_MAX = 10  # per window for AI endpoints


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple in-memory rate limiter."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Determine limit based on endpoint
    path = request.url.path
    if any(ai_path in path for ai_path in ["/ask", "/api/lab-report", "/api/symptom", "/api/chat"]):
        max_requests = _RATE_LIMIT_AI_MAX
    else:
        max_requests = _RATE_LIMIT_MAX_REQUESTS

    # Clean old entries and check
    if client_ip not in _rate_limit_store:
        _rate_limit_store[client_ip] = []

    # Remove timestamps outside the window
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


@app.get("/", tags=["健康检查"])
async def root():
    """服务健康检查"""
    return {
        "status": "running",
        "service": "baby-ai-health-backend",
        "version": VERSION,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["健康检查"])
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
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@app.post("/upload/preview", tags=["文件上传"])
async def preview_upload(
    file: UploadFile = File(...)
):
    """预识别文件内容，返回识别结果和识别出的日期（不保存）"""

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

    # 临时文件名
    temp_file_id = f"temp_{uuid.uuid4().hex[:8]}"
    temp_filename = f"{temp_file_id}{file_ext}"
    temp_path = UPLOAD_DIR / temp_filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Secondary size check
        actual_size = temp_path.stat().st_size
        if actual_size > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE / (1024*1024):.1f} MB")

        logger.info(f"临时文件已保存: {temp_path}")

        if file_ext == '.pdf':
            extracted_text = ocr_service.extract_text_from_pdf(str(temp_path))
        else:
            extracted_text = ocr_service.extract_text_from_image(str(temp_path))

        if not extracted_text:
            return {
                "success": False,
                "message": "未能从文件中提取到文字，请确保图片清晰",
                "extracted_text": "",
                "detected_date": None
            }

        desensitized_text = ocr_service.desensitize_text(extracted_text)
        
        # 识别日期
        detected_date = ocr_service.extract_date_from_text(desensitized_text)
        
        # 解析健康指标
        metrics = ocr_service.parse_health_metrics(desensitized_text)

        return {
            "success": True,
            "message": "识别成功",
            "extracted_text": desensitized_text,
            "detected_date": detected_date,
            "metrics": metrics,
            "temp_file_id": temp_file_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预识别失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")
    finally:
        if temp_path.exists():
            temp_path.unlink()


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

    if record_date and not re.match(r"^\d{4}-\d{2}-\d{2}$", record_date):
        raise HTTPException(status_code=400, detail="record_date 格式应为 YYYY-MM-DD")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"文件已保存: {file_path}")

        actual_size = file_path.stat().st_size
        if actual_size > MAX_UPLOAD_SIZE:
            file_path.unlink()
            raise HTTPException(status_code=400, detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE / (1024*1024):.1f} MB")

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

        # 使用用户提供的日期，如果没有则尝试从识别内容中提取，最后使用当前日期
        final_date = record_date
        if not final_date:
            final_date = ocr_service.extract_date_from_text(desensitized_text)
        if not final_date:
            final_date = datetime.now().strftime("%Y-%m-%d")

        metadata = {
            "type": record_type,
            "filename": filename,
            "original_filename": file.filename,
            "file_size": file.size or 0,
            "upload_time": datetime.now().isoformat(),
            "record_date": final_date,
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
            extracted_text=desensitized_text,
            record_date=final_date
        )
    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@app.post("/ask", response_model=AskResponse, tags=["智能问答"])
async def ask_question(request: AskRequest):
    """基于历史健康档案的智能问答"""

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        result = rag_service.answer_question(
            question=request.question,
            top_k=request.top_k,
            use_cloud=request.use_cloud,
            model=request.model
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
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


def generate_stream_response(question: str, top_k: int = 3, use_cloud: bool = False, model: str = "auto"):
    """生成流式响应"""
    try:
        result_gen = rag_service.answer_question_stream(
            question=question,
            top_k=top_k,
            use_cloud=use_cloud,
            model=model
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
    use_cloud: bool = Query(False, description="是否使用云端模型"),
    model: str = Query("auto", description="模型名称，'auto' 表示自动选择最聪明的模型")
):
    """基于历史健康档案的智能问答（流式输出）"""

    if not question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    return StreamingResponse(
        generate_stream_response(
            question=question,
            top_k=top_k,
            use_cloud=use_cloud,
            model=model
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
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


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
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


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
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@app.put("/record/{record_id}", tags=["档案管理"])
async def update_record(
    record_id: str,
    record_date: str = Query(None, description="记录日期 (YYYY-MM-DD)"),
    record_type: str = Query(None, description="记录类型: blood_test, urine_test, general"),
    new_text: str = Query(None, description="更新的文本内容")
):
    """更新指定健康档案的元数据或内容"""
    try:
        # 获取现有记录
        record = vector_db_service.get_record(record_id)
        if not record:
            raise HTTPException(status_code=404, detail="记录不存在")
        
        # 更新元数据
        metadata = record.get("metadata", {})
        if record_date:
            metadata["record_date"] = record_date
        if record_type:
            metadata["type"] = record_type
            metadata["record_type"] = record_type
        
        # 确定要存储的文本
        text = new_text if new_text else record.get("text", "")
        
        # 如果日期更新了，更新文本中的日期
        if record_date and not new_text:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("日期："):
                    lines[i] = f"日期：{record_date}"
                    break
            text = '\n'.join(lines)
        
        # 先添加新记录，再删除旧记录（避免添加失败导致数据丢失）
        success = vector_db_service.add_record(
            record_id=record_id,
            text=text,
            metadata=metadata,
            date=metadata.get("record_date")
        )
        
        if success:
            vector_db_service.delete_record(record_id)
            return {
                "success": True,
                "message": f"记录 {record_id} 已更新",
                "record": {
                    "id": record_id,
                    "text": text,
                    "metadata": metadata
                }
            }
        else:
            raise HTTPException(status_code=500, detail="更新失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@app.get("/records/filter", tags=["档案管理"])
async def filter_records(
    record_type: str = Query(None, description="按类型筛选: blood_test, urine_test, general"),
    start_date: str = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: str = Query(None, description="结束日期 (YYYY-MM-DD)"),
    keyword: str = Query(None, description="关键词搜索"),
    limit: int = Query(50, ge=1, le=100)
):
    """筛选健康档案记录"""
    try:
        records = vector_db_service.get_all_records(limit=limit)
        
        # 按类型筛选
        if record_type:
            records = [r for r in records if r.get("metadata", {}).get("type") == record_type]
        
        # 按日期范围筛选
        if start_date or end_date:
            filtered = []
            for record in records:
                record_date = record.get("metadata", {}).get("record_date", "")
                if record_date:
                    if start_date and record_date < start_date:
                        continue
                    if end_date and record_date > end_date:
                        continue
                filtered.append(record)
            records = filtered
        
        # 按关键词搜索
        if keyword:
            keyword_lower = keyword.lower()
            records = [
                r for r in records 
                if keyword_lower in r.get("text", "").lower() or 
                   keyword_lower in str(r.get("metadata", {})).lower()
            ]
        
        stats = vector_db_service.get_collection_stats()
        return {
            "success": True,
            "records": records,
            "total": len(records),
            "filters": {
                "record_type": record_type,
                "start_date": start_date,
                "end_date": end_date,
                "keyword": keyword
            }
        }
    except Exception as e:
        logger.error(f"筛选记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@app.get("/records/types", tags=["档案管理"])
async def get_record_types():
    """获取所有记录类型统计"""
    try:
        records = vector_db_service.get_all_records(limit=1000)
        
        type_counts = {}
        for record in records:
            record_type = record.get("metadata", {}).get("type", "general")
            type_counts[record_type] = type_counts.get(record_type, 0) + 1
        
        # 定义类型名称映射
        type_names = {
            "blood_test": "血液检测",
            "urine_test": "尿液检测",
            "general": "常规记录"
        }
        
        result = []
        for record_type, count in type_counts.items():
            result.append({
                "type": record_type,
                "name": type_names.get(record_type, record_type),
                "count": count
            })
        
        return {
            "success": True,
            "types": result,
            "total_records": len(records)
        }
    except Exception as e:
        logger.error(f"获取记录类型统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


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
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@app.get("/models", tags=["模型管理"])
async def get_available_models():
    """获取可用的 AI 模型列表"""
    try:
        ollama_status = rag_service.llm.check_ollama_health()
        if ollama_status:
            models = rag_service.llm.get_available_models()
            model_names = [m["name"] for m in models]
            auto_model = rag_service.llm.select_smartest_model()
            return {
                "success": True,
                "ollama_online": True,
                "models": model_names,
                "model_details": models,
                "default_model": rag_service.llm.default_model,
                "auto_model": auto_model,
                "message": f"Auto 模式将自动选择: {auto_model}"
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


@app.get("/search/online", tags=["联网搜索"])
async def online_search(
    query: str = Query(..., description="搜索关键词"),
    max_results: int = Query(5, ge=1, le=10, description="最大结果数")
):
    """使用 Tavily 进行联网搜索"""
    try:
        if not rag_service.tavily_enabled:
            return {
                "success": False,
                "error": "Tavily API Key 未配置，无法使用联网搜索功能",
                "results": []
            }

        result = rag_service.search_online(query, max_results)
        return result
    except Exception as e:
        logger.error(f"联网搜索失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


@app.get("/search/status", tags=["联网搜索"])
async def search_status():
    """获取联网搜索功能状态"""
    return {
        "enabled": rag_service.tavily_enabled,
        "provider": "Tavily Search" if rag_service.tavily_enabled else None,
        "quota": "每天 1000 次" if rag_service.tavily_enabled else None
    }


# ==================== MVP: 日常记录 API ====================

# --- 睡眠记录 ---
@app.post("/api/sleep", response_model=SleepRecordResponse, tags=["日常记录-睡眠"])
async def create_sleep_record(data: SleepRecordCreate):
    try:
        record = sleep_service.create(data.model_dump())
        return record
    except Exception as e:
        logger.error(f"创建睡眠记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/sleep", tags=["日常记录-睡眠"])
async def list_sleep_records(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    page: int = Query(None, ge=1),
    page_size: int = Query(None, ge=1, le=200),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    try:
        return sleep_service.list_records(
            limit=limit, offset=offset, page=page, page_size=page_size,
            start_date=start_date, end_date=end_date
        )
    except Exception as e:
        logger.error(f"获取睡眠记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/sleep/ongoing", tags=["日常记录-睡眠"])
async def get_ongoing_sleep():
    try:
        return sleep_service.get_ongoing()
    except Exception as e:
        logger.error(f"获取进行中睡眠失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/sleep/{record_id}", tags=["日常记录-睡眠"])
async def get_sleep_record(record_id: str):
    record = sleep_service.get_by_id(record_id)
    if record:
        return record
    raise HTTPException(status_code=404, detail="睡眠记录不存在")

@app.put("/api/sleep/{record_id}", tags=["日常记录-睡眠"])
async def update_sleep_record(record_id: str, data: SleepRecordUpdate):
    record = sleep_service.update(record_id, data.model_dump(exclude_none=True))
    if record:
        return record
    raise HTTPException(status_code=404, detail="睡眠记录不存在")

@app.delete("/api/sleep/{record_id}", tags=["日常记录-睡眠"])
async def delete_sleep_record(record_id: str):
    if sleep_service.delete(record_id):
        return {"success": True, "message": "睡眠记录已删除"}
    raise HTTPException(status_code=404, detail="睡眠记录不存在")

# --- 排泄记录 ---
@app.post("/api/diaper", response_model=DiaperRecordResponse, tags=["日常记录-排泄"])
async def create_diaper_record(data: DiaperRecordCreate):
    try:
        record = diaper_service.create(data.model_dump())
        return record
    except Exception as e:
        logger.error(f"创建排泄记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/diaper", tags=["日常记录-排泄"])
async def list_diaper_records(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    page: int = Query(None, ge=1),
    page_size: int = Query(None, ge=1, le=200),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    try:
        return diaper_service.list_records(
            limit=limit, offset=offset, page=page, page_size=page_size,
            start_date=start_date, end_date=end_date
        )
    except Exception as e:
        logger.error(f"获取排泄记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/diaper/{record_id}", tags=["日常记录-排泄"])
async def get_diaper_record(record_id: str):
    record = diaper_service.get_by_id(record_id)
    if record:
        return record
    raise HTTPException(status_code=404, detail="排泄记录不存在")

@app.put("/api/diaper/{record_id}", tags=["日常记录-排泄"])
async def update_diaper_record(record_id: str, data: DiaperRecordUpdate):
    record = diaper_service.update(record_id, data.model_dump(exclude_none=True))
    if record:
        return record
    raise HTTPException(status_code=404, detail="排泄记录不存在")

@app.delete("/api/diaper/{record_id}", tags=["日常记录-排泄"])
async def delete_diaper_record(record_id: str):
    if diaper_service.delete(record_id):
        return {"success": True, "message": "排泄记录已删除"}
    raise HTTPException(status_code=404, detail="排泄记录不存在")

# --- 哭声记录 ---
@app.get("/api/cry/analyze", tags=["日常记录-哭声"])
async def analyze_cry_reason():
    try:
        return cry_service.analyze_cry_reason()
    except Exception as e:
        logger.error(f"分析哭声原因失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/cry/ongoing", tags=["日常记录-哭声"])
async def get_ongoing_cry():
    try:
        return cry_service.get_ongoing()
    except Exception as e:
        logger.error(f"获取进行中哭闹失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.post("/api/cry", response_model=CryRecordResponse, tags=["日常记录-哭声"])
async def create_cry_record(data: CryRecordCreate):
    try:
        record = cry_service.create(data.model_dump())
        return record
    except Exception as e:
        logger.error(f"创建哭声记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/cry", tags=["日常记录-哭声"])
async def list_cry_records(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    page: int = Query(None, ge=1),
    page_size: int = Query(None, ge=1, le=200),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    try:
        return cry_service.list_records(
            limit=limit, offset=offset, page=page, page_size=page_size,
            start_date=start_date, end_date=end_date
        )
    except Exception as e:
        logger.error(f"获取哭声记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/cry/{record_id}", tags=["日常记录-哭声"])
async def get_cry_record(record_id: str):
    record = cry_service.get_by_id(record_id)
    if record:
        return record
    raise HTTPException(status_code=404, detail="哭声记录不存在")

@app.put("/api/cry/{record_id}", tags=["日常记录-哭声"])
async def update_cry_record(record_id: str, data: CryRecordUpdate):
    record = cry_service.update(record_id, data.model_dump(exclude_none=True))
    if record:
        return record
    raise HTTPException(status_code=404, detail="哭声记录不存在")

@app.delete("/api/cry/{record_id}", tags=["日常记录-哭声"])
async def delete_cry_record(record_id: str):
    if cry_service.delete(record_id):
        return {"success": True, "message": "哭声记录已删除"}
    raise HTTPException(status_code=404, detail="哭声记录不存在")

# --- 喂养记录 ---
@app.post("/api/feeding", response_model=FeedingRecordResponse, tags=["日常记录-喂养"])
async def create_feeding_record(data: FeedingRecordCreate):
    try:
        record = feeding_service.create(data.model_dump())
        return record
    except Exception as e:
        logger.error(f"创建喂养记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/feeding", tags=["日常记录-喂养"])
async def list_feeding_records(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    page: int = Query(None, ge=1),
    page_size: int = Query(None, ge=1, le=200),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    try:
        return feeding_service.list_records(
            limit=limit, offset=offset, page=page, page_size=page_size,
            start_date=start_date, end_date=end_date
        )
    except Exception as e:
        logger.error(f"获取喂养记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/feeding/{record_id}", tags=["日常记录-喂养"])
async def get_feeding_record(record_id: str):
    record = feeding_service.get_by_id(record_id)
    if record:
        return record
    raise HTTPException(status_code=404, detail="喂养记录不存在")

@app.put("/api/feeding/{record_id}", tags=["日常记录-喂养"])
async def update_feeding_record(record_id: str, data: FeedingRecordUpdate):
    record = feeding_service.update(record_id, data.model_dump(exclude_none=True))
    if record:
        return record
    raise HTTPException(status_code=404, detail="喂养记录不存在")

@app.delete("/api/feeding/{record_id}", tags=["日常记录-喂养"])
async def delete_feeding_record(record_id: str):
    if feeding_service.delete(record_id):
        return {"success": True, "message": "喂养记录已删除"}
    raise HTTPException(status_code=404, detail="喂养记录不存在")

# --- 生长发育记录 ---
@app.post("/api/growth", response_model=GrowthRecordResponse, tags=["日常记录-生长发育"])
async def create_growth_record(data: GrowthRecordCreate):
    try:
        record = growth_service.create(data.model_dump())
        return record
    except Exception as e:
        logger.error(f"创建生长发育记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/growth", tags=["日常记录-生长发育"])
async def list_growth_records(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    page: int = Query(None, ge=1),
    page_size: int = Query(None, ge=1, le=200),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    try:
        return growth_service.list_records(
            limit=limit, offset=offset, page=page, page_size=page_size,
            start_date=start_date, end_date=end_date
        )
    except Exception as e:
        logger.error(f"获取生长发育记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/growth/latest", tags=["日常记录-生长发育"])
async def get_latest_growth_record():
    """获取最新的生长发育记录"""
    record = growth_service.get_latest()
    if record:
        return {"success": True, "record": record}
    return {"success": False, "message": "暂无生长发育记录"}

@app.get("/api/growth/{record_id}", tags=["日常记录-生长发育"])
async def get_growth_record(record_id: str):
    record = growth_service.get_by_id(record_id)
    if record:
        return record
    raise HTTPException(status_code=404, detail="生长发育记录不存在")

@app.put("/api/growth/{record_id}", tags=["日常记录-生长发育"])
async def update_growth_record(record_id: str, data: GrowthRecordUpdate):
    record = growth_service.update(record_id, data.model_dump(exclude_none=True))
    if record:
        return record
    raise HTTPException(status_code=404, detail="生长发育记录不存在")

@app.delete("/api/growth/{record_id}", tags=["日常记录-生长发育"])
async def delete_growth_record(record_id: str):
    if growth_service.delete(record_id):
        return {"success": True, "message": "生长发育记录已删除"}
    raise HTTPException(status_code=404, detail="生长发育记录不存在")

# --- 提醒记录 ---
@app.post("/api/reminder", response_model=ReminderRecordResponse, tags=["日常记录-提醒"])
async def create_reminder_record(data: ReminderRecordCreate):
    try:
        record = reminder_service.create(data.model_dump())
        return record
    except Exception as e:
        logger.error(f"创建提醒记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/reminder", tags=["日常记录-提醒"])
async def list_reminder_records(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    page: int = Query(None, ge=1),
    page_size: int = Query(None, ge=1, le=200),
    status: str = Query(None),
):
    try:
        return reminder_service.list_records(
            limit=limit, offset=offset, page=page, page_size=page_size
        )
    except Exception as e:
        logger.error(f"获取提醒记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/reminder/pending", tags=["日常记录-提醒"])
async def get_pending_reminders():
    """获取待处理的提醒"""
    records = reminder_service.get_pending()
    return {"success": True, "records": records, "total": len(records)}

@app.get("/api/reminder/today", tags=["日常记录-提醒"])
async def get_today_reminders():
    """获取今天的提醒"""
    records = reminder_service.get_today_reminders()
    return {"success": True, "records": records, "total": len(records)}

@app.get("/api/reminder/{record_id}", tags=["日常记录-提醒"])
async def get_reminder_record(record_id: str):
    record = reminder_service.get_by_id(record_id)
    if record:
        return record
    raise HTTPException(status_code=404, detail="提醒记录不存在")

@app.put("/api/reminder/{record_id}", tags=["日常记录-提醒"])
async def update_reminder_record(record_id: str, data: ReminderRecordUpdate):
    record = reminder_service.update(record_id, data.model_dump(exclude_none=True))
    if record:
        return record
    raise HTTPException(status_code=404, detail="提醒记录不存在")

@app.delete("/api/reminder/{record_id}", tags=["日常记录-提醒"])
async def delete_reminder_record(record_id: str):
    if reminder_service.delete(record_id):
        return {"success": True, "message": "提醒记录已删除"}
    raise HTTPException(status_code=404, detail="提醒记录不存在")

# --- 今日汇总 ---
@app.get("/api/today/summary", tags=["日常记录"])
async def get_today_summary():
    try:
        today = date.today().isoformat()

        # Sleep summary
        sleep_records = sleep_service.get_today_records()
        total_minutes = 0
        nap_count = 0
        nap_minutes = 0
        night_minutes = 0
        is_ongoing = False

        for r in sleep_records:
            if r.get("is_ongoing"):
                is_ongoing = True
                # Calculate ongoing duration
                try:
                    start = datetime.strptime(r["start_time"], "%Y-%m-%d %H:%M")
                    diff = int((datetime.now() - start).total_seconds() / 60)
                    total_minutes += diff
                    if r.get("sleep_type") == "nap":
                        nap_count += 1
                        nap_minutes += diff
                    else:
                        night_minutes += diff
                except (ValueError, TypeError):
                    pass
            else:
                dur = r.get("duration_minutes", 0) or 0
                total_minutes += dur
                if r.get("sleep_type") == "nap":
                    nap_count += 1
                    nap_minutes += dur
                else:
                    night_minutes += dur

        hours = total_minutes // 60
        mins = total_minutes % 60
        total_display = f"{hours}小时{mins}分钟" if hours > 0 else f"{mins}分钟"

        sleep_summary = {
            "total_minutes": total_minutes,
            "total_display": total_display,
            "nap_count": nap_count,
            "nap_minutes": nap_minutes,
            "night_minutes": night_minutes,
            "is_ongoing": is_ongoing,
            "record_count": len(sleep_records),
        }

        # Diaper summary
        diaper_records = diaper_service.get_today_records()
        pee_count = sum(1 for r in diaper_records if r.get("diaper_type") in ("pee", "both"))
        poop_count = sum(1 for r in diaper_records if r.get("diaper_type") in ("poop", "both"))
        both_count = sum(1 for r in diaper_records if r.get("diaper_type") == "both")
        colors = list(set(r.get("poop_color") for r in diaper_records if r.get("poop_color")))
        has_abnormal = any(c in ("red", "black", "white") for c in colors)

        diaper_summary = {
            "total_count": len(diaper_records),
            "pee_count": pee_count,
            "poop_count": poop_count,
            "both_count": both_count,
            "colors": colors,
            "has_abnormal": has_abnormal,
            "last_record": diaper_records[0] if diaper_records else None,
        }

        # Cry summary
        cry_records = cry_service.get_today_records()
        cry_total_minutes = sum(r.get("duration_minutes", 0) or 0 for r in cry_records)
        reason_counts = {}
        for r in cry_records:
            reason = r.get("reason", "unknown")
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        top_reason = max(reason_counts, key=reason_counts.get) if reason_counts else None
        cry_is_ongoing = any(not r.get("end_time") for r in cry_records)

        cry_summary = {
            "total_minutes": cry_total_minutes,
            "total_count": len(cry_records),
            "reason_counts": reason_counts,
            "top_reason": top_reason,
            "is_ongoing": cry_is_ongoing,
        }

        # Feeding summary
        feeding_records = feeding_service.get_today_records()
        breast_count = sum(1 for r in feeding_records if r.get("feeding_type") == "breast")
        formula_count = sum(1 for r in feeding_records if r.get("feeding_type") == "formula")
        solid_count = sum(1 for r in feeding_records if r.get("feeding_type") == "solid")
        water_count = sum(1 for r in feeding_records if r.get("feeding_type") == "water")
        total_duration = sum(r.get("duration_minutes", 0) or 0 for r in feeding_records)
        total_amount = sum(r.get("amount_ml", 0) or 0 for r in feeding_records)

        feeding_summary = {
            "total_count": len(feeding_records),
            "breast_count": breast_count,
            "formula_count": formula_count,
            "solid_count": solid_count,
            "water_count": water_count,
            "total_duration_minutes": total_duration,
            "total_amount_ml": total_amount,
        }

        # Growth summary
        latest_growth = growth_service.get_latest()
        growth_summary = {
            "has_record": latest_growth is not None,
            "latest_record": latest_growth,
        }

        # Generate insights
        insights = []
        if total_minutes < 720:  # less than 12 hours
            insights.append(f"今日睡眠总计 {total_display}，低于推荐时长，注意观察宝宝状态")
        insights.append(f"今日换尿布 {len(diaper_records)} 次（尿{pee_count}次，便{poop_count}次）")
        if cry_records:
            insights.append(f"今日哭闹 {len(cry_records)} 次，累计 {cry_total_minutes} 分钟")
            if top_reason:
                reason_names = {
                    "hungry": "饿了", "sleepy": "困了", "diaper": "尿布湿了",
                    "discomfort": "不舒服", "pain": "疼痛", "lonely": "需要安抚",
                    "overstimulated": "过度刺激", "unknown": "未知",
                }
                insights.append(f"今日哭闹主要原因为「{reason_names.get(top_reason, top_reason)}」")

        return TodaySummaryResponse(
            date=today,
            sleep=sleep_summary,
            diaper=diaper_summary,
            cry=cry_summary,
            feeding=feeding_summary,
            growth=growth_summary,
            insights=insights,
        )
    except Exception as e:
        logger.error(f"获取今日汇总失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

# --- 知识库 ---
@app.get("/api/knowledge/search", tags=["知识库"])
async def search_knowledge(
    query: str = Query(..., description="搜索关键词"),
    n_results: int = Query(3, ge=1, le=10),
):
    try:
        return knowledge_service.search(query, n_results)
    except Exception as e:
        logger.error(f"知识库搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.get("/api/knowledge/status", tags=["知识库"])
async def knowledge_status():
    return knowledge_service.get_status()


# --- Knowledge Base Dynamic Management ---
@app.get("/api/knowledge/list", tags=["Knowledge Base"])
async def list_knowledge_entries(category: str = Query(None, description="按分类过滤")):
    """List knowledge base entries, optionally filtered by category."""
    try:
        entries = knowledge_service.list_entries(category=category)
        return {
            "success": True,
            "entries": entries,
            "total": len(entries),
        }
    except Exception as e:
        logger.error(f"List knowledge entries failed: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@app.get("/api/knowledge/{entry_id}", tags=["Knowledge Base"])
async def get_knowledge_entry(entry_id: str):
    """Get a single knowledge entry by id."""
    entry = knowledge_service.get_entry(entry_id)
    if entry:
        return {
            "success": True,
            "entry": entry,
        }
    raise HTTPException(status_code=404, detail="知识条目不存在")


@app.post("/api/knowledge", tags=["Knowledge Base"])
async def add_knowledge_entry(data: KnowledgeEntryCreate):
    """Add a new knowledge entry to the knowledge base."""
    try:
        import uuid
        entry_id = f"kb_custom_{uuid.uuid4().hex[:8]}"
        entry = knowledge_service.add_entry({
            "id": entry_id,
            "title": data.title,
            "content": data.content,
            "source": data.source or "",
            "keywords": data.keywords,
            "category": data.category or "",
        })
        return {
            "success": True,
            "entry": entry,
            "message": f"知识条目 {entry_id} 已添加",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Add knowledge entry failed: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@app.delete("/api/knowledge/{entry_id}", tags=["Knowledge Base"])
async def delete_knowledge_entry(entry_id: str):
    """Delete a knowledge entry by id."""
    if knowledge_service.delete_entry(entry_id):
        return {
            "success": True,
            "message": f"知识条目 {entry_id} 已删除",
        }
    raise HTTPException(status_code=404, detail="知识条目不存在")


# ==================== 生长发育 API ====================

@app.get("/api/growth/standards", tags=["生长发育"])
async def get_growth_standards(
    gender: str = Query(..., description="性别: boys/girls"),
    metric: str = Query(..., description="指标: weight/height/bmi/head_circumference"),
    age_months: int = Query(..., ge=0, le=144, description="月龄"),
):
    """获取指定年龄的生长标准值"""
    try:
        # 验证参数
        if gender not in ["boys", "girls"]:
            raise HTTPException(status_code=400, detail="性别参数必须是 boys 或 girls")
        
        valid_metrics = ["weight", "height", "bmi", "head_circumference"]
        if metric not in valid_metrics:
            raise HTTPException(
                status_code=400, 
                detail=f"指标参数必须是以下之一: {', '.join(valid_metrics)}"
            )
        
        standard = get_growth_standard(gender, metric, age_months)
        if not standard:
            raise HTTPException(status_code=404, detail="未找到该年龄段的生长标准")
        
        return {
            "success": True,
            "gender": gender,
            "metric": metric,
            "age_months": age_months,
            "standards": standard,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取生长标准失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@app.get("/api/growth/percentile", tags=["生长发育"])
async def calculate_growth_percentile(
    value: float = Query(..., description="测量值"),
    gender: str = Query(..., description="性别: boys/girls"),
    metric: str = Query(..., description="指标: weight/height/bmi"),
    age_months: int = Query(..., ge=0, le=144, description="月龄"),
):
    """计算生长指标百分位"""
    try:
        # 验证参数
        if gender not in ["boys", "girls"]:
            raise HTTPException(status_code=400, detail="性别参数必须是 boys 或 girls")
        
        valid_metrics = ["weight", "height", "bmi"]
        if metric not in valid_metrics:
            raise HTTPException(
                status_code=400, 
                detail=f"指标参数必须是以下之一: {', '.join(valid_metrics)}"
            )
        
        percentile = calculate_percentile(value, gender, metric, age_months)
        if percentile is None:
            raise HTTPException(status_code=404, detail="无法计算百分位")
        
        return {
            "success": True,
            "value": value,
            "gender": gender,
            "metric": metric,
            "age_months": age_months,
            "percentile": round(percentile, 1),
            "evaluation": _evaluate_percentile(percentile),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"计算百分位失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


def _evaluate_percentile(percentile: float) -> str:
    """评估百分位"""
    if percentile < 3:
        return "偏低"
    elif percentile < 25:
        return "偏矮/偏轻"
    elif percentile < 75:
        return "正常"
    elif percentile < 97:
        return "偏高/偏重"
    else:
        return "高/重"


@app.get("/api/growth/age-groups", tags=["生长发育"])
async def get_age_groups():
    """获取年龄段定义"""
    return {
        "success": True,
        "age_groups": AGE_GROUPS,
    }


@app.get("/api/growth/metrics", tags=["生长发育"])
async def get_available_metrics():
    """获取可用的生长指标"""
    return {
        "success": True,
        "metrics": {
            "weight": {
                "name": "体重",
                "unit": "kg",
                "description": "儿童体重",
                "age_range": "0-144个月",
            },
            "height": {
                "name": "身高",
                "unit": "cm",
                "description": "儿童身高/身长",
                "age_range": "0-144个月",
            },
            "bmi": {
                "name": "BMI",
                "unit": "kg/m²",
                "description": "身体质量指数",
                "age_range": "24-144个月",
            },
            "head_circumference": {
                "name": "头围",
                "unit": "cm",
                "description": "头部周长",
                "age_range": "0-24个月",
            },
        },
    }


# ==================== Lab Report Parser API ====================

@app.post("/api/lab-report/parse", response_model=LabReportResponse, tags=["AI - Lab Report"])
async def parse_lab_report(data: LabReportParseRequest):
    """Parse lab report OCR text into structured JSON using LLM or regex fallback."""
    try:
        parsed = await lab_report_parser.parse_with_llm(
            text=data.text,
            report_type=data.report_type,
        )
        # Return basic parsed result without evaluation
        items = []
        for item in parsed.get("items", []):
            items.append({
                "name": item.get("name", ""),
                "value": item.get("value"),
                "unit": item.get("unit", ""),
                "reference_range": item.get("reference", ""),
                "status": item.get("status", "normal"),
            })
        return LabReportResponse(
            report_type=parsed.get("report_type", "unknown"),
            items=items,
            summary="Parsed successfully. Use /api/lab-report/evaluate for clinical evaluation.",
            abnormal_count=0,
            total_count=len(items),
        )
    except Exception as e:
        logger.error(f"Lab report parsing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service internal error, please try again later")


@app.post("/api/lab-report/evaluate", response_model=LabReportResponse, tags=["AI - Lab Report"])
async def evaluate_lab_report(data: LabReportParseRequest):
    """Parse lab report and evaluate results against age-specific reference ranges."""
    try:
        # Step 1: Parse the OCR text
        parsed = await lab_report_parser.parse_with_llm(
            text=data.text,
            report_type=data.report_type,
        )
        # Step 2: Evaluate against reference ranges
        evaluated = lab_report_parser.evaluate_results(
            parsed_data=parsed,
            age_months=data.age_months,
        )
        # Build response
        items = []
        for item in evaluated.get("items", []):
            items.append({
                "name": item.get("name", ""),
                "value": item.get("value"),
                "unit": item.get("unit", ""),
                "reference_range": item.get("reference_range", ""),
                "status": item.get("status", "normal"),
            })
        return LabReportResponse(
            report_type=evaluated.get("report_type", "unknown"),
            items=items,
            summary=evaluated.get("summary", ""),
            abnormal_count=evaluated.get("abnormal_count", 0),
            total_count=evaluated.get("total_count", 0),
        )
    except Exception as e:
        logger.error(f"Lab report evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service internal error, please try again later")


# ==================== Symptom Checker API ====================

@app.post("/api/symptom/analyze", tags=["AI - Symptom Checker"])
async def analyze_symptoms(data: SymptomCheckRequest):
    """Analyze infant/toddler symptoms and return classification with related knowledge.

    This endpoint does NOT provide medical advice. It only classifies symptoms
    and retrieves relevant knowledge base entries for reference.
    """
    if not data.symptoms:
        raise HTTPException(status_code=400, detail="Symptoms list cannot be empty")
    try:
        result = await symptom_checker.analyze_symptoms(
            symptoms=data.symptoms,
            age_months=data.age_months,
            duration_days=data.duration_days,
            severity=data.severity,
        )
        return result
    except Exception as e:
        logger.error(f"Symptom analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service internal error, please try again later")


@app.get("/api/symptom/categories", tags=["AI - Symptom Checker"])
async def get_symptom_categories():
    """Get all available symptom categories and their symptoms."""
    try:
        categories = symptom_checker.get_all_categories()
        return {
            "success": True,
            "categories": categories,
            "total_categories": len(categories),
        }
    except Exception as e:
        logger.error(f"Get symptom categories failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service internal error, please try again later")


# ==================== Chat History API ====================

@app.post("/api/chat/sessions", tags=["AI - Chat History"])
async def create_chat_session(data: ChatSessionCreate):
    """Create a new chat session."""
    try:
        session = chat_history_service.create_session(title=data.title)
        return session
    except Exception as e:
        logger.error(f"Create chat session failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service internal error, please try again later")


@app.get("/api/chat/sessions", tags=["AI - Chat History"])
async def list_chat_sessions(
    limit: int = Query(50, ge=1, le=200),
):
    """List all chat sessions, sorted by most recently updated."""
    try:
        sessions = chat_history_service.list_sessions(limit=limit)
        return {
            "success": True,
            "sessions": sessions,
            "total": len(sessions),
        }
    except Exception as e:
        logger.error(f"List chat sessions failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service internal error, please try again later")


@app.get("/api/chat/sessions/{session_id}/messages", tags=["AI - Chat History"])
async def get_chat_messages(
    session_id: str,
    limit: int = Query(20, ge=1, le=100),
):
    """Get message history for a chat session."""
    try:
        session = chat_history_service.get_session_history(session_id, limit=limit)
        if session:
            return {
                "success": True,
                "session": session,
            }
        raise HTTPException(status_code=404, detail="Chat session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get chat messages failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service internal error, please try again later")


@app.post("/api/chat/sessions/{session_id}/messages", tags=["AI - Chat History"])
async def add_chat_message(session_id: str, data: ChatMessageCreate):
    """Add a message to a chat session."""
    if data.session_id != session_id:
        raise HTTPException(status_code=400, detail="Session ID in path and body must match")
    try:
        session = chat_history_service.add_message(
            session_id=data.session_id,
            role=data.role,
            content=data.content,
        )
        if session:
            return {
                "success": True,
                "session": session,
            }
        raise HTTPException(status_code=404, detail="Chat session not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add chat message failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service internal error, please try again later")


@app.delete("/api/chat/sessions/{session_id}", tags=["AI - Chat History"])
async def delete_chat_session(session_id: str):
    """Delete a chat session and all its messages."""
    try:
        if chat_history_service.delete_session(session_id):
            return {"success": True, "message": "Chat session deleted"}
        raise HTTPException(status_code=404, detail="Chat session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete chat session failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service internal error, please try again later")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
