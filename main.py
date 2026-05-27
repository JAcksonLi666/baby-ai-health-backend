from fastapi import FastAPI, Request, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
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
from database import init_db
from monitor import monitor

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

    # 初始化数据库
    try:
        init_db(migrate=True)
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.warning(f"数据库初始化失败，但服务仍可运行: {str(e)}")

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

# ==================== 性能监控中间件 ====================
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    monitor.record_request(str(request.url.path), request.method, response.status_code, duration)
    return response


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


# ==================== 数据导入导出 API ====================
from data_io import export_to_csv, export_to_excel, import_from_csv, import_from_excel, get_supported_types

@app.get("/api/export/types")
async def get_export_types():
    """获取支持的导入导出类型"""
    return {"success": True, "types": get_supported_types()}

@app.get("/api/export/{record_type}")
async def export_records(record_type: str, format: str = "csv", date_from: str = None, date_to: str = None):
    """导出记录（支持 csv 和 xlsx 格式）"""
    from database import get_db
    db = get_db()

    table_map = {
        "sleep": "sleep_records", "diaper": "diaper_records",
        "cry": "cry_records", "feeding": "feeding_records",
        "growth": "growth_records", "reminder": "reminder_records",
    }

    if record_type not in table_map:
        raise HTTPException(status_code=400, detail=f"不支持的记录类型: {record_type}")

    table = table_map[record_type]
    conditions = []
    params = []

    if date_from:
        conditions.append("created_at >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("created_at <= ?")
        params.append(date_to + "T23:59:59")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = db.execute(f"SELECT * FROM {table} {where} ORDER BY created_at DESC", params).fetchall()
    records = [dict(row) for row in rows]

    if format == "xlsx":
        content = export_to_excel(records, record_type)
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={record_type}_export.xlsx"}
        )
    else:
        content = export_to_csv(records, record_type)
        return StreamingResponse(
            iter([content]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={record_type}_export.csv"}
        )

@app.post("/api/import/{record_type}")
async def import_records(record_type: str, file: UploadFile):
    """导入记录（支持 csv 和 xlsx 格式）"""
    content = await file.read()
    filename = file.filename or ""

    try:
        if filename.endswith(".xlsx"):
            records = import_from_excel(content, record_type)
        elif filename.endswith(".csv"):
            records = import_from_csv(content.decode("utf-8"), record_type)
        else:
            raise HTTPException(status_code=400, detail="仅支持 .csv 和 .xlsx 格式")

        # 写入数据库
        from database import get_db
        from base_service_db import BaseRecordServiceDB

        table_map = {
            "sleep": ("sleep_records", "sleep", "start_time"),
            "diaper": ("diaper_records", "diaper", "time"),
            "cry": ("cry_records", "cry", "start_time"),
            "feeding": ("feeding_records", "feeding", "time"),
            "growth": ("growth_records", "growth", "record_date"),
            "reminder": ("reminder_records", "reminder", "reminder_date"),
        }

        if record_type not in table_map:
            raise HTTPException(status_code=400, detail=f"不支持的记录类型: {record_type}")

        table_name, prefix, date_field = table_map[record_type]
        service = BaseRecordServiceDB(table_name, date_field)

        imported = 0
        for record in records:
            # 过滤掉全空记录
            if all(v is None or v == "" for k, v in record.items()):
                continue
            service.create(prefix, record)
            imported += 1

        return {"success": True, "imported": imported, "total": len(records)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")


# ==================== 疫苗接种计划 API ====================
from vaccine_schedule import get_vaccine_schedule, get_recommended_vaccines, generate_vaccine_reminders

@app.get("/api/vaccine/schedule")
async def vaccine_schedule():
    """获取完整疫苗接种时间表"""
    return {"success": True, "schedule": get_vaccine_schedule()}

@app.get("/api/vaccine/recommend")
async def vaccine_recommend(birth_date: str):
    """根据出生日期获取推荐接种计划"""
    plan = get_recommended_vaccines(birth_date)
    return {"success": True, **plan}

@app.post("/api/vaccine/reminders")
async def vaccine_reminders(birth_date: str):
    """生成疫苗接种提醒"""
    reminders = generate_vaccine_reminders(birth_date)
    return {"success": True, "reminders": reminders, "count": len(reminders)}


# ==================== 性能监控 API ====================
@app.get("/api/monitor/stats")
async def monitor_stats():
    """获取性能统计"""
    return monitor.get_stats()

@app.get("/api/monitor/errors")
async def monitor_errors(limit: int = 50):
    """获取最近的错误请求"""
    return {"success": True, "errors": monitor.get_recent_errors(limit)}

@app.post("/api/monitor/reset")
async def monitor_reset():
    """重置性能统计"""
    monitor.reset()
    return {"success": True, "message": "性能统计已重置"}
