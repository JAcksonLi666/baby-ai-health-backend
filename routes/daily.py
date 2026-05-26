from fastapi import APIRouter, Query, HTTPException
from models import (
    SleepRecordCreate, SleepRecordUpdate, SleepRecordResponse,
    DiaperRecordCreate, DiaperRecordUpdate, DiaperRecordResponse,
    CryRecordCreate, CryRecordUpdate, CryRecordResponse,
    FeedingRecordCreate, FeedingRecordUpdate, FeedingRecordResponse,
    GrowthRecordCreate, GrowthRecordUpdate, GrowthRecordResponse,
    ReminderRecordCreate, ReminderRecordUpdate, ReminderRecordResponse,
    TodaySummaryResponse,
)
from daily_records import sleep_service, diaper_service, cry_service, feeding_service, growth_service, reminder_service
from datetime import date, datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["日常记录"])


@router.post("/api/sleep", response_model=SleepRecordResponse)
async def create_sleep_record(data: SleepRecordCreate):
    try:
        return sleep_service.create(data.model_dump())
    except Exception as e:
        logger.error(f"创建睡眠记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/api/sleep")
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


@router.get("/api/sleep/ongoing")
async def get_ongoing_sleep():
    try:
        return sleep_service.get_ongoing()
    except Exception as e:
        logger.error(f"获取进行中睡眠失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/api/sleep/{record_id}")
async def get_sleep_record(record_id: str):
    record = sleep_service.get_by_id(record_id)
    if record:
        return record
    raise HTTPException(status_code=404, detail="睡眠记录不存在")


@router.put("/api/sleep/{record_id}")
async def update_sleep_record(record_id: str, data: SleepRecordUpdate):
    record = sleep_service.update(record_id, data.model_dump(exclude_none=True))
    if record:
        return record
    raise HTTPException(status_code=404, detail="睡眠记录不存在")


@router.delete("/api/sleep/{record_id}")
async def delete_sleep_record(record_id: str):
    if sleep_service.delete(record_id):
        return {"success": True, "message": "睡眠记录已删除"}
    raise HTTPException(status_code=404, detail="睡眠记录不存在")


# --- 排泄记录 ---
@router.post("/api/diaper", response_model=DiaperRecordResponse)
async def create_diaper_record(data: DiaperRecordCreate):
    try:
        return diaper_service.create(data.model_dump())
    except Exception as e:
        logger.error(f"创建排泄记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/api/diaper")
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


@router.get("/api/diaper/{record_id}")
async def get_diaper_record(record_id: str):
    record = diaper_service.get_by_id(record_id)
    if record:
        return record
    raise HTTPException(status_code=404, detail="排泄记录不存在")


@router.put("/api/diaper/{record_id}")
async def update_diaper_record(record_id: str, data: DiaperRecordUpdate):
    record = diaper_service.update(record_id, data.model_dump(exclude_none=True))
    if record:
        return record
    raise HTTPException(status_code=404, detail="排泄记录不存在")


@router.delete("/api/diaper/{record_id}")
async def delete_diaper_record(record_id: str):
    if diaper_service.delete(record_id):
        return {"success": True, "message": "排泄记录已删除"}
    raise HTTPException(status_code=404, detail="排泄记录不存在")


# --- 哭声记录 ---
@router.get("/api/cry/analyze")
async def analyze_cry_reason():
    try:
        return cry_service.analyze_cry_reason()
    except Exception as e:
        logger.error(f"分析哭声原因失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/api/cry/ongoing")
async def get_ongoing_cry():
    try:
        return cry_service.get_ongoing()
    except Exception as e:
        logger.error(f"获取进行中哭闹失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.post("/api/cry", response_model=CryRecordResponse)
async def create_cry_record(data: CryRecordCreate):
    try:
        return cry_service.create(data.model_dump())
    except Exception as e:
        logger.error(f"创建哭声记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/api/cry")
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


@router.get("/api/cry/{record_id}")
async def get_cry_record(record_id: str):
    record = cry_service.get_by_id(record_id)
    if record:
        return record
    raise HTTPException(status_code=404, detail="哭声记录不存在")


@router.put("/api/cry/{record_id}")
async def update_cry_record(record_id: str, data: CryRecordUpdate):
    record = cry_service.update(record_id, data.model_dump(exclude_none=True))
    if record:
        return record
    raise HTTPException(status_code=404, detail="哭声记录不存在")


@router.delete("/api/cry/{record_id}")
async def delete_cry_record(record_id: str):
    if cry_service.delete(record_id):
        return {"success": True, "message": "哭声记录已删除"}
    raise HTTPException(status_code=404, detail="哭声记录不存在")


# --- 喂养记录 ---
@router.post("/api/feeding", response_model=FeedingRecordResponse)
async def create_feeding_record(data: FeedingRecordCreate):
    try:
        return feeding_service.create(data.model_dump())
    except Exception as e:
        logger.error(f"创建喂养记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/api/feeding")
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


@router.get("/api/feeding/{record_id}")
async def get_feeding_record(record_id: str):
    record = feeding_service.get_by_id(record_id)
    if record:
        return record
    raise HTTPException(status_code=404, detail="喂养记录不存在")


@router.put("/api/feeding/{record_id}")
async def update_feeding_record(record_id: str, data: FeedingRecordUpdate):
    record = feeding_service.update(record_id, data.model_dump(exclude_none=True))
    if record:
        return record
    raise HTTPException(status_code=404, detail="喂养记录不存在")


@router.delete("/api/feeding/{record_id}")
async def delete_feeding_record(record_id: str):
    if feeding_service.delete(record_id):
        return {"success": True, "message": "喂养记录已删除"}
    raise HTTPException(status_code=404, detail="喂养记录不存在")


# --- 生长发育记录 ---
@router.post("/api/growth", response_model=GrowthRecordResponse)
async def create_growth_record(data: GrowthRecordCreate):
    try:
        return growth_service.create(data.model_dump())
    except Exception as e:
        logger.error(f"创建生长发育记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/api/growth")
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


@router.get("/api/growth/latest")
async def get_latest_growth_record():
    record = growth_service.get_latest()
    if record:
        return {"success": True, "record": record}
    return {"success": False, "message": "暂无生长发育记录"}


@router.get("/api/growth/{record_id}")
async def get_growth_record(record_id: str):
    record = growth_service.get_by_id(record_id)
    if record:
        return record
    raise HTTPException(status_code=404, detail="生长发育记录不存在")


@router.put("/api/growth/{record_id}")
async def update_growth_record(record_id: str, data: GrowthRecordUpdate):
    record = growth_service.update(record_id, data.model_dump(exclude_none=True))
    if record:
        return record
    raise HTTPException(status_code=404, detail="生长发育记录不存在")


@router.delete("/api/growth/{record_id}")
async def delete_growth_record(record_id: str):
    if growth_service.delete(record_id):
        return {"success": True, "message": "生长发育记录已删除"}
    raise HTTPException(status_code=404, detail="生长发育记录不存在")


# --- 提醒记录 ---
@router.post("/api/reminder", response_model=ReminderRecordResponse)
async def create_reminder_record(data: ReminderRecordCreate):
    try:
        return reminder_service.create(data.model_dump())
    except Exception as e:
        logger.error(f"创建提醒记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/api/reminder")
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


@router.get("/api/reminder/pending")
async def get_pending_reminders():
    records = reminder_service.get_pending()
    return {"success": True, "records": records, "total": len(records)}


@router.get("/api/reminder/today")
async def get_today_reminders():
    records = reminder_service.get_today_reminders()
    return {"success": True, "records": records, "total": len(records)}


@router.get("/api/reminder/{record_id}")
async def get_reminder_record(record_id: str):
    record = reminder_service.get_by_id(record_id)
    if record:
        return record
    raise HTTPException(status_code=404, detail="提醒记录不存在")


@router.put("/api/reminder/{record_id}")
async def update_reminder_record(record_id: str, data: ReminderRecordUpdate):
    record = reminder_service.update(record_id, data.model_dump(exclude_none=True))
    if record:
        return record
    raise HTTPException(status_code=404, detail="提醒记录不存在")


@router.delete("/api/reminder/{record_id}")
async def delete_reminder_record(record_id: str):
    if reminder_service.delete(record_id):
        return {"success": True, "message": "提醒记录已删除"}
    raise HTTPException(status_code=404, detail="提醒记录不存在")


# --- 今日汇总 ---
@router.get("/api/today/summary", response_model=TodaySummaryResponse)
async def get_today_summary():
    try:
        today = date.today().isoformat()

        sleep_records = sleep_service.get_today_records()
        total_minutes = 0
        nap_count = 0
        nap_minutes = 0
        night_minutes = 0
        is_ongoing = False

        for r in sleep_records:
            if r.get("is_ongoing"):
                is_ongoing = True
                start = datetime.strptime(r["start_time"], "%Y-%m-%d %H:%M")
                diff = int((datetime.now() - start).total_seconds() / 60)
                total_minutes += diff
            else:
                duration = r.get("duration_minutes", 0)
                total_minutes += duration
                
                time_str = r.get("start_time", "")
                if time_str:
                    hour = int(time_str.split()[1].split(":")[0])
                    if 6 <= hour < 18:
                        nap_count += 1
                        nap_minutes += duration
                    else:
                        night_minutes += duration

        diaper_records = diaper_service.get_today_records()
        diaper_count = len(diaper_records)
        last_diaper = None
        abnormal_color = False
        
        if diaper_records:
            diaper_records.sort(key=lambda x: x["timestamp"], reverse=True)
            last_diaper = diaper_records[0]
            color = last_diaper.get("color")
            abnormal_color = color in ["red", "black", "white"]

        cry_records = cry_service.get_today_records()
        cry_count = len(cry_records)
        last_cry = None
        
        if cry_records:
            cry_records.sort(key=lambda x: x["timestamp"], reverse=True)
            last_cry = cry_records[0]

        feeding_records = feeding_service.get_today_records()
        feeding_count = len(feeding_records)
        total_milk_volume = sum(r.get("milk_volume", 0) for r in feeding_records)

        return {
            "success": True,
            "date": today,
            "sleep": {
                "total_minutes": total_minutes,
                "nap_count": nap_count,
                "nap_minutes": nap_minutes,
                "night_minutes": night_minutes,
                "is_ongoing": is_ongoing,
                "records": sleep_records
            },
            "diaper": {
                "count": diaper_count,
                "last_diaper": last_diaper,
                "abnormal_color": abnormal_color,
                "records": diaper_records
            },
            "cry": {
                "count": cry_count,
                "last_cry": last_cry,
                "records": cry_records
            },
            "feeding": {
                "count": feeding_count,
                "total_milk_volume": total_milk_volume,
                "records": feeding_records
            }
        }
    except Exception as e:
        logger.error(f"获取今日汇总失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")
