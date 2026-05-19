from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class UploadResponse(BaseModel):
    success: bool
    file_id: str
    filename: str
    message: str
    extracted_text: Optional[str] = None


class HealthRecord(BaseModel):
    id: str
    date: datetime
    record_type: str
    extracted_data: dict
    raw_text: str
    file_path: str


class AskRequest(BaseModel):
    question: str
    use_cloud: bool = False
    top_k: int = Field(default=3, ge=1, le=10)
    model: Optional[str] = Field(default="auto", description="模型名称，'auto' 表示自动选择最聪明的模型")


class AskResponse(BaseModel):
    success: bool
    answer: str
    sources: List[dict]
    model_used: str
    cloud_used: bool = False


class HealthMetrics(BaseModel):
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    is_abnormal: Optional[bool] = None


class ExtractedReport(BaseModel):
    report_date: Optional[str] = None
    report_type: Optional[str] = None
    metrics: List[HealthMetrics] = []
    raw_text: str
    recommendations: Optional[List[str]] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


# ==================== MVP 日常记录模型 ====================

class SleepType(str, Enum):
    night = "night"
    nap = "nap"

class DiaperType(str, Enum):
    pee = "pee"
    poop = "poop"
    both = "both"

class PoopColor(str, Enum):
    yellow = "yellow"
    green = "green"
    brown = "brown"
    black = "black"
    red = "red"
    white = "white"
    orange = "orange"

class PoopConsistency(str, Enum):
    watery = "watery"
    soft = "soft"
    normal = "normal"
    hard = "hard"
    pellet = "pellet"

class Amount(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"

class CryReason(str, Enum):
    hungry = "hungry"
    sleepy = "sleepy"
    diaper = "diaper"
    discomfort = "discomfort"
    pain = "pain"
    lonely = "lonely"
    overstimulated = "overstimulated"
    unknown = "unknown"

# Sleep record models
class SleepRecordCreate(BaseModel):
    start_time: str = Field(..., description="入睡时间 YYYY-MM-DD HH:mm")
    sleep_type: SleepType = Field(default=SleepType.night)
    notes: Optional[str] = ""
    is_ongoing: bool = False

class SleepRecordUpdate(BaseModel):
    end_time: Optional[str] = None
    sleep_type: Optional[SleepType] = None
    quality: Optional[int] = Field(None, ge=1, le=5)
    notes: Optional[str] = None
    is_ongoing: Optional[bool] = None

class SleepRecordResponse(BaseModel):
    id: str
    start_time: str
    end_time: Optional[str] = None
    sleep_type: str
    quality: Optional[int] = None
    duration_minutes: Optional[int] = None
    notes: Optional[str] = None
    is_ongoing: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

# Diaper record models
class DiaperRecordCreate(BaseModel):
    time: str = Field(..., description="时间 YYYY-MM-DD HH:mm")
    diaper_type: DiaperType
    poop_color: Optional[PoopColor] = None
    poop_consistency: Optional[PoopConsistency] = None
    amount: Optional[Amount] = None
    has_photo: bool = False
    notes: Optional[str] = None

class DiaperRecordUpdate(BaseModel):
    time: Optional[str] = None
    diaper_type: Optional[DiaperType] = None
    poop_color: Optional[PoopColor] = None
    poop_consistency: Optional[PoopConsistency] = None
    amount: Optional[Amount] = None
    has_photo: Optional[bool] = None
    notes: Optional[str] = None

class DiaperRecordResponse(BaseModel):
    id: str
    time: str
    diaper_type: str
    poop_color: Optional[str] = None
    poop_consistency: Optional[str] = None
    amount: Optional[str] = None
    has_photo: bool = False
    notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

# Cry record models
class CryRecordCreate(BaseModel):
    start_time: str = Field(..., description="开始时间 YYYY-MM-DD HH:mm")
    end_time: Optional[str] = None
    reason: Optional[CryReason] = None
    intensity: Optional[int] = Field(None, ge=1, le=5)
    soothing_method: Optional[str] = None
    has_audio: bool = False
    notes: Optional[str] = None

class CryRecordUpdate(BaseModel):
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    reason: Optional[CryReason] = None
    intensity: Optional[int] = Field(None, ge=1, le=5)
    soothing_method: Optional[str] = None
    has_audio: Optional[bool] = None
    notes: Optional[str] = None

class CryRecordResponse(BaseModel):
    id: str
    start_time: str
    end_time: Optional[str] = None
    reason: Optional[str] = None
    intensity: Optional[int] = None
    soothing_method: Optional[str] = None
    has_audio: bool = False
    notes: Optional[str] = None
    duration_minutes: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class TodaySummaryResponse(BaseModel):
    date: str
    sleep: dict
    diaper: dict
    cry: dict
    insights: Optional[List[str]] = None
