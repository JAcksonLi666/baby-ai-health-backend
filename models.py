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
    feeding: Optional[dict] = None
    growth: Optional[dict] = None
    insights: Optional[List[str]] = None


# ==================== 喂养记录模型 ====================

class FeedingType(str, Enum):
    breast = "breast"           # 母乳
    formula = "formula"        # 配方奶
    solid = "solid"            # 辅食
    water = "water"            # 喝水

class BreastSide(str, Enum):
    left = "left"
    right = "right"
    both = "both"

class FeedingRecordCreate(BaseModel):
    time: str = Field(..., description="喂养时间 YYYY-MM-DD HH:mm")
    feeding_type: FeedingType
    duration_minutes: Optional[int] = Field(None, ge=0, description="喂养时长(分钟)")
    amount_ml: Optional[float] = Field(None, ge=0, description="奶量(ml)")
    breast_side: Optional[BreastSide] = None
    solid_food: Optional[str] = Field(None, description="辅食名称")
    water_amount_ml: Optional[float] = Field(None, ge=0, description="喝水量(ml)")
    notes: Optional[str] = None

class FeedingRecordUpdate(BaseModel):
    time: Optional[str] = None
    feeding_type: Optional[FeedingType] = None
    duration_minutes: Optional[int] = None
    amount_ml: Optional[float] = None
    breast_side: Optional[BreastSide] = None
    solid_food: Optional[str] = None
    water_amount_ml: Optional[float] = None
    notes: Optional[str] = None

class FeedingRecordResponse(BaseModel):
    id: str
    time: str
    feeding_type: str
    duration_minutes: Optional[int] = None
    amount_ml: Optional[float] = None
    breast_side: Optional[str] = None
    solid_food: Optional[str] = None
    water_amount_ml: Optional[float] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ==================== 生长发育记录模型 ====================

class GrowthRecordCreate(BaseModel):
    record_date: str = Field(..., description="记录日期 YYYY-MM-DD")
    weight_kg: Optional[float] = Field(None, ge=0, le=150, description="体重(kg)")
    height_cm: Optional[float] = Field(None, ge=0, le=250, description="身高/身长(cm)")
    head_circumference_cm: Optional[float] = Field(None, ge=0, le=70, description="头围(cm)")
    temperature_c: Optional[float] = Field(None, ge=35, le=42, description="体温(°C)")
    notes: Optional[str] = None

class GrowthRecordUpdate(BaseModel):
    record_date: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    head_circumference_cm: Optional[float] = None
    temperature_c: Optional[float] = None
    notes: Optional[str] = None

class GrowthRecordResponse(BaseModel):
    id: str
    record_date: str
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    head_circumference_cm: Optional[float] = None
    temperature_c: Optional[float] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ==================== 提醒记录模型 ====================

class ReminderType(str, Enum):
    vaccine = "vaccine"       # 疫苗接种
    checkup = "checkup"       # 体检
    feeding = "feeding"       # 喂养提醒
    medicine = "medicine"     # 用药提醒
    other = "other"           # 其他

class ReminderStatus(str, Enum):
    pending = "pending"       # 待处理
    completed = "completed"   # 已完成
    overdue = "overdue"       # 已过期
    cancelled = "cancelled"   # 已取消

class RepeatType(str, Enum):
    none = "none"             # 不重复
    daily = "daily"           # 每天
    weekly = "weekly"         # 每周
    monthly = "monthly"       # 每月

class ReminderRecordCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="提醒标题")
    reminder_type: ReminderType
    reminder_date: str = Field(..., description="提醒日期 YYYY-MM-DD")
    reminder_time: Optional[str] = Field(None, description="提醒时间 HH:mm")
    repeat_type: RepeatType = RepeatType.none
    notes: Optional[str] = None
    status: ReminderStatus = ReminderStatus.pending

class ReminderRecordUpdate(BaseModel):
    title: Optional[str] = None
    reminder_type: Optional[ReminderType] = None
    reminder_date: Optional[str] = None
    reminder_time: Optional[str] = None
    repeat_type: Optional[RepeatType] = None
    notes: Optional[str] = None
    status: Optional[ReminderStatus] = None

class ReminderRecordResponse(BaseModel):
    id: str
    title: str
    reminder_type: str
    reminder_date: str
    reminder_time: Optional[str] = None
    repeat_type: str
    status: str
    notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ==================== Lab Report Parser Models ====================

class ReportType(str, Enum):
    """Enum for lab report types."""
    auto = "auto"
    blood = "blood"
    urine = "urine"
    liver = "liver"
    kidney = "kidney"
    blood_routine = "blood_routine"
    urine_routine = "urine_routine"
    liver_function = "liver_function"
    kidney_function = "kidney_function"

class LabReportParseRequest(BaseModel):
    """Request model for lab report parsing."""
    text: str = Field(..., description="OCR extracted text from lab report")
    report_type: ReportType = Field(default=ReportType.auto, description="Report type: auto/blood/urine/liver/kidney")
    age_months: int = Field(default=6, ge=0, le=144, description="Patient age in months")


class LabReportItem(BaseModel):
    """Single lab report indicator item."""
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    status: str = Field(default="normal", description="normal/low/high/critical")


class LabReportResponse(BaseModel):
    """Response model for lab report parsing and evaluation."""
    report_type: str
    items: List[LabReportItem] = []
    summary: str = ""
    abnormal_count: int = 0
    total_count: int = 0


# ==================== Symptom Checker Models ====================

class SymptomCheckRequest(BaseModel):
    symptoms: List[str] = Field(..., description="List of symptom names")
    age_months: int = Field(..., ge=0, le=144, description="Baby's age in months")
    duration_days: Optional[int] = Field(None, ge=0, description="Duration of symptoms in days")
    severity: Optional[int] = Field(None, ge=1, le=5, description="User-reported severity 1-5")

class SymptomAnalysis(BaseModel):
    category: str
    description: str
    possible_causes: List[str]
    severity: str
    related_knowledge: List[str]
    precautions: List[str]


# ==================== Chat History Models ====================

class MessageRole(str, Enum):
    """Enum for chat message roles."""
    user = "user"
    assistant = "assistant"
    system = "system"

class ChatSessionCreate(BaseModel):
    title: Optional[str] = None

class ChatMessageCreate(BaseModel):
    session_id: str
    role: MessageRole = Field(..., description="Message role: user/assistant/system")
    content: str

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str


# ==================== Knowledge Base Models ====================

class KnowledgeEntryCreate(BaseModel):
    title: str
    content: str
    source: Optional[str] = None
    keywords: List[str] = []
    category: Optional[str] = None
