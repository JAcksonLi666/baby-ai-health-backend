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
