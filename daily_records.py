"""
日常记录服务 - JSON 文件存储
支持睡眠、排泄、哭声记录的增删改查
"""
import json
import os
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional
from config import RECORDS_DIR

RECORDS_DIR.mkdir(parents=True, exist_ok=True)


class BaseRecordService:
    """基础记录服务，提供通用 CRUD 操作"""

    def __init__(self, filename: str):
        self.file_path = RECORDS_DIR / filename
        self._ensure_file()

    def _ensure_file(self):
        if not self.file_path.exists():
            self.file_path.write_text("[]", encoding="utf-8")

    def _read_all(self) -> List[Dict]:
        content = self.file_path.read_text(encoding="utf-8")
        return json.loads(content) if content.strip() else []

    def _write_all(self, records: List[Dict]):
        self.file_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def _generate_id(self, prefix: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        import uuid
        unique = uuid.uuid4().hex[:6]
        return f"{prefix}_{timestamp}_{unique}"

    def create(self, data: Dict) -> Dict:
        records = self._read_all()
        record_id = self._generate_id(self.prefix)
        now = datetime.now().isoformat()
        record = {
            "id": record_id,
            **data,
            "created_at": now,
            "updated_at": now,
        }
        records.append(record)
        self._write_all(records)
        return record

    def get_by_id(self, record_id: str) -> Optional[Dict]:
        for r in self._read_all():
            if r["id"] == record_id:
                return r
        return None

    def update(self, record_id: str, data: Dict) -> Optional[Dict]:
        records = self._read_all()
        for i, r in enumerate(records):
            if r["id"] == record_id:
                data.pop("id", None)
                data.pop("created_at", None)
                data["updated_at"] = datetime.now().isoformat()
                records[i] = {**r, **data}
                self._write_all(records)
                return records[i]
        return None

    def delete(self, record_id: str) -> bool:
        records = self._read_all()
        new_records = [r for r in records if r["id"] != record_id]
        if len(new_records) < len(records):
            self._write_all(new_records)
            return True
        return False

    def list_records(
        self,
        limit: int = 50,
        offset: int = 0,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict:
        records = self._read_all()

        # Date filtering
        if start_date or end_date:
            filtered = []
            for r in records:
                r_date = self._extract_date(r)
                if r_date:
                    if start_date and r_date < start_date:
                        continue
                    if end_date and r_date > end_date:
                        continue
                filtered.append(r)
            records = filtered

        # Sort by created_at descending
        records.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        total = len(records)

        # Pagination
        if page and page_size:
            offset = (page - 1) * page_size
            limit = page_size

        paginated = records[offset:offset + limit]

        return {
            "success": True,
            "records": paginated,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def _extract_date(self, record: Dict) -> Optional[str]:
        """Extract date string from record for filtering. Override in subclass."""
        return None


class SleepRecordService(BaseRecordService):
    """睡眠记录服务"""

    def __init__(self):
        super().__init__("sleep_records.json")
        self.prefix = "sleep"

    def create(self, data: Dict) -> Dict:
        """创建睡眠记录，处理枚举值"""
        if isinstance(data.get("sleep_type"), str):
            data["sleep_type"] = data["sleep_type"].lower().strip()
            valid_types = {"night", "nap"}
            if data["sleep_type"] not in valid_types:
                data["sleep_type"] = "night"
        # Calculate duration if end_time provided
        if data.get("end_time") and data.get("start_time"):
            try:
                start = datetime.strptime(data["start_time"], "%Y-%m-%d %H:%M")
                end = datetime.strptime(data["end_time"], "%Y-%m-%d %H:%M")
                data["duration_minutes"] = int((end - start).total_seconds() / 60)
            except (ValueError, TypeError):
                pass
        return super().create(data)

    def update(self, record_id: str, data: Dict) -> Optional[Dict]:
        """更新睡眠记录，重新计算时长"""
        if data.get("end_time") or data.get("start_time"):
            record = self.get_by_id(record_id)
            if record:
                start_time = data.get("start_time") or record.get("start_time")
                end_time = data.get("end_time") or record.get("end_time")
                if start_time and end_time:
                    try:
                        start = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
                        end = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
                        data["duration_minutes"] = int((end - start).total_seconds() / 60)
                    except (ValueError, TypeError):
                        pass
        return super().update(record_id, data)

    def get_ongoing(self) -> Dict:
        """获取当前进行中的睡眠"""
        records = self._read_all()
        for r in records:
            if r.get("is_ongoing"):
                return {"success": True, "record": r}
        return {"success": False, "message": "当前没有进行中的睡眠"}

    def get_today_records(self) -> List[Dict]:
        """获取今天的所有睡眠记录"""
        today = date.today().isoformat()
        return [r for r in self._read_all() if r.get("start_time", "").startswith(today)]

    def _extract_date(self, record: Dict) -> Optional[str]:
        return record.get("start_time", "")[:10] if record.get("start_time") else None


class DiaperRecordService(BaseRecordService):
    """排泄记录服务"""

    def __init__(self):
        super().__init__("diaper_records.json")
        self.prefix = "diaper"

    def get_today_records(self) -> List[Dict]:
        today = date.today().isoformat()
        return [r for r in self._read_all() if r.get("time", "").startswith(today)]

    def _extract_date(self, record: Dict) -> Optional[str]:
        return record.get("time", "")[:10] if record.get("time") else None


class CryRecordService(BaseRecordService):
    """哭声记录服务"""

    def __init__(self):
        super().__init__("cry_records.json")
        self.prefix = "cry"

    def create(self, data: Dict) -> Dict:
        """创建哭声记录，计算时长"""
        if data.get("end_time") and data.get("start_time"):
            try:
                start = datetime.strptime(data["start_time"], "%Y-%m-%d %H:%M")
                end = datetime.strptime(data["end_time"], "%Y-%m-%d %H:%M")
                data["duration_minutes"] = int((end - start).total_seconds() / 60)
            except (ValueError, TypeError):
                pass
        return super().create(data)

    def update(self, record_id: str, data: Dict) -> Optional[Dict]:
        if data.get("end_time") or data.get("start_time"):
            record = self.get_by_id(record_id)
            if record:
                start_time = data.get("start_time") or record.get("start_time")
                end_time = data.get("end_time") or record.get("end_time")
                if start_time and end_time:
                    try:
                        start = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
                        end = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
                        data["duration_minutes"] = int((end - start).total_seconds() / 60)
                    except (ValueError, TypeError):
                        pass
        return super().update(record_id, data)

    def get_ongoing(self) -> Dict:
        """获取当前进行中的哭闹"""
        records = self._read_all()
        for r in records:
            if not r.get("end_time"):
                return {"success": True, "data": r}
        return {"success": False, "message": "当前没有进行中的哭闹"}

    def get_today_records(self) -> List[Dict]:
        today = date.today().isoformat()
        return [r for r in self._read_all() if r.get("start_time", "").startswith(today)]

    def analyze_cry_reason(self) -> Dict:
        """基于规则引擎分析哭声原因"""
        recent_sleeps = sleep_service.get_today_records()
        recent_diapers = diaper_service.get_today_records()
        recent_cries = self.get_today_records()

        now = datetime.now()
        reasons = {}

        # Check diaper: if last change > 2 hours ago
        if recent_diapers:
            last_diaper = recent_diapers[0]  # sorted desc
            try:
                last_time = datetime.strptime(last_diaper["time"], "%Y-%m-%d %H:%M")
                minutes_since_diaper = (now - last_time).total_seconds() / 60
                if minutes_since_diaper > 120:
                    reasons["diaper"] = 0.7
                elif minutes_since_diaper > 90:
                    reasons["diaper"] = 0.4
            except (ValueError, TypeError):
                pass
        else:
            reasons["diaper"] = 0.6

        # Check sleep: if last wake > 1.5 hours ago or no sleep
        if recent_sleeps:
            last_sleep = None
            for s in recent_sleeps:
                if not s.get("is_ongoing") and s.get("end_time"):
                    last_sleep = s
                    break
            if last_sleep:
                try:
                    wake_time = datetime.strptime(last_sleep["end_time"], "%Y-%m-%d %H:%M")
                    minutes_since_wake = (now - wake_time).total_seconds() / 60
                    if minutes_since_wake > 90:
                        reasons["sleepy"] = 0.6
                    elif minutes_since_wake > 60:
                        reasons["sleepy"] = 0.3
                except (ValueError, TypeError):
                    pass
        else:
            reasons["sleepy"] = 0.5

        # Check hunger: if last feeding was > 2 hours ago (approximate)
        hours_since_midnight = now.hour
        if hours_since_midnight < 10:
            reasons["hungry"] = 0.5

        # Check frequency
        cry_count = len(recent_cries)
        if cry_count >= 5:
            reasons["discomfort"] = 0.4

        # Default
        if not reasons:
            reasons["unknown"] = 0.5

        suggested_reasons = [
            {"reason": k, "confidence": v}
            for k, v in sorted(reasons.items(), key=lambda x: x[1], reverse=True)
        ]

        return {
            "success": True,
            "suggested_reasons": suggested_reasons,
            "analysis_basis": f"基于今日 {len(recent_sleeps)} 条睡眠、{len(recent_diapers)} 条排泄、{len(recent_cries)} 条哭声记录分析",
            "note": "MVP 规则引擎分析，后续将集成 AI 音频分类模型",
        }

    def _extract_date(self, record: Dict) -> Optional[str]:
        return record.get("start_time", "")[:10] if record.get("start_time") else None


# Module-level instances
sleep_service = SleepRecordService()
diaper_service = DiaperRecordService()
cry_service = CryRecordService()


class FeedingRecordService(BaseRecordService):
    """喂养记录服务"""

    def __init__(self):
        super().__init__("feeding_records.json")
        self.prefix = "feeding"

    def get_today_records(self) -> List[Dict]:
        today = date.today().isoformat()
        return [r for r in self._read_all() if r.get("time", "").startswith(today)]

    def _extract_date(self, record: Dict) -> Optional[str]:
        return record.get("time", "")[:10] if record.get("time") else None


class GrowthRecordService(BaseRecordService):
    """生长发育记录服务"""

    def __init__(self):
        super().__init__("growth_records.json")
        self.prefix = "growth"

    def get_today_records(self) -> List[Dict]:
        today = date.today().isoformat()
        return [r for r in self._read_all() if r.get("record_date", "").startswith(today)]

    def get_latest(self) -> Optional[Dict]:
        """获取最新的一条生长发育记录"""
        records = self._read_all()
        if not records:
            return None
        # Sort by record_date descending
        records.sort(key=lambda x: x.get("record_date", ""), reverse=True)
        return records[0]

    def _extract_date(self, record: Dict) -> Optional[str]:
        return record.get("record_date", "") if record.get("record_date") else None


class ReminderRecordService(BaseRecordService):
    """提醒记录服务"""

    def __init__(self):
        super().__init__("reminder_records.json")
        self.prefix = "reminder"

    def get_pending(self) -> List[Dict]:
        """获取待处理的提醒"""
        records = self._read_all()
        return [r for r in records if r.get("status") == "pending"]

    def get_today_reminders(self) -> List[Dict]:
        """获取今天的提醒"""
        today = date.today().isoformat()
        return [r for r in self._read_all() if r.get("reminder_date", "").startswith(today)]

    def _extract_date(self, record: Dict) -> Optional[str]:
        return record.get("reminder_date", "") if record.get("reminder_date") else None


# Module-level instances
feeding_service = FeedingRecordService()
growth_service = GrowthRecordService()
reminder_service = ReminderRecordService()
