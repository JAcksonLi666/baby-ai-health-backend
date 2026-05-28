"""
中国国家免疫规划疫苗接种时间表
根据《国家免疫规划疫苗儿童免疫程序（2021年版）》
"""
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple

# 疫苗接种时间表
# 格式: (疫苗名称, 接种月龄, 剂次, 疫苗类型)
VACCINE_SCHEDULE = [
    # 出生时
    ("乙肝疫苗", 0, 1, "hepatitis_b"),
    ("卡介苗", 0, 1, "bcg"),

    # 1月龄
    ("乙肝疫苗", 1, 2, "hepatitis_b"),

    # 2月龄
    ("脊灰灭活疫苗", 2, 1, "polio"),

    # 3月龄
    ("脊灰灭活疫苗", 3, 2, "polio"),
    ("百白破疫苗", 3, 1, "dtap"),

    # 4月龄
    ("脊灰灭活疫苗", 4, 3, "polio"),
    ("百白破疫苗", 4, 2, "dtap"),

    # 5月龄
    ("百白破疫苗", 5, 3, "dtap"),

    # 6月龄
    ("乙肝疫苗", 6, 3, "hepatitis_b"),
    ("A群流脑多糖疫苗", 6, 1, "meningococcal_a"),

    # 8月龄
    ("麻腮风疫苗", 8, 1, "mmr"),
    ("乙脑减毒活疫苗", 8, 1, "je"),

    # 9月龄
    ("A群流脑多糖疫苗", 9, 2, "meningococcal_a"),

    # 18月龄
    ("百白破疫苗", 18, 4, "dtap"),
    ("麻腮风疫苗", 18, 2, "mmr"),

    # 2岁
    ("乙脑减毒活疫苗", 24, 2, "je"),
    ("甲肝减毒活疫苗", 24, 1, "hepatitis_a"),

    # 3岁
    ("A+C群流脑多糖疫苗", 36, 1, "meningococcal_ac"),

    # 4岁
    ("脊灰灭活疫苗", 48, 4, "polio"),

    # 6岁
    ("白破疫苗", 72, 1, "dt"),
    ("A+C群流脑多糖疫苗", 72, 2, "meningococcal_ac"),
]

# 疫苗信息
VACCINE_INFO = {
    "hepatitis_b": {"name": "乙肝疫苗", "doses": 3, "description": "预防乙型肝炎"},
    "bcg": {"name": "卡介苗", "doses": 1, "description": "预防结核病"},
    "polio": {"name": "脊灰灭活疫苗", "doses": 4, "description": "预防脊髓灰质炎（小儿麻痹）"},
    "dtap": {"name": "百白破疫苗", "doses": 4, "description": "预防百日咳、白喉、破伤风"},
    "meningococcal_a": {"name": "A群流脑多糖疫苗", "doses": 2, "description": "预防A群流行性脑脊髓膜炎"},
    "mmr": {"name": "麻腮风疫苗", "doses": 2, "description": "预防麻疹、流行性腮腺炎、风疹"},
    "je": {"name": "乙脑减毒活疫苗", "doses": 2, "description": "预防流行性乙型脑炎"},
    "hepatitis_a": {"name": "甲肝减毒活疫苗", "doses": 1, "description": "预防甲型肝炎"},
    "meningococcal_ac": {"name": "A+C群流脑多糖疫苗", "doses": 2, "description": "预防A群C群流行性脑脊髓膜炎"},
    "dt": {"name": "白破疫苗", "doses": 1, "description": "预防白喉、破伤风"},
}


def get_vaccine_schedule() -> List[Dict]:
    """获取完整疫苗接种时间表"""
    schedule = []
    for name, month_age, dose, vaccine_type in VACCINE_SCHEDULE:
        info = VACCINE_INFO.get(vaccine_type, {})
        schedule.append({
            "vaccine_name": name,
            "vaccine_type": vaccine_type,
            "month_age": month_age,
            "dose": dose,
            "total_doses": info.get("doses", 1),
            "description": info.get("description", ""),
            "recommended_date": None,  # 需要根据出生日期计算
        })
    return schedule


def get_recommended_vaccines(birth_date: str, completed_vaccines: List[Dict] = None) -> Dict:
    """
    根据出生日期获取推荐疫苗接种计划

    Args:
        birth_date: 出生日期 (YYYY-MM-DD)
        completed_vaccines: 已完成的疫苗接种记录列表

    Returns:
        {
            "birth_date": "2026-01-15",
            "current_age_months": 6,
            "overdue": [...],      # 已过期未接种
            "upcoming": [...],     # 即将到期（30天内）
            "completed": [...],     # 已完成
            "future": [...]         # 未来接种
        }
    """
    birth = datetime.strptime(birth_date, "%Y-%m-%d").date()
    today = date.today()
    age_days = (today - birth).days
    age_months = age_days // 30

    # 构建已完成疫苗集合
    completed_set = set()
    if completed_vaccines:
        for v in completed_vaccines:
            key = f"{v.get('vaccine_type', '')}_{v.get('dose', 0)}"
            completed_set.add(key)

    overdue = []
    upcoming = []
    completed = []
    future = []

    for name, month_age, dose, vaccine_type in VACCINE_SCHEDULE:
        # 计算推荐接种日期
        recommended_date = birth + timedelta(days=month_age * 30)

        # 判断是否已完成
        key = f"{vaccine_type}_{dose}"
        is_completed = key in completed_set

        # 判断状态
        days_until = (recommended_date - today).days

        info = VACCINE_INFO.get(vaccine_type, {})
        entry = {
            "vaccine_name": name,
            "vaccine_type": vaccine_type,
            "month_age": month_age,
            "dose": dose,
            "total_doses": info.get("doses", 1),
            "recommended_date": recommended_date.isoformat(),
            "description": info.get("description", ""),
            "is_completed": is_completed,
            "days_until": days_until,
        }

        if is_completed:
            completed.append(entry)
        elif days_until < -30:
            overdue.append(entry)
        elif days_until < 0:
            overdue.append(entry)
        elif days_until <= 30:
            upcoming.append(entry)
        else:
            future.append(entry)

    # 排序
    overdue.sort(key=lambda x: x["recommended_date"])
    upcoming.sort(key=lambda x: x["recommended_date"])
    completed.sort(key=lambda x: x["recommended_date"])
    future.sort(key=lambda x: x["recommended_date"])

    return {
        "birth_date": birth_date,
        "current_age_months": age_months,
        "overdue": overdue,
        "upcoming": upcoming,
        "completed": completed,
        "future": future,
        "summary": {
            "total": len(VACCINE_SCHEDULE),
            "completed_count": len(completed),
            "overdue_count": len(overdue),
            "upcoming_count": len(upcoming),
            "completion_rate": f"{len(completed) / len(VACCINE_SCHEDULE) * 100:.1f}%" if VACCINE_SCHEDULE else "0%",
        }
    }


def generate_vaccine_reminders(birth_date: str, completed_vaccines: List[Dict] = None) -> List[Dict]:
    """生成疫苗接种提醒（用于自动创建提醒记录）"""
    plan = get_recommended_vaccines(birth_date, completed_vaccines)
    reminders = []

    for entry in plan["upcoming"] + plan["overdue"]:
        reminders.append({
            "title": f"{entry['vaccine_name']} 第{entry['dose']}剂",
            "reminder_type": "vaccine",
            "reminder_date": entry["recommended_date"][:10],
            "reminder_time": "09:00",
            "repeat_type": "none",
            "status": "pending",
            "notes": f"{entry['description']}（推荐月龄：{entry['month_age']}个月）",
        })

    return reminders
