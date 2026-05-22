"""
Lab Report Intelligent Parser Module

Provides structured parsing of pediatric lab reports (blood, urine, liver, kidney)
with age-specific reference ranges for Chinese children clinical standards.
Supports LLM-based parsing with regex fallback.
"""

import re
import json
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

from llm_service import llm_service

logger = logging.getLogger(__name__)


# ==================== Age Group Definitions ====================

AGE_GROUPS = {
    "infant": {"label": "Infant (0-1y)", "min_months": 0, "max_months": 12},
    "toddler": {"label": "Toddler (1-3y)", "min_months": 12, "max_months": 36},
    "preschool": {"label": "Preschool (3-6y)", "min_months": 36, "max_months": 72},
    "school": {"label": "School-age (6-12y)", "min_months": 72, "max_months": 144},
}


def get_age_group(age_months: int) -> str:
    """Determine age group from age in months."""
    if age_months <= 12:
        return "infant"
    elif age_months <= 36:
        return "toddler"
    elif age_months <= 72:
        return "preschool"
    else:
        return "school"


# ==================== Reference Ranges (Chinese Pediatric Clinical Standards) ====================
# Format: { "indicator_key": { "age_group": {"min": float, "max": float, "unit": str} } }
# Values based on Chinese pediatric clinical laboratory reference ranges.

REFERENCE_RANGES = {
    # ---- Blood Routine (血液常规) ----
    "WBC": {
        "name_cn": "白细胞",
        "name_en": "WBC",
        "unit": "×10⁹/L",
        "infant": {"min": 6.0, "max": 18.0},
        "toddler": {"min": 5.0, "max": 14.0},
        "preschool": {"min": 4.5, "max": 12.0},
        "school": {"min": 4.0, "max": 10.0},
    },
    "RBC": {
        "name_cn": "红细胞",
        "name_en": "RBC",
        "unit": "×10¹²/L",
        "infant": {"min": 3.5, "max": 5.5},
        "toddler": {"min": 3.8, "max": 5.2},
        "preschool": {"min": 4.0, "max": 5.2},
        "school": {"min": 4.0, "max": 5.5},
    },
    "HGB": {
        "name_cn": "血红蛋白",
        "name_en": "HGB",
        "unit": "g/L",
        "infant": {"min": 100, "max": 140},
        "toddler": {"min": 110, "max": 145},
        "preschool": {"min": 115, "max": 150},
        "school": {"min": 120, "max": 160},
    },
    "HCT": {
        "name_cn": "红细胞压积",
        "name_en": "HCT",
        "unit": "%",
        "infant": {"min": 30, "max": 42},
        "toddler": {"min": 33, "max": 42},
        "preschool": {"min": 34, "max": 43},
        "school": {"min": 37, "max": 48},
    },
    "MCV": {
        "name_cn": "平均红细胞体积",
        "name_en": "MCV",
        "unit": "fL",
        "infant": {"min": 75, "max": 95},
        "toddler": {"min": 73, "max": 90},
        "preschool": {"min": 75, "max": 90},
        "school": {"min": 78, "max": 94},
    },
    "MCH": {
        "name_cn": "平均红细胞血红蛋白",
        "name_en": "MCH",
        "unit": "pg",
        "infant": {"min": 24, "max": 32},
        "toddler": {"min": 23, "max": 31},
        "preschool": {"min": 24, "max": 31},
        "school": {"min": 26, "max": 34},
    },
    "MCHC": {
        "name_cn": "平均红细胞血红蛋白浓度",
        "name_en": "MCHC",
        "unit": "g/L",
        "infant": {"min": 300, "max": 370},
        "toddler": {"min": 310, "max": 370},
        "preschool": {"min": 320, "max": 370},
        "school": {"min": 320, "max": 370},
    },
    "PLT": {
        "name_cn": "血小板",
        "name_en": "PLT",
        "unit": "×10⁹/L",
        "infant": {"min": 150, "max": 400},
        "toddler": {"min": 150, "max": 400},
        "preschool": {"min": 150, "max": 400},
        "school": {"min": 150, "max": 400},
    },
    "NEUT": {
        "name_cn": "中性粒细胞%",
        "name_en": "NEUT%",
        "unit": "%",
        "infant": {"min": 15, "max": 40},
        "toddler": {"min": 25, "max": 55},
        "preschool": {"min": 35, "max": 65},
        "school": {"min": 45, "max": 75},
    },

    # ---- Urine Routine (尿液常规) ----
    "pH": {
        "name_cn": "pH",
        "name_en": "pH",
        "unit": "",
        "infant": {"min": 4.5, "max": 8.0},
        "toddler": {"min": 4.5, "max": 8.0},
        "preschool": {"min": 5.0, "max": 8.0},
        "school": {"min": 5.0, "max": 8.0},
    },
    "SG": {
        "name_cn": "比重",
        "name_en": "SG",
        "unit": "",
        "infant": {"min": 1.002, "max": 1.020},
        "toddler": {"min": 1.005, "max": 1.025},
        "preschool": {"min": 1.005, "max": 1.025},
        "school": {"min": 1.005, "max": 1.030},
    },
    "PRO": {
        "name_cn": "蛋白",
        "name_en": "PRO",
        "unit": "",
        "infant": {"min": 0, "max": 0},  # negative
        "toddler": {"min": 0, "max": 0},
        "preschool": {"min": 0, "max": 0},
        "school": {"min": 0, "max": 0},
    },
    "GLU": {
        "name_cn": "葡萄糖",
        "name_en": "GLU",
        "unit": "",
        "infant": {"min": 0, "max": 0},  # negative
        "toddler": {"min": 0, "max": 0},
        "preschool": {"min": 0, "max": 0},
        "school": {"min": 0, "max": 0},
    },
    "KET": {
        "name_cn": "酮体",
        "name_en": "KET",
        "unit": "",
        "infant": {"min": 0, "max": 0},  # negative
        "toddler": {"min": 0, "max": 0},
        "preschool": {"min": 0, "max": 0},
        "school": {"min": 0, "max": 0},
    },
    "BIL": {
        "name_cn": "胆红素",
        "name_en": "BIL",
        "unit": "",
        "infant": {"min": 0, "max": 0},  # negative
        "toddler": {"min": 0, "max": 0},
        "preschool": {"min": 0, "max": 0},
        "school": {"min": 0, "max": 0},
    },
    "URO": {
        "name_cn": "尿胆原",
        "name_en": "URO",
        "unit": "",
        "infant": {"min": 0, "max": 1},  # normal <= 1+
        "toddler": {"min": 0, "max": 1},
        "preschool": {"min": 0, "max": 1},
        "school": {"min": 0, "max": 1},
    },
    "BLD": {
        "name_cn": "隐血",
        "name_en": "BLD",
        "unit": "",
        "infant": {"min": 0, "max": 0},  # negative
        "toddler": {"min": 0, "max": 0},
        "preschool": {"min": 0, "max": 0},
        "school": {"min": 0, "max": 0},
    },
    "NIT": {
        "name_cn": "亚硝酸盐",
        "name_en": "NIT",
        "unit": "",
        "infant": {"min": 0, "max": 0},  # negative
        "toddler": {"min": 0, "max": 0},
        "preschool": {"min": 0, "max": 0},
        "school": {"min": 0, "max": 0},
    },
    "UWBC": {
        "name_cn": "白细胞",
        "name_en": "WBC",
        "unit": "/HPF",
        "infant": {"min": 0, "max": 5},
        "toddler": {"min": 0, "max": 5},
        "preschool": {"min": 0, "max": 5},
        "school": {"min": 0, "max": 5},
    },

    # ---- Liver Function (肝功能) ----
    "ALT": {
        "name_cn": "谷丙转氨酶",
        "name_en": "ALT",
        "unit": "U/L",
        "infant": {"min": 5, "max": 50},
        "toddler": {"min": 5, "max": 45},
        "preschool": {"min": 5, "max": 40},
        "school": {"min": 5, "max": 40},
    },
    "AST": {
        "name_cn": "谷草转氨酶",
        "name_en": "AST",
        "unit": "U/L",
        "infant": {"min": 15, "max": 60},
        "toddler": {"min": 10, "max": 50},
        "preschool": {"min": 10, "max": 45},
        "school": {"min": 10, "max": 40},
    },
    "TBIL": {
        "name_cn": "总胆红素",
        "name_en": "TBIL",
        "unit": "μmol/L",
        "infant": {"min": 3.4, "max": 20.5},
        "toddler": {"min": 3.4, "max": 17.1},
        "preschool": {"min": 3.4, "max": 17.1},
        "school": {"min": 3.4, "max": 17.1},
    },
    "DBIL": {
        "name_cn": "直接胆红素",
        "name_en": "DBIL",
        "unit": "μmol/L",
        "infant": {"min": 0, "max": 6.8},
        "toddler": {"min": 0, "max": 5.1},
        "preschool": {"min": 0, "max": 5.1},
        "school": {"min": 0, "max": 6.8},
    },
    "ALB": {
        "name_cn": "白蛋白",
        "name_en": "ALB",
        "unit": "g/L",
        "infant": {"min": 28, "max": 44},
        "toddler": {"min": 32, "max": 48},
        "preschool": {"min": 35, "max": 52},
        "school": {"min": 37, "max": 55},
    },
    "TP": {
        "name_cn": "总蛋白",
        "name_en": "TP",
        "unit": "g/L",
        "infant": {"min": 46, "max": 70},
        "toddler": {"min": 58, "max": 76},
        "preschool": {"min": 60, "max": 80},
        "school": {"min": 62, "max": 82},
    },

    # ---- Kidney Function (肾功能) ----
    "BUN": {
        "name_cn": "尿素氮",
        "name_en": "BUN",
        "unit": "mmol/L",
        "infant": {"min": 1.8, "max": 6.4},
        "toddler": {"min": 2.5, "max": 6.4},
        "preschool": {"min": 2.5, "max": 6.8},
        "school": {"min": 2.9, "max": 7.1},
    },
    "CREA": {
        "name_cn": "肌酐",
        "name_en": "CREA",
        "unit": "μmol/L",
        "infant": {"min": 18, "max": 40},
        "toddler": {"min": 20, "max": 45},
        "preschool": {"min": 25, "max": 53},
        "school": {"min": 30, "max": 62},
    },
    "UA": {
        "name_cn": "尿酸",
        "name_en": "UA",
        "unit": "μmol/L",
        "infant": {"min": 90, "max": 360},
        "toddler": {"min": 120, "max": 380},
        "preschool": {"min": 150, "max": 420},
        "school": {"min": 180, "max": 480},
    },
}


# ==================== Indicator Alias Mapping ====================
# Maps common OCR variations to canonical indicator keys.

INDICATOR_ALIASES: Dict[str, str] = {
    # Blood routine
    "wbc": "WBC",
    "白细胞": "WBC",
    "白细胞计数": "WBC",
    "white blood cell": "WBC",
    "rbc": "RBC",
    "红细胞": "RBC",
    "红细胞计数": "RBC",
    "red blood cell": "RBC",
    "hgb": "HGB",
    "hb": "HGB",
    "血红蛋白": "HGB",
    "hemoglobin": "HGB",
    "hct": "HCT",
    "红细胞压积": "HCT",
    "红细胞比容": "HCT",
    "hematocrit": "HCT",
    "mcv": "MCV",
    "平均红细胞体积": "MCV",
    "mean corpuscular volume": "MCV",
    "mch": "MCH",
    "平均红细胞血红蛋白": "MCH",
    "平均红细胞血红蛋白量": "MCH",
    "mean corpuscular hemoglobin": "MCH",
    "mchc": "MCHC",
    "平均红细胞血红蛋白浓度": "MCHC",
    "mean corpuscular hemoglobin concentration": "MCHC",
    "plt": "PLT",
    "血小板": "PLT",
    "血小板计数": "PLT",
    "platelet": "PLT",
    "neut": "NEUT",
    "中性粒细胞": "NEUT",
    "中性粒细胞%": "NEUT",
    "中性粒细胞比率": "NEUT",
    "neutrophil": "NEUT",
    "neut%": "NEUT",

    # Urine routine
    "ph": "pH",
    "sg": "SG",
    "比重": "SG",
    "尿比重": "SG",
    "specific gravity": "SG",
    "pro": "PRO",
    "蛋白": "PRO",
    "尿蛋白": "PRO",
    "蛋白质": "PRO",
    "protein": "PRO",
    "glu": "GLU",
    "葡萄糖": "GLU",
    "尿糖": "GLU",
    "glucose": "GLU",
    "ket": "KET",
    "酮体": "KET",
    "尿酮体": "KET",
    "ketone": "KET",
    "bil": "BIL",
    "胆红素": "BIL",
    "尿胆红素": "BIL",
    "bilirubin": "BIL",
    "uro": "URO",
    "尿胆原": "URO",
    "urobilinogen": "URO",
    "bld": "BLD",
    "隐血": "BLD",
    "尿隐血": "BLD",
    "潜血": "BLD",
    "occult blood": "BLD",
    "nit": "NIT",
    "亚硝酸盐": "NIT",
    "尿亚硝酸盐": "NIT",
    "nitrite": "NIT",
    "uwbc": "UWBC",
    "尿白细胞": "UWBC",
    "白细胞/HPF": "UWBC",
    "urine wbc": "UWBC",

    # Liver function
    "alt": "ALT",
    "谷丙转氨酶": "ALT",
    "丙氨酸氨基转移酶": "ALT",
    "alanine aminotransferase": "ALT",
    "gpt": "ALT",
    "ast": "AST",
    "谷草转氨酶": "AST",
    "天冬氨酸氨基转移酶": "AST",
    "aspartate aminotransferase": "AST",
    "got": "AST",
    "tbil": "TBIL",
    "总胆红素": "TBIL",
    "total bilirubin": "TBIL",
    "dbil": "DBIL",
    "直接胆红素": "DBIL",
    "direct bilirubin": "DBIL",
    "alb": "ALB",
    "白蛋白": "ALB",
    "albumin": "ALB",
    "tp": "TP",
    "总蛋白": "TP",
    "total protein": "TP",

    # Kidney function
    "bun": "BUN",
    "尿素氮": "BUN",
    "blood urea nitrogen": "BUN",
    "crea": "CREA",
    "肌酐": "CREA",
    "血肌酐": "CREA",
    "creatinine": "CREA",
    "cr": "CREA",
    "ua": "UA",
    "尿酸": "UA",
    "血尿酸": "UA",
    "uric acid": "UA",
}


# ==================== Report Type Detection ====================

REPORT_TYPE_KEYWORDS = {
    "blood": [
        "血常规", "血液常规", "全血细胞计数", "CBC", "complete blood count",
        "白细胞", "红细胞", "血红蛋白", "血小板", "WBC", "RBC", "HGB", "PLT",
    ],
    "urine": [
        "尿常规", "尿液常规", "尿液分析", "urinalysis", "尿检",
        "尿蛋白", "尿糖", "尿酮体", "尿胆原", "隐血", "PRO", "GLU", "KET",
    ],
    "liver": [
        "肝功能", "肝功", "liver function", "肝功能检查",
        "谷丙转氨酶", "谷草转氨酶", "ALT", "AST", "总胆红素", "白蛋白",
    ],
    "kidney": [
        "肾功能", "肾功", "kidney function", "肾功能检查",
        "尿素氮", "肌酐", "尿酸", "BUN", "CREA", "UA",
    ],
}


def detect_report_type(text: str) -> str:
    """Auto-detect report type from OCR text content."""
    if not text:
        return "blood"  # default

    text_lower = text.lower()
    scores: Dict[str, int] = {}

    for report_type, keywords in REPORT_TYPE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw.lower() in text_lower:
                score += 1
        scores[report_type] = score

    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return "blood"  # default fallback
    return best_type


# ==================== LLM Prompt for Structured Parsing ====================

LLM_PARSE_PROMPT_TEMPLATE = """You are a professional clinical laboratory report parser for Chinese pediatric patients.
Your task is to extract structured data from the OCR text of a lab report.

IMPORTANT RULES:
1. Extract ALL test items with their values, units, and reference ranges.
2. For each item, determine the status: "normal", "low", "high", or "critical".
3. The report type is: {report_type}
4. Output ONLY valid JSON, no other text.

Output format (strict JSON):
{{
  "report_type": "{report_type}",
  "items": [
    {{
      "name": "indicator name (e.g., WBC)",
      "value": 5.2,
      "unit": "×10⁹/L",
      "reference": "4-10",
      "status": "normal"
    }}
  ]
}}

Common indicators for {report_type} reports:
{indicator_hints}

OCR text to parse:
---
{text}
---

Please output the JSON now:"""


def _build_indicator_hints(report_type: str) -> str:
    """Build a hint string listing common indicators for the given report type."""
    type_indicators = {
        "blood": ["WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "NEUT%"],
        "urine": ["pH", "SG", "PRO", "GLU", "KET", "BIL", "URO", "BLD", "NIT", "WBC"],
        "liver": ["ALT", "AST", "TBIL", "DBIL", "ALB", "TP"],
        "kidney": ["BUN", "CREA", "UA"],
    }
    indicators = type_indicators.get(report_type, [])
    hints = []
    for ind_key in indicators:
        ref = REFERENCE_RANGES.get(ind_key)
        if ref:
            hints.append(f"  - {ref['name_cn']} ({ref['name_en']}): unit={ref['unit']}")
    return "\n".join(hints) if hints else "  - (see OCR text)"


# ==================== Regex Fallback Parser ====================

# Patterns for extracting indicator name, value, unit, and reference range from text.
# Supports various formats found in Chinese lab reports.

REGEX_PATTERNS = [
    # Format: "WBC  5.2  ×10⁹/L  4-10" or "WBC: 5.2 ×10⁹/L (4-10)"
    re.compile(
        r'([\w\u4e00-\u9fa5%]+(?:\([^)]+\))?)'  # name (allow parentheses for Chinese names)
        r'[\s:：]*'
        r'([+-]?\d+\.?\d*)'  # value
        r'[\s]*'
        r'([^0-9\n\r]*?)'  # unit (non-numeric)
        r'(?:[\s]*[（(]?\s*([\d.]+)\s*[-~—到至]\s*([\d.]+)\s*[）)]?)?'  # optional reference range
        r'[\s]*(?:\n|$)',
        re.MULTILINE,
    ),
    # Format: "白细胞(WBC) 5.2 ×10⁹/L 参考值: 4-10"
    re.compile(
        r'([\w\u4e00-\u9fa5]+)'
        r'(?:[\s]*[（(]\s*([\w\u4e00-\u9fa5]+)\s*[）)])?'  # optional alias in parens
        r'[\s:：]*'
        r'([+-]?\d+\.?\d*)'  # value
        r'[\s]*'
        r'([^0-9\n\r参考参]*?)'  # unit
        r'(?:参考[值范围]*[：:]\s*([\d.]+)\s*[-~—到至]\s*([\d.]+))?'  # reference
        r'[\s]*(?:\n|$)',
        re.MULTILINE,
    ),
]


def _resolve_indicator_key(name: str) -> Optional[str]:
    """Resolve an indicator name to its canonical key using aliases."""
    name_clean = name.strip()
    # Direct match
    if name_clean in INDICATOR_ALIASES:
        return INDICATOR_ALIASES[name_clean]
    # Case-insensitive match
    name_lower = name_clean.lower()
    for alias, key in INDICATOR_ALIASES.items():
        if alias.lower() == name_lower:
            return key
    return None


def _parse_qualitative_value(value_str: str) -> Tuple[float, str]:
    """Parse qualitative values like Negative, +, ++, +++ into numeric scores."""
    value_str = value_str.strip().lower()
    qualitative_map = {
        "阴性": (0, "阴性"),
        "negative": (0, "阴性"),
        "-": (0, "阴性"),
        "neg": (0, "阴性"),
        "±": (0.5, "±"),
        "+": (1, "+"),
        "1+": (1, "+"),
        "++": (2, "++"),
        "2+": (2, "++"),
        "+++": (3, "+++"),
        "3+": (3, "+++"),
        "++++": (4, "++++"),
        "4+": (4, "++++"),
        "弱阳性": (0.5, "弱阳性"),
        "弱阳性(±)": (0.5, "±"),
        "positive": (1, "阳性"),
        "阳性": (1, "阳性"),
    }
    for key, (score, label) in qualitative_map.items():
        if value_str == key:
            return score, label
    # Try numeric parse
    try:
        return float(value_str), value_str
    except ValueError:
        return 0, value_str


def _parse_with_regex(text: str, report_type: str = "auto") -> dict:
    """Fallback regex-based parser when LLM is unavailable."""
    if report_type == "auto":
        report_type = detect_report_type(text)

    items = []
    seen_names = set()

    for pattern in REGEX_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if not groups:
                continue

            # Try to extract name and value from different pattern formats
            if len(groups) >= 3:
                name = groups[0].strip()
                # Check if there's an alias in parens
                alias = groups[1] if len(groups) > 3 and groups[1] else None
                value_str = groups[2] if len(groups) > 3 else groups[1]

                # Determine the actual name and alias
                if alias and value_str:
                    # Format 2: name, alias, value, unit, ref_min, ref_max
                    indicator_name = alias.strip()
                elif value_str:
                    indicator_name = name
                else:
                    continue

                # Try to parse value as float
                try:
                    value = float(value_str)
                except (ValueError, TypeError):
                    # Try qualitative parse
                    value, _ = _parse_qualitative_value(value_str)
                    if value == 0 and value_str.strip() not in ("0", "阴性", "negative", "-"):
                        continue

                # Skip if already seen
                if indicator_name in seen_names:
                    continue
                seen_names.add(indicator_name)

                # Resolve to canonical key
                key = _resolve_indicator_key(indicator_name)
                if not key:
                    # Try with the first name too
                    key = _resolve_indicator_key(name)
                if not key:
                    continue

                ref = REFERENCE_RANGES.get(key, {})
                unit = ref.get("unit", "")

                # Extract reference range from match
                ref_min = groups[-2] if len(groups) >= 5 and groups[-2] else None
                ref_max = groups[-1] if len(groups) >= 5 and groups[-1] else None
                if ref_min and ref_max:
                    reference_range = f"{ref_min}-{ref_max}"
                else:
                    # Use built-in reference (infant default)
                    age_range = ref.get("infant", {})
                    if age_range:
                        reference_range = f"{age_range['min']}-{age_range['max']}"
                    else:
                        reference_range = ""

                items.append({
                    "name": ref.get("name_en", indicator_name),
                    "value": value,
                    "unit": unit,
                    "reference": reference_range,
                    "status": "normal",  # Will be evaluated later
                })

    return {
        "report_type": report_type,
        "items": items,
    }


# ==================== Main Service Class ====================


class LabReportParser:
    """Intelligent lab report parser with LLM-based parsing and regex fallback."""

    def __init__(self):
        self._eval_cache: Dict[str, dict] = {}
        logger.info("Lab report parser initialized")

    async def parse_with_llm(self, text: str, report_type: str = "auto") -> dict:
        """
        Parse OCR text into structured JSON using LLM.
        Falls back to regex parsing if LLM is unavailable.

        Args:
            text: OCR extracted text from lab report
            report_type: "auto" for auto-detection, or one of "blood"/"urine"/"liver"/"kidney"

        Returns:
            dict with keys: report_type, items (list of indicator dicts)
        """
        import asyncio

        # Auto-detect report type if not specified
        if report_type == "auto":
            report_type = self.detect_report_type(text)
            logger.info(f"Auto-detected report type: {report_type}")

        # Check if Ollama is available (run in thread pool to avoid blocking)
        try:
            llm_available = await asyncio.to_thread(llm_service.check_ollama_health)
        except Exception:
            llm_available = False

        if llm_available:
            try:
                result = await self._parse_via_llm(text, report_type)
                if result and result.get("items"):
                    logger.info(f"LLM parsing succeeded, extracted {len(result['items'])} items")
                    return result
                else:
                    logger.warning("LLM returned empty result, falling back to regex")
            except Exception as e:
                logger.error(f"LLM parsing failed: {str(e)}, falling back to regex")
        else:
            logger.warning("Ollama not available, using regex parsing")

        # Fallback to regex parsing
        return self._parse_with_regex(text, report_type)

    async def _parse_via_llm(self, text: str, report_type: str) -> Optional[dict]:
        """Internal method to call LLM for structured parsing."""
        import asyncio

        indicator_hints = _build_indicator_hints(report_type)
        prompt = LLM_PARSE_PROMPT_TEMPLATE.format(
            report_type=report_type,
            indicator_hints=indicator_hints,
            text=text,
        )

        # Use local LLM with low temperature for structured output
        # Run in thread pool to avoid blocking the event loop
        model = llm_service.select_smartest_model()
        result = await asyncio.to_thread(
            llm_service.generate_local,
            prompt=prompt,
            model=model,
            temperature=0.1,
            max_tokens=4096,
        )

        if not result.get("success"):
            return None

        response_text = result["response"].strip()

        # Extract JSON from response (handle markdown code blocks)
        json_str = self._extract_json_from_response(response_text)
        if not json_str:
            logger.warning("Failed to extract JSON from LLM response")
            return None

        try:
            parsed = json.loads(json_str)
            # Validate structure
            if "items" not in parsed:
                parsed["items"] = []
            if "report_type" not in parsed:
                parsed["report_type"] = report_type

            # Normalize items
            normalized_items = []
            for item in parsed.get("items", []):
                if not isinstance(item, dict):
                    continue
                normalized_item = {
                    "name": str(item.get("name", "")),
                    "value": self._safe_float(item.get("value")),
                    "unit": str(item.get("unit", "")),
                    "reference": str(item.get("reference", "")),
                    "status": str(item.get("status", "normal")),
                }
                if normalized_item["name"]:
                    normalized_items.append(normalized_item)

            parsed["items"] = normalized_items
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {str(e)}")
            return None

    def _extract_json_from_response(self, text: str) -> Optional[str]:
        """Extract JSON string from LLM response, handling markdown code blocks."""
        # Try to find JSON in code blocks
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Try to find raw JSON object
        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, text)
        if matches:
            # Return the longest match (most likely the complete JSON)
            return max(matches, key=len)

        return None

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert a value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def evaluate_results(self, parsed_data: dict, age_months: int) -> dict:
        """
        Evaluate lab results against age-specific reference ranges.
        Uses LRU cache to avoid re-evaluating identical inputs.
        """
        if not parsed_data or not parsed_data.get("items"):
            return {
                "report_type": parsed_data.get("report_type", "unknown"),
                "items": [],
                "summary": "No items to evaluate.",
                "abnormal_count": 0,
                "total_count": 0,
            }

        # Build cache key from items and age
        cache_data = json.dumps(
            {"items": parsed_data["items"], "age_months": age_months},
            sort_keys=True, ensure_ascii=False,
        )
        cache_key = hashlib.md5(cache_data.encode()).hexdigest()

        # Check cache
        cached = self._eval_cache.get(cache_key)
        if cached:
            return cached

        result = self._evaluate_impl(parsed_data, age_months)

        # Store in cache (limit size)
        if len(self._eval_cache) > 100:
            # Evict oldest entries
            keys = list(self._eval_cache.keys())[:50]
            for k in keys:
                del self._eval_cache[k]
        self._eval_cache[cache_key] = result

        return result

    def _evaluate_impl(self, parsed_data: dict, age_months: int) -> dict:

        age_group = get_age_group(age_months)
        age_label = AGE_GROUPS[age_group]["label"]

        evaluated_items = []
        abnormal_items = []
        critical_items = []

        for item in parsed_data["items"]:
            name = item.get("name", "")
            value = item.get("value")
            unit = item.get("unit", "")
            reference = item.get("reference", "")

            # Try to resolve indicator key
            key = _resolve_indicator_key(name)
            ref_data = REFERENCE_RANGES.get(key) if key else None

            status = "normal"
            evaluated_ref = reference

            if ref_data and value is not None:
                age_ref = ref_data.get(age_group)
                if age_ref:
                    ref_min = age_ref["min"]
                    ref_max = age_ref["max"]
                    evaluated_ref = f"{ref_min}-{ref_max}"
                    unit = unit or ref_data.get("unit", "")

                    # Check for critical values
                    is_critical = False

                    if key in ("PLT",):
                        # Platelet critical thresholds
                        if value < 50 or value > 1000:
                            is_critical = True
                    elif key in ("HGB",):
                        # Hemoglobin critical thresholds
                        if value < 60:
                            is_critical = True
                    elif key in ("WBC",):
                        # WBC critical thresholds
                        if value < 2.0 or value > 30.0:
                            is_critical = True
                    elif key in ("CREA",):
                        # Creatinine critical thresholds
                        if value > 200:
                            is_critical = True
                    elif key in ("ALT", "AST"):
                        # Liver enzymes critical thresholds
                        if value > 500:
                            is_critical = True

                    if is_critical:
                        status = "critical"
                        critical_items.append(item)
                    elif value < ref_min:
                        status = "low"
                        abnormal_items.append(item)
                    elif value > ref_max:
                        status = "high"
                        abnormal_items.append(item)

            evaluated_items.append({
                "name": name,
                "value": value,
                "unit": unit,
                "reference_range": evaluated_ref,
                "status": status,
            })

        # Generate summary
        total_count = len(evaluated_items)
        abnormal_count = len(abnormal_items)
        critical_count = len(critical_items)

        summary_parts = [f"Age group: {age_label} ({age_months} months)."]
        summary_parts.append(f"Total {total_count} indicators evaluated.")

        if critical_count > 0:
            critical_names = ", ".join([it.get("name", "") for it in critical_items])
            summary_parts.append(
                f"CRITICAL: {critical_count} critical value(s) detected: {critical_names}. "
                "Please seek immediate medical attention!"
            )

        if abnormal_count > 0:
            abnormal_names = ", ".join([it.get("name", "") for it in abnormal_items])
            summary_parts.append(
                f"Abnormal: {abnormal_count} indicator(s) outside reference range: {abnormal_names}."
            )
            if critical_count == 0:
                summary_parts.append("Please consult a pediatrician for further evaluation.")
        else:
            summary_parts.append("All indicators are within normal range.")

        summary = " ".join(summary_parts)

        return {
            "report_type": parsed_data.get("report_type", "unknown"),
            "items": evaluated_items,
            "summary": summary,
            "abnormal_count": abnormal_count + critical_count,
            "total_count": total_count,
        }


# Singleton instance
lab_report_parser = LabReportParser()
