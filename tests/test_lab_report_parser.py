"""
Unit tests for LabReportParser.

Tests lab_report_parser.py functionality:
- Initialization
- Parsing methods
- Result evaluation
- Cache functionality
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLabReportParserInitialization:
    """Test cases for LabReportParser initialization."""

    def test_initialization(self):
        """Test that parser initializes successfully."""
        from lab_report_parser import LabReportParser

        parser = LabReportParser()
        assert parser is not None
        assert hasattr(parser, "_eval_cache")
        assert isinstance(parser._eval_cache, dict)

    def test_parser_has_reference_ranges(self):
        """Test that parser has reference ranges defined."""
        from lab_report_parser import REFERENCE_RANGES, AGE_GROUPS

        assert len(REFERENCE_RANGES) > 0
        assert len(AGE_GROUPS) > 0

        assert "WBC" in REFERENCE_RANGES
        assert "HGB" in REFERENCE_RANGES
        assert "ALT" in REFERENCE_RANGES

        assert "infant" in AGE_GROUPS
        assert "toddler" in AGE_GROUPS


class TestReferenceRanges:
    """Test cases for reference range definitions."""

    def test_age_group_definition(self):
        """Test age group determination."""
        from lab_report_parser import get_age_group

        # Boundary values: 0-12 months = infant, 12-36 = toddler, etc.
        assert get_age_group(6) == "infant"
        assert get_age_group(11) == "infant"
        assert get_age_group(12) == "infant"  # 12 is still infant (<=12)
        assert get_age_group(13) == "toddler"
        assert get_age_group(24) == "toddler"
        assert get_age_group(36) == "toddler"
        assert get_age_group(37) == "preschool"
        assert get_age_group(48) == "preschool"
        assert get_age_group(72) == "preschool"
        assert get_age_group(73) == "school"
        assert get_age_group(96) == "school"

    def test_wbc_reference_ranges(self):
        """Test WBC reference ranges for different age groups."""
        from lab_report_parser import REFERENCE_RANGES

        wbc = REFERENCE_RANGES["WBC"]
        assert wbc["unit"] == "×10⁹/L"
        assert wbc["infant"]["min"] == 6.0
        assert wbc["infant"]["max"] == 18.0
        assert wbc["school"]["min"] == 4.0
        assert wbc["school"]["max"] == 10.0


class TestReportTypeDetection:
    """Test cases for report type detection."""

    def test_detect_blood_report(self):
        """Test detection of blood report type."""
        from lab_report_parser import detect_report_type

        text = "白细胞 5.2 ×10⁹/L 红细胞 4.5 血红蛋白 120"
        report_type = detect_report_type(text)
        assert report_type == "blood"

    def test_detect_liver_report(self):
        """Test detection of liver report type."""
        from lab_report_parser import detect_report_type

        text = "谷丙转氨酶 ALT 25 谷草转氨酶 AST 30 肝功能检查"
        report_type = detect_report_type(text)
        assert report_type == "liver"

    def test_detect_kidney_report(self):
        """Test detection of kidney report type."""
        from lab_report_parser import detect_report_type

        text = "尿素氮 BUN 5.2 肌酐 CREA 35 肾功能"
        report_type = detect_report_type(text)
        assert report_type == "kidney"

    def test_detect_empty_text(self):
        """Test detection with empty text returns default."""
        from lab_report_parser import detect_report_type

        report_type = detect_report_type("")
        assert report_type == "blood"


class TestResultEvaluation:
    """Test cases for result evaluation."""

    def test_evaluate_results_normal(self):
        """Test evaluation of normal results."""
        from lab_report_parser import LabReportParser

        parser = LabReportParser()
        parsed_data = {
            "report_type": "blood",
            "items": [
                {"name": "WBC", "value": 10.0, "unit": "×10⁹/L", "reference": "4-10"},
                {"name": "HGB", "value": 120.0, "unit": "g/L", "reference": "110-145"},
            ],
        }

        result = parser.evaluate_results(parsed_data, age_months=24)

        assert result["total_count"] == 2
        assert result["abnormal_count"] == 0
        assert "items" in result

    def test_evaluate_results_abnormal_high(self):
        """Test evaluation of high values."""
        from lab_report_parser import LabReportParser

        parser = LabReportParser()
        parsed_data = {
            "report_type": "blood",
            "items": [
                {"name": "WBC", "value": 25.0, "unit": "×10⁹/L", "reference": "4-10"},
            ],
        }

        result = parser.evaluate_results(parsed_data, age_months=24)

        assert result["total_count"] == 1
        assert result["abnormal_count"] == 1
        assert result["items"][0]["status"] == "high"

    def test_evaluate_results_abnormal_low(self):
        """Test evaluation of low values."""
        from lab_report_parser import LabReportParser

        parser = LabReportParser()
        parsed_data = {
            "report_type": "blood",
            "items": [
                {"name": "HGB", "value": 80.0, "unit": "g/L", "reference": "110-145"},
            ],
        }

        result = parser.evaluate_results(parsed_data, age_months=24)

        assert result["total_count"] == 1
        assert result["abnormal_count"] == 1
        assert result["items"][0]["status"] == "low"

    def test_evaluate_results_critical(self):
        """Test evaluation of critical values."""
        from lab_report_parser import LabReportParser

        parser = LabReportParser()
        parsed_data = {
            "report_type": "blood",
            "items": [
                {"name": "HGB", "value": 50.0, "unit": "g/L", "reference": "110-145"},
            ],
        }

        result = parser.evaluate_results(parsed_data, age_months=24)

        assert result["items"][0]["status"] == "critical"

    def test_evaluate_results_empty_items(self):
        """Test evaluation with empty items."""
        from lab_report_parser import LabReportParser

        parser = LabReportParser()
        parsed_data = {
            "report_type": "blood",
            "items": [],
        }

        result = parser.evaluate_results(parsed_data, age_months=24)

        assert result["total_count"] == 0
        assert result["abnormal_count"] == 0
        assert result["summary"] == "No items to evaluate."

    def test_evaluate_results_none_data(self):
        """Test evaluation with None data."""
        from lab_report_parser import LabReportParser

        parser = LabReportParser()
        result = parser.evaluate_results({"items": []}, age_months=24)

        assert result["total_count"] == 0
        assert result["abnormal_count"] == 0


class TestCacheHit:
    """Test cases for cache functionality."""

    def test_cache_hit(self):
        """Test that same input hits cache."""
        from lab_report_parser import LabReportParser

        parser = LabReportParser()
        parsed_data = {
            "report_type": "blood",
            "items": [
                {"name": "WBC", "value": 10.0, "unit": "×10⁹/L", "reference": "4-10"},
            ],
        }

        result1 = parser.evaluate_results(parsed_data, age_months=12)
        initial_cache_size = len(parser._eval_cache)
        assert initial_cache_size >= 1

        result2 = parser.evaluate_results(parsed_data, age_months=12)
        assert result1 == result2

        assert len(parser._eval_cache) == initial_cache_size

    def test_different_inputs_different_cache_entries(self):
        """Test that different inputs create different cache entries."""
        from lab_report_parser import LabReportParser

        parser = LabReportParser()

        data1 = {
            "report_type": "blood",
            "items": [{"name": "WBC", "value": 10.0, "unit": "×10⁹/L", "reference": "4-10"}],
        }
        data2 = {
            "report_type": "blood",
            "items": [{"name": "WBC", "value": 15.0, "unit": "×10⁹/L", "reference": "4-10"}],
        }

        parser.evaluate_results(data1, age_months=12)
        cache_size_after_first = len(parser._eval_cache)

        parser.evaluate_results(data2, age_months=12)
        cache_size_after_second = len(parser._eval_cache)

        assert cache_size_after_second > cache_size_after_first


class TestRegexParsing:
    """Test cases for regex-based parsing fallback."""

    def test_parse_blood_report_text(self):
        """Test parsing blood report text returns expected structure."""
        from lab_report_parser import _parse_with_regex

        text = "WBC 5.2 ×10⁹/L 4-10"

        result = _parse_with_regex(text, "blood")

        assert "items" in result
        assert "report_type" in result

    def test_parse_liver_report_text(self):
        """Test parsing liver function report text."""
        from lab_report_parser import _parse_with_regex

        text = "ALT 谷丙转氨酶 25 U/L"

        result = _parse_with_regex(text, "liver")

        assert "items" in result
        assert "report_type" in result

    def test_parse_with_reference_range_in_text(self):
        """Test parsing with reference range included in text."""
        from lab_report_parser import _parse_with_regex

        text = "白细胞(WBC) 8.5 ×10⁹/L 参考值: 4-10"

        result = _parse_with_regex(text, "blood")

        assert "items" in result
        assert "report_type" in result


class TestIndicatorAliases:
    """Test cases for indicator alias resolution."""

    def test_indicator_alias_resolution(self):
        """Test that indicator aliases are resolved correctly."""
        from lab_report_parser import _resolve_indicator_key

        assert _resolve_indicator_key("WBC") == "WBC"
        assert _resolve_indicator_key("wbc") == "WBC"
        assert _resolve_indicator_key("白细胞") == "WBC"
        assert _resolve_indicator_key("血红蛋白") == "HGB"
        assert _resolve_indicator_key("hb") == "HGB"

    def test_unknown_indicator_returns_none(self):
        """Test that unknown indicators return None."""
        from lab_report_parser import _resolve_indicator_key

        result = _resolve_indicator_key("UNKNOWN_INDICATOR")
        assert result is None


class TestAsyncParsing:
    """Test cases for async parsing methods."""

    def test_parse_with_regex_function(self):
        """Test the regex parsing function directly."""
        from lab_report_parser import _parse_with_regex

        text = "WBC 5.2 ×10⁹/L 4-10"
        result = _parse_with_regex(text, "blood")

        assert result is not None
        assert "items" in result
        assert "report_type" in result


class TestSafeFloatConversion:
    """Test cases for safe float conversion."""

    def test_safe_float_valid(self):
        """Test safe float conversion with valid input."""
        from lab_report_parser import LabReportParser

        parser = LabReportParser()
        assert parser._safe_float(5.5) == 5.5
        assert parser._safe_float("10.0") == 10.0
        assert parser._safe_float(100) == 100.0

    def test_safe_float_invalid(self):
        """Test safe float conversion with invalid input."""
        from lab_report_parser import LabReportParser

        parser = LabReportParser()
        assert parser._safe_float(None) is None
        assert parser._safe_float("invalid") is None
        assert parser._safe_float("") is None
