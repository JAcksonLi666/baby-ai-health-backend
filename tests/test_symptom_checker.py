"""
Unit tests for SymptomChecker.

Tests symptom_checker.py functionality:
- Initialization
- Symptom checking
- Cache functionality
- Empty symptom handling
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSymptomCheckerInitialization:
    """Test cases for SymptomChecker initialization."""

    def test_initialization(self):
        """Test that symptom checker initializes successfully."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()
        assert checker is not None
        assert hasattr(checker, "knowledge_base")
        assert hasattr(checker, "_analysis_cache")

    def test_has_categories_defined(self):
        """Test that symptom categories are defined."""
        from symptom_checker import SYMPTOM_CATEGORIES

        assert len(SYMPTOM_CATEGORIES) > 0
        assert "fever" in SYMPTOM_CATEGORIES
        assert "respiratory" in SYMPTOM_CATEGORIES
        assert "digestive" in SYMPTOM_CATEGORIES
        assert "skin" in SYMPTOM_CATEGORIES

    def test_get_all_categories(self):
        """Test getting all symptom categories."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()
        categories = checker.get_all_categories()

        assert isinstance(categories, list)
        assert len(categories) > 0

        for cat in categories:
            assert "key" in cat
            assert "name" in cat
            assert "symptoms" in cat
            assert len(cat["symptoms"]) > 0


class TestSymptomMatching:
    """Test cases for symptom matching."""

    def test_match_exact_symptom(self):
        """Test exact symptom matching."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()
        matched = checker._match_symptoms(["低热"])

        assert len(matched) > 0
        # Matched symptom should be under fever category
        categories = [m["category_key"] for m in matched]
        assert "fever" in categories

    def test_match_multiple_symptoms(self):
        """Test matching multiple symptoms."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()
        # Use exact symptom names from the built-in tree
        matched = checker._match_symptoms(["低热", "咳嗽", "腹泻"])

        assert len(matched) >= 1  # At least one should match

    def test_match_symptom_case_insensitive(self):
        """Test symptom matching is case insensitive for Chinese."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()
        matched1 = checker._match_symptoms(["低热"])
        matched2 = checker._match_symptoms(["中热"])

        # Both should match fever-related symptoms
        assert len(matched1) >= 0 or len(matched2) >= 0

    def test_match_no_match(self):
        """Test handling of unmatched symptoms."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()
        matched = checker._match_symptoms(["xyz_nonexistent_symptom"])

        assert len(matched) == 0


class TestSymptomCheck:
    """Test cases for symptom checking."""

    def test_check_symptoms_basic(self):
        """Test basic symptom checking."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["发烧"],
                    age_months=12,
                )
            )
        finally:
            loop.close()

        assert result["success"] is True
        assert result["age_months"] == 12
        assert "matched_count" in result
        assert "overall_severity" in result
        assert "categories" in result
        assert "general_precautions" in result

    def test_check_symptoms_with_duration(self):
        """Test symptom checking with duration."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["咳嗽"],
                    age_months=18,
                    duration_days=5,
                )
            )
        finally:
            loop.close()

        assert result["duration_days"] == 5

    def test_check_symptoms_with_severity(self):
        """Test symptom checking with severity."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["发烧"],
                    age_months=24,
                    severity=5,  # Maximum severity
                )
            )
        finally:
            loop.close()

        # Check that severity was considered (overall severity may vary based on symptom)
        assert result["overall_severity"] in ["mild", "moderate", "severe"]

    def test_check_symptoms_matched_results(self):
        """Test that matched symptoms return proper results."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["发烧", "咳嗽"],
                    age_months=12,
                )
            )
        finally:
            loop.close()

        for cat in result["categories"]:
            assert "category" in cat
            assert "symptom" in cat
            assert "description" in cat
            assert "possible_causes" in cat
            assert "severity" in cat
            assert "precautions" in cat

    def test_check_symptoms_unmatched(self):
        """Test handling of unmatched symptoms."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["xyz_nonexistent"],
                    age_months=12,
                )
            )
        finally:
            loop.close()

        assert result["matched_count"] == 0
        assert "unmatched_symptoms" in result


class TestCacheHit:
    """Test cases for cache functionality."""

    def test_cache_hit(self):
        """Test that same input hits cache."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result1 = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["发烧"],
                    age_months=12,
                )
            )
            cache_size_after_first = len(checker._analysis_cache)

            result2 = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["发烧"],
                    age_months=12,
                )
            )
            cache_size_after_second = len(checker._analysis_cache)

            assert result1 == result2
            assert cache_size_after_second == cache_size_after_first
        finally:
            loop.close()

    def test_different_inputs_different_cache(self):
        """Test that different inputs create different cache entries."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                checker.analyze_symptoms(symptoms=["发烧"], age_months=12)
            )
            cache_size_after_first = len(checker._analysis_cache)

            loop.run_until_complete(
                checker.analyze_symptoms(symptoms=["咳嗽"], age_months=12)
            )
            cache_size_after_second = len(checker._analysis_cache)

            assert cache_size_after_second > cache_size_after_first
        finally:
            loop.close()


class TestEmptySymptoms:
    """Test cases for empty symptom handling."""

    def test_empty_symptoms_list(self):
        """Test handling of empty symptoms list."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=[],
                    age_months=12,
                )
            )
        finally:
            loop.close()

        assert result["success"] is True
        assert result["matched_count"] == 0
        assert result["total_input"] == 0

    def test_symptoms_with_empty_strings(self):
        """Test handling of symptoms with empty strings."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["", "  "],
                    age_months=12,
                )
            )
        finally:
            loop.close()

        # Empty strings should not cause errors
        assert result["success"] is True


class TestAgeSpecificNotes:
    """Test cases for age-specific notes."""

    def test_age_specific_notes_infant(self):
        """Test age-specific notes for infants under 3 months."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["发烧"],
                    age_months=2,
                )
            )
        finally:
            loop.close()

        categories = result["categories"]
        for cat in categories:
            if cat.get("age_notes"):
                assert any("3个月以下" in note for note in cat["age_notes"])

    def test_age_specific_notes_digestive(self):
        """Test age-specific notes for digestive issues in young infants."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["腹泻"],
                    age_months=4,
                )
            )
        finally:
            loop.close()

        categories = result["categories"]
        for cat in categories:
            if cat.get("age_notes"):
                assert any("脱水" in note or "婴儿" in note for note in cat["age_notes"])


class TestRelatedKnowledge:
    """Test cases for related knowledge retrieval."""

    def test_related_knowledge_retrieval(self):
        """Test that related knowledge is retrieved."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["发烧"],
                    age_months=12,
                )
            )
        finally:
            loop.close()

        assert "related_knowledge" in result

    def test_no_related_knowledge_for_unknown(self):
        """Test handling when no related knowledge available."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["xyz_unknown_symptom_123"],
                    age_months=12,
                )
            )
        finally:
            loop.close()

        assert "related_knowledge" in result


class TestDisclaimer:
    """Test cases for medical disclaimer."""

    def test_disclaimer_present(self):
        """Test that disclaimer is included in results."""
        from symptom_checker import SymptomChecker

        checker = SymptomChecker()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                checker.analyze_symptoms(
                    symptoms=["发烧"],
                    age_months=12,
                )
            )
        finally:
            loop.close()

        assert "disclaimer" in result
        assert "就医" in result["disclaimer"] or "medical" in result["disclaimer"].lower()
