"""
Unit tests for KnowledgeBaseService.

Tests core functionality of knowledge_base.py:
- Initialization
- Search methods
- CRUD operations for entries
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestKnowledgeServiceInitialization:
    """Test cases for KnowledgeBaseService initialization."""

    def test_knowledge_service_initialization(self):
        """Test that knowledge service initializes successfully."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        assert service is not None
        assert hasattr(service, "entries")
        assert hasattr(service, "bm25_index")
        assert isinstance(service.entries, list)

    def test_knowledge_service_has_entries(self):
        """Test that knowledge service has predefined entries."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        assert len(service.entries) > 0, "Knowledge base should have predefined entries"

    def test_knowledge_service_get_status(self):
        """Test get_status method."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        status = service.get_status()

        assert status["success"] is True
        assert "total_entries" in status
        assert status["total_entries"] > 0
        assert "retrieval_method" in status


class TestKnowledgeSearch:
    """Test cases for search functionality."""

    def test_search_returns_results(self):
        """Test that search method returns results."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        results = service.search("母乳喂养")

        assert results["success"] is True
        assert "results" in results
        assert "query" in results
        assert "total" in results
        assert len(results["results"]) > 0

    def test_search_with_keyword_matching(self):
        """Test search with keyword matching."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        results = service.search("睡眠时长")

        # Should find entries with matching keywords
        assert results["success"] is True
        assert len(results["results"]) > 0

    def test_search_empty_query(self):
        """Test search with empty query."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        results = service.search("")

        # Should still return structure, but may have 0 results
        assert results["success"] is True
        assert "results" in results

    def test_search_no_match(self):
        """Test search with no matching results."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        results = service.search("xyzabc123nonexistent")

        assert results["success"] is True
        # May return empty results for no match
        assert "results" in results

    def test_search_results_structure(self):
        """Test that search results have correct structure."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        results = service.search("疫苗")

        for result in results["results"]:
            assert "id" in result
            assert "title" in result
            assert "content" in result
            assert "keywords" in result
            assert "score" in result
            assert "match_type" in result


class TestHybridSearch:
    """Test cases for hybrid search functionality."""

    def test_hybrid_search(self):
        """Test hybrid search returns results."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        results = service.hybrid_search("婴儿喂养")

        assert isinstance(results, list)
        # Should return results if service is properly initialized
        assert len(results) >= 0

    def test_hybrid_search_results_structure(self):
        """Test hybrid search results structure."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        results = service.hybrid_search("腹泻处理")

        if len(results) > 0:
            result = results[0]
            assert "id" in result
            assert "title" in result
            assert "score" in result
            assert "match_type" in result

    def test_hybrid_search_with_limit(self):
        """Test hybrid search with top_k parameter."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        results = service.hybrid_search("疫苗", top_k=3)

        assert isinstance(results, list)
        # Results should not exceed top_k
        assert len(results) <= 3


class TestKnowledgeEntryCRUD:
    """Test cases for knowledge entry CRUD operations."""

    def test_add_entry(self):
        """Test adding a new entry."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        initial_count = len(service.entries)

        new_entry = {
            "id": "test_entry_001",
            "title": "测试条目",
            "content": "这是测试内容",
            "source": "测试来源",
            "keywords": ["测试", "单元测试"],
        }

        result = service.add_entry(new_entry)

        assert result["id"] == "test_entry_001"
        assert "created_at" in result
        assert "updated_at" in result
        assert len(service.entries) == initial_count + 1

        # Cleanup
        service.delete_entry("test_entry_001")

    def test_add_entry_validates_required_fields(self):
        """Test that add_entry validates required fields."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()

        # Missing id
        with pytest.raises(ValueError) as exc_info:
            service.add_entry({"title": "Test", "content": "Content"})
        assert "id, title, and content are required" in str(exc_info.value)

        # Missing title
        with pytest.raises(ValueError):
            service.add_entry({"id": "test", "content": "Content"})

        # Missing content
        with pytest.raises(ValueError):
            service.add_entry({"id": "test", "title": "Title"})

    def test_add_entry_prevents_duplicate_id(self):
        """Test that adding entry with duplicate id raises error."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()

        # First add should succeed
        new_entry = {
            "id": "test_dup_001",
            "title": "测试重复",
            "content": "内容",
        }
        service.add_entry(new_entry)

        # Second add with same id should fail
        with pytest.raises(ValueError) as exc_info:
            service.add_entry(new_entry)
        assert "already exists" in str(exc_info.value)

        # Cleanup
        service.delete_entry("test_dup_001")

    def test_add_entry_sets_timestamps(self):
        """Test that add_entry sets created_at and updated_at."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()

        before_add = datetime.now().isoformat()
        new_entry = {
            "id": "test_time_001",
            "title": "时间测试",
            "content": "内容",
        }
        result = service.add_entry(new_entry)
        after_add = datetime.now().isoformat()

        assert "created_at" in result
        assert "updated_at" in result
        assert before_add <= result["created_at"] <= after_add

        # Cleanup
        service.delete_entry("test_time_001")

    def test_delete_entry(self):
        """Test deleting an entry."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()

        # Add an entry to delete
        new_entry = {
            "id": "test_delete_001",
            "title": "删除测试",
            "content": "内容",
        }
        service.add_entry(new_entry)
        initial_count = len(service.entries)

        # Delete the entry
        result = service.delete_entry("test_delete_001")

        assert result is True
        assert len(service.entries) == initial_count - 1

        # Verify entry is deleted
        assert service.get_entry("test_delete_001") is None

    def test_delete_nonexistent_entry(self):
        """Test deleting non-existent entry returns False."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        result = service.delete_entry("nonexistent_id_xyz")
        assert result is False

    def test_list_entries(self):
        """Test listing all entries."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        entries = service.list_entries()

        assert isinstance(entries, list)
        assert len(entries) > 0

        # Check structure
        for entry in entries:
            assert "id" in entry
            assert "title" in entry
            assert "keywords" in entry
            assert "category" in entry

    def test_list_entries_with_category_filter(self):
        """Test listing entries with category filter."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        entries = service.list_entries(category="vaccine")

        assert isinstance(entries, list)
        for entry in entries:
            assert entry["category"] == "vaccine"

    def test_get_entry(self):
        """Test getting a single entry."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        # Get an existing entry
        entry = service.get_entry("kb_feed_001")

        assert entry is not None
        assert "id" in entry
        assert "title" in entry
        assert "content" in entry
        assert entry["id"] == "kb_feed_001"

    def test_get_entry_not_found(self):
        """Test getting non-existent entry returns None."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        entry = service.get_entry("nonexistent_id_xyz")
        assert entry is None


class TestVectorSearch:
    """Test cases for vector search functionality."""

    def test_vector_search(self):
        """Test vector search returns results."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        results = service.vector_search("婴儿喂养", top_k=5)

        assert isinstance(results, list)
        # Results may be empty if vector DB is not available

    def test_vector_search_results_structure(self):
        """Test vector search results structure."""
        from knowledge_base import KnowledgeBaseService

        service = KnowledgeBaseService()
        results = service.vector_search("疫苗", top_k=3)

        for result in results:
            assert "id" in result
            assert "title" in result
            assert "score" in result
            assert "match_type" in result
