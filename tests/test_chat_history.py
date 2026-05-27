"""
Unit tests for ChatHistoryService.

Tests chat_history.py functionality:
- Session management
- Message operations
- Concurrent access
"""
import pytest
import sys
import os
import tempfile
import json
import threading
import time
from unittest.mock import patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a temporary directory for test data
TEST_DATA_DIR = tempfile.mkdtemp()
TEST_CHAT_FILE = os.path.join(TEST_DATA_DIR, "test_chat_history.json")


@pytest.fixture
def chat_service():
    """Fixture that provides a ChatHistoryService with isolated test data."""
    from daily_records import BaseRecordService
    from chat_history import ChatHistoryService

    # Create service instance with mocked config
    service = ChatHistoryService.__new__(ChatHistoryService)
    service._file_path = Path(TEST_CHAT_FILE)
    service.file_path = Path(TEST_CHAT_FILE)  # BaseRecordService uses this
    service.prefix = "chat"
    service._lock = threading.Lock()
    
    # Ensure test file exists
    service.file_path.write_text("[]", encoding="utf-8")
    
    return service


@pytest.fixture
def chat_service_with_session(chat_service):
    """Fixture that provides a ChatHistoryService with an existing session."""
    session = chat_service.create_session()
    chat_service._session_id = session["id"]
    return chat_service


class TestChatSessionCreation:
    """Test cases for chat session creation."""

    def test_create_session(self, chat_service):
        """Test creating a new chat session."""
        session = chat_service.create_session()

        assert session is not None
        assert "id" in session
        assert session["title"] == "New Chat"
        assert session["messages"] == []
        assert session["message_count"] == 0
        assert "created_at" in session
        assert "updated_at" in session
        assert session["id"].startswith("session_")

    def test_create_session_with_title(self, chat_service):
        """Test creating session with custom title."""
        session = chat_service.create_session(title="My Chat Session")

        assert session["title"] == "My Chat Session"

    def test_create_multiple_sessions(self, chat_service):
        """Test creating multiple sessions."""
        session1 = chat_service.create_session()
        session2 = chat_service.create_session()

        assert session1["id"] != session2["id"]


class TestChatMessageOperations:
    """Test cases for chat message operations."""

    def test_add_message(self, chat_service):
        """Test adding a message to session."""
        session = chat_service.create_session()

        result = chat_service.add_message(
            session_id=session["id"],
            role="user",
            content="Hello, how are you?",
        )

        assert result is not None
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello, how are you?"
        assert "id" in result["messages"][0]
        assert "created_at" in result["messages"][0]

    def test_add_message_invalid_role(self, chat_service):
        """Test adding message with invalid role raises error."""
        session = chat_service.create_session()

        with pytest.raises(ValueError) as exc_info:
            chat_service.add_message(
                session_id=session["id"],
                role="invalid_role",
                content="Test",
            )
        assert "Invalid role" in str(exc_info.value)

    def test_add_message_nonexistent_session(self, chat_service):
        """Test adding message to non-existent session."""
        result = chat_service.add_message(
            session_id="nonexistent_id",
            role="user",
            content="Test",
        )
        assert result is None

    def test_add_message_auto_title_from_first_user_message(self, chat_service):
        """Test that session title is auto-generated from first user message."""
        session = chat_service.create_session()
        assert session["title"] == "New Chat"

        chat_service.add_message(
            session_id=session["id"],
            role="user",
            content="This is my first question about baby care",
        )

        updated_session = chat_service.get_session(session["id"])
        assert updated_session["title"] == "This is my first question about baby care"


class TestSessionRetrieval:
    """Test cases for session retrieval operations."""

    def test_get_session_messages(self, chat_service):
        """Test retrieving messages from a session."""
        session = chat_service.create_session()

        chat_service.add_message(session["id"], "user", "Hello")
        chat_service.add_message(session["id"], "assistant", "Hi there!")
        chat_service.add_message(session["id"], "user", "How are you?")

        result = chat_service.get_session_history(session["id"])

        assert result is not None
        assert result["id"] == session["id"]
        assert len(result["messages"]) == 3
        assert result["message_count"] == 3

    def test_get_session_messages_with_limit(self, chat_service):
        """Test retrieving messages with limit."""
        session = chat_service.create_session()

        for i in range(5):
            chat_service.add_message(session["id"], "user", f"Message {i}")

        result = chat_service.get_session_history(session["id"], limit=2)

        assert len(result["messages"]) == 2
        assert result["messages"][0]["content"] == "Message 3"
        assert result["messages"][1]["content"] == "Message 4"

    def test_get_session_messages_nonexistent(self, chat_service):
        """Test retrieving messages from non-existent session."""
        result = chat_service.get_session_history("nonexistent_id")
        assert result is None

    def test_get_full_session(self, chat_service):
        """Test getting full session data."""
        session = chat_service.create_session()
        chat_service.add_message(session["id"], "user", "Test")

        full_session = chat_service.get_session(session["id"])

        assert full_session is not None
        assert full_session["id"] == session["id"]
        assert "messages" in full_session

    def test_list_sessions(self, chat_service):
        """Test listing all sessions."""
        for i in range(3):
            chat_service.create_session(title=f"Session {i}")

        sessions = chat_service.list_sessions()

        assert len(sessions) >= 3
        for session in sessions:
            assert "id" in session
            assert "title" in session
            assert "message_count" in session


class TestSessionDeletion:
    """Test cases for session deletion."""

    def test_delete_session(self, chat_service):
        """Test deleting a session."""
        session = chat_service.create_session()
        session_id = session["id"]

        assert chat_service.get_session(session_id) is not None

        result = chat_service.delete_session(session_id)
        assert result is True

        assert chat_service.get_session(session_id) is None

    def test_delete_nonexistent_session(self, chat_service):
        """Test deleting non-existent session returns False."""
        result = chat_service.delete_session("nonexistent_id")
        assert result is False


class TestContextMessages:
    """Test cases for context message formatting."""

    def test_get_context_messages(self, chat_service):
        """Test getting messages formatted for LLM context."""
        session = chat_service.create_session()
        chat_service.add_message(session["id"], "user", "Hello")
        chat_service.add_message(session["id"], "assistant", "Hi!")
        chat_service.add_message(session["id"], "user", "How are you?")

        context = chat_service.get_context_messages(session["id"])

        assert len(context) == 3
        for msg in context:
            assert "role" in msg
            assert "content" in msg

    def test_get_context_messages_with_limit(self, chat_service):
        """Test getting limited context messages."""
        session = chat_service.create_session()
        for i in range(5):
            chat_service.add_message(session["id"], "user", f"Msg {i}")

        context = chat_service.get_context_messages(session["id"], max_messages=2)

        assert len(context) == 2

    def test_get_context_messages_nonexistent_session(self, chat_service):
        """Test getting context for non-existent session."""
        context = chat_service.get_context_messages("nonexistent_id")
        assert context == []


class TestConcurrentAccess:
    """Test cases for concurrent access to chat service."""

    def test_concurrent_access(self, chat_service):
        """Test thread-safe concurrent access to chat service."""
        errors = []
        results = []

        def create_and_add_messages(thread_id):
            try:
                session = chat_service.create_session(title=f"Thread {thread_id}")
                session_id = session["id"]

                for i in range(5):
                    chat_service.add_message(
                        session_id,
                        "user",
                        f"Thread {thread_id} message {i}",
                    )
                    time.sleep(0.001)

                messages = chat_service.get_session_history(session_id)
                results.append(messages)

            except Exception as e:
                errors.append(e)

        threads = []
        num_threads = 5
        for i in range(num_threads):
            t = threading.Thread(target=create_and_add_messages, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads

    def test_concurrent_read_write(self, chat_service):
        """Test concurrent read and write operations."""
        session = chat_service.create_session()

        errors = []

        def write_messages(thread_id):
            try:
                for i in range(3):
                    chat_service.add_message(
                        session["id"],
                        "user",
                        f"Thread {thread_id} msg {i}",
                    )
            except Exception as e:
                errors.append(e)

        def read_messages():
            try:
                for _ in range(3):
                    chat_service.get_session_history(session["id"])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=write_messages, args=(i,)))
        threads.append(threading.Thread(target=read_messages))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"


# Cleanup after all tests
def teardown_module(module):
    """Clean up test files after all tests complete."""
    import shutil
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
