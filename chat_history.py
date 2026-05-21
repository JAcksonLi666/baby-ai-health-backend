"""
Chat History Service - JSON file storage for conversation sessions.
Provides session management and message history for AI chat interactions.
"""
import json
from datetime import datetime
from typing import List, Dict, Optional
from threading import Lock

from daily_records import BaseRecordService
from config import RECORDS_DIR


class ChatHistoryService(BaseRecordService):
    """Chat history service for managing conversation sessions and messages.

    Inherits from BaseRecordService to reuse JSON file storage and basic CRUD.
    Each session contains a list of messages (user/assistant/system).
    Thread-safe operations using file locking to prevent race conditions.
    """

    def __init__(self):
        super().__init__("chat_history.json")
        self.prefix = "chat"
        self._lock = Lock()  # Thread-safe lock for concurrent access

    def create_session(self, title: Optional[str] = None) -> Dict:
        """Create a new chat session.

        Args:
            title: Optional session title. Auto-generated from first message if not provided.

        Returns:
            The created session dictionary.
        """
        with self._lock:
            now = datetime.now().isoformat()
            session_id = self._generate_id("session")
            session = {
                "id": session_id,
                "title": title or "New Chat",
                "messages": [],
                "message_count": 0,
                "created_at": now,
                "updated_at": now,
            }
            records = self._read_all()
            records.append(session)
            self._write_all(records)
            return session

    def add_message(
        self, session_id: str, role: str, content: str
    ) -> Optional[Dict]:
        """Add a message to an existing session.

        Args:
            session_id: The session ID to add the message to.
            role: Message role - must be one of 'user', 'assistant', 'system'.
            content: The message content.

        Returns:
            The updated session dictionary, or None if session not found.
        """
        # Validate role
        valid_roles = {"user", "assistant", "system"}
        if role not in valid_roles:
            raise ValueError(f"Invalid role '{role}'. Must be one of: {valid_roles}")

        with self._lock:
            records = self._read_all()
            for session in records:
                if session["id"] == session_id:
                    now = datetime.now().isoformat()
                    message = {
                        "id": self._generate_id("msg"),
                        "role": role,
                        "content": content,
                        "created_at": now,
                    }
                    session["messages"].append(message)
                    session["message_count"] = len(session["messages"])
                    session["updated_at"] = now

                    # Auto-generate title from first user message
                    if (
                        session["title"] == "New Chat"
                        and role == "user"
                        and len(session["messages"]) == 1
                    ):
                        session["title"] = content[:50] + ("..." if len(content) > 50 else "")

                    self._write_all(records)
                    return session
            return None

    def get_session_history(
        self, session_id: str, limit: int = 20
    ) -> Optional[Dict]:
        """Get the message history for a session.

        Args:
            session_id: The session ID.
            limit: Maximum number of messages to return (most recent first).

        Returns:
            Session dictionary with messages, or None if not found.
        """
        for session in self._read_all():
            if session["id"] == session_id:
                # Return messages in chronological order, limited
                messages = session.get("messages", [])
                if limit and limit > 0:
                    messages = messages[-limit:]
                return {
                    "id": session["id"],
                    "title": session["title"],
                    "messages": messages,
                    "message_count": session["message_count"],
                    "created_at": session["created_at"],
                    "updated_at": session["updated_at"],
                }
        return None

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        """List all chat sessions, sorted by most recently updated.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of session summary dictionaries.
        """
        records = self._read_all()
        # Sort by updated_at descending
        records.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        sessions = []
        for session in records[:limit]:
            sessions.append({
                "id": session["id"],
                "title": session["title"],
                "message_count": session["message_count"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
            })
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            records = self._read_all()
            new_records = [r for r in records if r["id"] != session_id]
            if len(new_records) < len(records):
                self._write_all(new_records)
                return True
            return False

    def get_context_messages(
        self, session_id: str, max_messages: int = 10
    ) -> List[Dict]:
        """Get messages formatted for LLM context.

        Returns the most recent messages in a format suitable for
        passing to an LLM as conversation history.

        Args:
            session_id: The session ID.
            max_messages: Maximum number of messages to include.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        for session in self._read_all():
            if session["id"] == session_id:
                messages = session.get("messages", [])
                if max_messages and max_messages > 0:
                    messages = messages[-max_messages:]
                return [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages
                ]
        return []

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a session by ID (full data including all messages).

        Args:
            session_id: The session ID.

        Returns:
            Full session dictionary or None if not found.
        """
        for session in self._read_all():
            if session["id"] == session_id:
                return session
        return None


# Module-level instance
chat_history_service = ChatHistoryService()
