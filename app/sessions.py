"""
Chat Session & Conversational Memory Store (`app/sessions.py`).
Persists multi-turn conversations, enabling the Agent to recall past context,
resolve conversational pronouns (e.g., "what about its net profit?"),
and powers the LLM Chatbot UI (+ New Chat, Rename, Pin, Delete, Session switching).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger("quant_agent.sessions")

SESSIONS_FILE = settings.corpus_dir / "chat_sessions.json"


class ChatMessageItem(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:10]}")
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: float = Field(default_factory=time.time)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    agent_trace: List[Dict[str, Any]] = Field(default_factory=list)
    abstained: bool = False


class ChatSessionItem(BaseModel):
    id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:8]}")
    title: str = "New Conversation"
    pinned: bool = False
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    messages: List[ChatMessageItem] = Field(default_factory=list)


class SessionManager:
    """Manages chat sessions stored in a JSON file."""

    def __init__(self, storage_path: Path = SESSIONS_FILE) -> None:
        self.storage_path = storage_path
        self._sessions: Dict[str, ChatSessionItem] = {}
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                for item in data:
                    sess = ChatSessionItem.model_validate(item)
                    self._sessions[sess.id] = sess
            except Exception as e:
                logger.warning(f"Could not load chat sessions from {self.storage_path}: {e}")
                self._sessions = {}

    def _save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = [s.model_dump() for s in self._sessions.values()]
            self.storage_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions ordered by pinned (descending), then updated_at (descending)."""
        sorted_sessions = sorted(
            self._sessions.values(),
            key=lambda s: (1 if getattr(s, "pinned", False) else 0, s.updated_at),
            reverse=True,
        )
        return [
            {
                "id": s.id,
                "title": s.title,
                "pinned": getattr(s, "pinned", False),
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "message_count": len(s.messages),
                "last_message": s.messages[-1].content[:60] if s.messages else "",
            }
            for s in sorted_sessions
        ]

    def get_session(self, session_id: str) -> Optional[ChatSessionItem]:
        return self._sessions.get(session_id)

    def create_session(self, title: Optional[str] = None) -> ChatSessionItem:
        sess = ChatSessionItem(title=title or "New Conversation")
        self._sessions[sess.id] = sess
        self._save()
        return sess

    def get_or_create_session(self, session_id: Optional[str] = None) -> ChatSessionItem:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session()

    def rename_session(self, session_id: str, new_title: str) -> Optional[ChatSessionItem]:
        sess = self.get_session(session_id)
        if sess:
            clean = new_title.strip()
            sess.title = clean if clean else "Untitled Chat"
            sess.updated_at = time.time()
            self._save()
            return sess
        return None

    def toggle_pin_session(self, session_id: str, pinned: Optional[bool] = None) -> Optional[ChatSessionItem]:
        sess = self.get_session(session_id)
        if sess:
            sess.pinned = not getattr(sess, "pinned", False) if pinned is None else pinned
            sess.updated_at = time.time()
            self._save()
            return sess
        return None

    def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant"],
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        agent_trace: Optional[List[Dict[str, Any]]] = None,
        abstained: bool = False,
    ) -> ChatMessageItem:
        sess = self.get_or_create_session(session_id)
        msg = ChatMessageItem(
            role=role,
            content=content,
            citations=citations or [],
            retrieved_chunks=retrieved_chunks or [],
            agent_trace=agent_trace or [],
            abstained=abstained,
        )
        sess.messages.append(msg)
        sess.updated_at = time.time()

        # Auto-name title on first user message if still default
        if role == "user" and (sess.title == "New Conversation" or not sess.title):
            clean_title = content.strip().replace("\n", " ")
            if len(clean_title) > 35:
                clean_title = clean_title[:32] + "..."
            sess.title = clean_title

        self._save()
        return msg

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save()
            return True
        return False

    def clear_all(self) -> None:
        self._sessions = {}
        self._save()


# Singleton instance
session_manager = SessionManager()
