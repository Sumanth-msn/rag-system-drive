"""
src/chat_history.py
───────────────────
Persists chat sessions to JSON files (reused from existing DocMind codebase).
Adapted to include Drive sync metadata per session.
"""

import json
import os
from datetime import datetime
from typing import List, Optional


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np

            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if hasattr(obj, "page_content"):
            return {
                "page_content": obj.page_content[:300],
                "source_file": obj.metadata.get("file_name", "unknown"),
                "page": obj.metadata.get("page", 0),
            }
        try:
            return str(obj)
        except Exception:
            return "[non-serializable]"


SESSIONS_DIR = "chat_sessions"


def ensure_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def generate_session_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def create_new_session(source: str = "gdrive") -> dict:
    return {
        "session_id": generate_session_id(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "messages": [],
    }


def save_session(session: dict):
    ensure_sessions_dir()
    path = get_session_path(session["session_id"])
    with open(path, "w") as f:
        json.dump(session, f, indent=2, cls=SafeJSONEncoder)


def load_session(session_id: str) -> Optional[dict]:
    path = get_session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def add_message_to_session(session: dict, message: dict) -> dict:
    message["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session["messages"].append(message)
    save_session(session)
    return session


def get_all_sessions() -> List[dict]:
    ensure_sessions_dir()
    sessions = []
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            path = os.path.join(SESSIONS_DIR, filename)
            try:
                with open(path, "r") as f:
                    sessions.append(json.load(f))
            except Exception:
                pass
    sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return sessions


def delete_session(session_id: str):
    path = get_session_path(session_id)
    if os.path.exists(path):
        os.remove(path)


def format_session_label(session: dict) -> str:
    created = session.get("created_at", "")
    try:
        dt = datetime.strptime(created, "%Y-%m-%d %H:%M:%S")
        today = datetime.now().date()
        time_label = (
            dt.strftime("%H:%M") if dt.date() == today else dt.strftime("%b %d")
        )
    except Exception:
        time_label = ""
    msg_count = len(session.get("messages", []))
    return f"Drive · {time_label} · {msg_count} msgs"


def group_sessions_by_date(sessions: List[dict]) -> dict:
    today = datetime.now().date()
    groups = {"Today": [], "Yesterday": [], "Older": []}
    for session in sessions:
        created = session.get("created_at", "")
        try:
            dt = datetime.strptime(created, "%Y-%m-%d %H:%M:%S")
            diff = (today - dt.date()).days
            if diff == 0:
                groups["Today"].append(session)
            elif diff == 1:
                groups["Yesterday"].append(session)
            else:
                groups["Older"].append(session)
        except Exception:
            groups["Older"].append(session)
    return groups
