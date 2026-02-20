import sqlite3
import os
import datetime
from typing import List, Dict, Any, Optional

class DatabaseManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to assets/database/mers.db
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            self.db_dir = os.path.join(base_dir, "assets", "database")
            self.db_path = os.path.join(self.db_dir, "mers.db")
        else:
            self.db_path = db_path
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize Tables
        self._create_tables()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Table: Sessions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            dominant_emotion TEXT
        )
        ''')
        
        # Table: Logs (Individual measurements)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            emotion TEXT,
            confidence REAL,
            source TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
        ''')
        
        conn.commit()
        conn.close()

    def create_session(self, session_id: str):
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.datetime.now().isoformat()
        cursor.execute("INSERT OR IGNORE INTO sessions (id, start_time) VALUES (?, ?)", (session_id, now))
        conn.commit()
        conn.close()

    def end_session(self, session_id: str, dominant_emotion: str = None):
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.datetime.now().isoformat()
        cursor.execute("UPDATE sessions SET end_time = ?, dominant_emotion = ? WHERE id = ?", (now, dominant_emotion, session_id))
        conn.commit()
        conn.close()

    def log_emotion(self, session_id: str, emotion: str, confidence: float, source: str = "fusion"):
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.datetime.now().isoformat()
        cursor.execute("INSERT INTO logs (session_id, timestamp, emotion, confidence, source) VALUES (?, ?, ?, ?, ?)",
                       (session_id, now, emotion, confidence, source))
        conn.commit()
        conn.close()

    def get_session_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns a list of sessions with their stats."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, start_time, end_time, dominant_emotion FROM sessions ORDER BY start_time DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
        history = []
        for r in rows:
            history.append({
                "id": r[0],
                "start_time": r[1],
                "end_time": r[2],
                "dominant_emotion": r[3]
            })
            
        conn.close()
        return history

    def get_session_history_all(self):
        """Returns all sessions."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM sessions ORDER BY start_time DESC")
        rows = cursor.fetchall()
        
        data = [dict(row) for row in rows]
        conn.close()
        return data

    def get_all_logs(self) -> List[Dict[str, Any]]:
        """Returns all logs for visualization."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM logs ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        
        data = [dict(row) for row in rows]
        conn.close()
        return data
