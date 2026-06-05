import sqlite3
from rag.logger import logger

DB_PATH = "chat.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ✅ INIT DB
def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # ✅ chats table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ✅ NEW: sessions table (for titles)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ✅ SAVE MESSAGE + CREATE TITLE
def save_message(session_id, role, content):
    conn = get_connection()
    cursor = conn.cursor()

    # ✅ If first user message → create title
    if role == "user":
        cursor.execute(
            "SELECT COUNT(*) FROM chats WHERE session_id = ?",
            (session_id,)
        )
        count = cursor.fetchone()[0]

        if count == 0:
            title = content[:40]  # first message = title

            cursor.execute("""
                INSERT OR IGNORE INTO sessions (session_id, title)
                VALUES (?, ?)
            """, (session_id, title))

    # ✅ save message
    cursor.execute("""
        INSERT INTO chats (session_id, role, content)
        VALUES (?, ?, ?)
    """, (session_id, role, content))

    conn.commit()
    conn.close()
    logger.info("Saved message for session_id=%s role=%s content_len=%s", session_id, role, len(content))


# ✅ LOAD CHAT
def load_messages(session_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT role, content
        FROM chats
        WHERE session_id = ?
        ORDER BY created_at
    """, (session_id,))

    rows = cursor.fetchall()
    conn.close()
    logger.info("Loaded %s messages for session_id=%s", len(rows), session_id)

    return [{"role": r[0], "content": r[1]} for r in rows]


# ✅ GET CHAT THREADS (FAST + CLEAN)
def get_chat_titles(user_id=None):
    conn = get_connection()
    cursor = conn.cursor()

    if user_id:
        cursor.execute("""
            SELECT session_id, title
            FROM sessions
            WHERE session_id LIKE ?
            ORDER BY created_at DESC
        """, (f"{user_id}_%",))
    else:
        cursor.execute("""
            SELECT session_id, title
            FROM sessions
            ORDER BY created_at DESC
        """)

    rows = cursor.fetchall()
    conn.close()
    logger.info("Loaded %s session titles for user_id=%s", len(rows), user_id)

    return rows


# ✅ (optional) delete chat
def delete_chat(session_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM chats WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    conn.commit()
    conn.close()
