"""
SQLite 数据库管理模块
提供数据库连接、表创建、数据迁移功能
"""
import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import Optional, Any, List, Dict

logger = logging.getLogger(__name__)

class Database:
    """SQLite 数据库管理器"""

    def __init__(self, db_path: str = "data/baby_health.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._ensure_dir()
        self.connect()
        self.create_tables()

    def _ensure_dir(self):
        """确保数据库目录存在"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def connect(self):
        """建立数据库连接"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """执行 SQL"""
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_list: List[tuple]) -> sqlite3.Cursor:
        """批量执行 SQL"""
        return self.conn.executemany(sql, params_list)

    def commit(self):
        """提交事务"""
        self.conn.commit()

    def create_tables(self):
        """创建所有数据表"""
        cursor = self.conn.cursor()

        # 睡眠记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sleep_records (
                id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                sleep_type TEXT DEFAULT 'night',
                quality INTEGER,
                duration_minutes INTEGER,
                notes TEXT,
                is_ongoing INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # 排泄记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diaper_records (
                id TEXT PRIMARY KEY,
                time TEXT NOT NULL,
                diaper_type TEXT DEFAULT 'pee',
                poop_color TEXT,
                poop_consistency TEXT,
                amount TEXT DEFAULT 'medium',
                has_photo INTEGER DEFAULT 0,
                notes TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # 哭声记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cry_records (
                id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                reason TEXT,
                intensity INTEGER,
                soothing_method TEXT,
                has_audio INTEGER DEFAULT 0,
                notes TEXT,
                duration_minutes INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # 喂养记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feeding_records (
                id TEXT PRIMARY KEY,
                time TEXT NOT NULL,
                feeding_type TEXT DEFAULT 'breast',
                duration_minutes INTEGER,
                amount_ml REAL,
                breast_side TEXT,
                solid_food TEXT,
                water_amount_ml REAL,
                notes TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # 生长发育记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS growth_records (
                id TEXT PRIMARY KEY,
                record_date TEXT NOT NULL,
                weight_kg REAL,
                height_cm REAL,
                head_circumference_cm REAL,
                temperature_c REAL,
                notes TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # 提醒记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminder_records (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                reminder_type TEXT DEFAULT 'other',
                reminder_date TEXT NOT NULL,
                reminder_time TEXT,
                repeat_type TEXT DEFAULT 'none',
                status TEXT DEFAULT 'pending',
                notes TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # 对话会话表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                message_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # 对话消息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        """)

        # 知识库条目表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                keywords TEXT,
                category TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sleep_start ON sleep_records(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diaper_time ON diaper_records(time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cry_start ON cry_records(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feeding_time ON feeding_records(time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_growth_date ON growth_records(record_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_date ON reminder_records(reminder_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_status ON reminder_records(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_msg_session ON chat_messages(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_entries(category)")

        self.conn.commit()
        logger.info("数据库表创建完成")

    def migrate_from_json(self, json_dir: str = "data/records"):
        """从 JSON 文件迁移数据到数据库"""
        json_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), json_dir)
        if not os.path.exists(json_dir):
            logger.info(f"JSON 目录不存在，跳过迁移: {json_dir}")
            return

        migration_map = {
            "sleep_records.json": "sleep_records",
            "diaper_records.json": "diaper_records",
            "cry_records.json": "cry_records",
            "feeding_records.json": "feeding_records",
            "growth_records.json": "growth_records",
            "reminder_records.json": "reminder_records",
        }

        for json_file, table_name in migration_map.items():
            json_path = os.path.join(json_dir, json_file)
            if not os.path.exists(json_path):
                logger.info(f"跳过不存在的文件: {json_file}")
                continue

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)

                if not records:
                    logger.info(f"空数据文件: {json_file}")
                    continue

                # 检查是否已有数据
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                if count > 0:
                    logger.info(f"表 {table_name} 已有 {count} 条数据，跳过迁移")
                    continue

                # 迁移数据
                for record in records:
                    columns = list(record.keys())
                    values = [record[c] for c in columns]
                    placeholders = ', '.join(['?'] * len(columns))
                    col_names = ', '.join(columns)
                    try:
                        self.conn.execute(
                            f"INSERT OR IGNORE INTO {table_name} ({col_names}) VALUES ({placeholders})",
                            values
                        )
                    except Exception as e:
                        logger.warning(f"迁移记录失败: {e}")

                self.conn.commit()
                logger.info(f"迁移 {json_file}: {len(records)} 条记录 -> {table_name}")
            except Exception as e:
                logger.error(f"迁移 {json_file} 失败: {e}")

        # 迁移对话历史（特殊结构）
        self._migrate_chat_history(json_dir)

        # 迁移知识库
        self._migrate_knowledge_base()

    def _migrate_chat_history(self, json_dir: str):
        """迁移对话历史（嵌套结构拆分为两张表）"""
        json_path = os.path.join(json_dir, "chat_history.json")
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                sessions = json.load(f)

            if not sessions:
                return

            count = self.conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0]
            if count > 0:
                logger.info("chat_sessions 已有数据，跳过迁移")
                return

            for session in sessions:
                messages = session.pop('messages', [])
                self.conn.execute(
                    "INSERT OR IGNORE INTO chat_sessions (id, title, message_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (session.get('id'), session.get('title'), session.get('message_count', len(messages)),
                     session.get('created_at'), session.get('updated_at'))
                )

                for msg in messages:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO chat_messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                        (msg.get('id'), session.get('id'), msg.get('role'), msg.get('content'), msg.get('created_at'))
                    )

            self.conn.commit()
            logger.info(f"迁移 chat_history: {len(sessions)} 个会话")
        except Exception as e:
            logger.error(f"迁移对话历史失败: {e}")

    def _migrate_knowledge_base(self):
        """迁移知识库（从 Python 代码中的硬编码数据）"""
        try:
            from knowledge_base import KNOWLEDGE_ENTRIES
        except ImportError:
            logger.info("无法导入知识库数据，跳过迁移")
            return

        count = self.conn.execute("SELECT COUNT(*) FROM knowledge_entries").fetchone()[0]
        if count > 0:
            logger.info("knowledge_entries 已有数据，跳过迁移")
            return

        for entry in KNOWLEDGE_ENTRIES:
            keywords = json.dumps(entry.get('keywords', []), ensure_ascii=False)
            self.conn.execute(
                "INSERT OR IGNORE INTO knowledge_entries (id, title, content, source, keywords, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (entry.get('id'), entry.get('title'), entry.get('content'),
                 entry.get('source'), keywords, entry.get('category'),
                 entry.get('created_at', datetime.now().isoformat()),
                 entry.get('updated_at', datetime.now().isoformat()))
            )

        self.conn.commit()
        logger.info(f"迁移知识库: {len(KNOWLEDGE_ENTRIES)} 条")


# 全局数据库实例
db: Optional[Database] = None

def get_db() -> Database:
    """获取全局数据库实例"""
    global db
    if db is None:
        db = Database()
    return db

def init_db(db_path: str = "data/baby_health.db", migrate: bool = True):
    """初始化数据库"""
    global db
    db = Database(db_path)
    if migrate:
        db.migrate_from_json()
    return db
