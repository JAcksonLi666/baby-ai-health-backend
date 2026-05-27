"""
基于 SQLite 的基础记录服务
替换 daily_records.py 中的 BaseRecordService（JSON 文件存储）
"""
import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from database import get_db

logger = logging.getLogger(__name__)


class BaseRecordServiceDB:
    """基于 SQLite 的基础记录服务"""

    def __init__(self, table_name: str, date_field: str = "created_at"):
        self.table_name = table_name
        self.date_field = date_field
        self.db = get_db()

    def _generate_id(self, prefix: str) -> str:
        """生成唯一 ID"""
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        import uuid
        short_uuid = uuid.uuid4().hex[:6]
        return f"{prefix}_{now}_{short_uuid}"

    def create(self, prefix: str, data: dict) -> dict:
        """创建记录"""
        now = datetime.now().isoformat()
        record_id = self._generate_id(prefix)

        data['id'] = record_id
        data['created_at'] = now
        data['updated_at'] = now

        columns = list(data.keys())
        values = [data[c] for c in columns]
        placeholders = ', '.join(['?'] * len(columns))
        col_names = ', '.join(columns)

        self.db.execute(
            f"INSERT INTO {self.table_name} ({col_names}) VALUES ({placeholders})",
            values
        )
        self.db.commit()

        return data

    def get_by_id(self, record_id: str) -> Optional[dict]:
        """根据 ID 获取记录"""
        row = self.db.execute(
            f"SELECT * FROM {self.table_name} WHERE id = ?", (record_id,)
        ).fetchone()

        if row:
            return dict(row)
        return None

    def list_records(self, date_from: str = None, date_to: str = None,
                     status: str = None, limit: int = 100, offset: int = 0,
                     extra_where: str = None, extra_params: list = None) -> dict:
        """列表查询"""
        conditions = []
        params = []

        if date_from:
            conditions.append(f"{self.date_field} >= ?")
            params.append(date_from)

        if date_to:
            conditions.append(f"{self.date_field} <= ?")
            params.append(date_to + "T23:59:59" if len(date_to) == 10 else date_to)

        if status:
            conditions.append("status = ?")
            params.append(status)

        if extra_where and extra_params:
            conditions.append(extra_where)
            params.extend(extra_params)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # 查询总数
        count_sql = f"SELECT COUNT(*) FROM {self.table_name} {where_clause}"
        total = self.db.execute(count_sql, params).fetchone()[0]

        # 查询数据
        data_sql = f"SELECT * FROM {self.table_name} {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.db.execute(data_sql, params).fetchall()

        records = [dict(row) for row in rows]

        return {
            "success": True,
            "records": records,
            "total": total,
            "limit": limit,
            "offset": offset
        }

    def update(self, record_id: str, data: dict) -> Optional[dict]:
        """更新记录"""
        data['updated_at'] = datetime.now().isoformat()

        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        values = list(data.values()) + [record_id]

        self.db.execute(
            f"UPDATE {self.table_name} SET {set_clause} WHERE id = ?",
            values
        )
        self.db.commit()

        return self.get_by_id(record_id)

    def delete(self, record_id: str) -> bool:
        """删除记录"""
        cursor = self.db.execute(
            f"DELETE FROM {self.table_name} WHERE id = ?", (record_id,)
        )
        self.db.commit()
        return cursor.rowcount > 0

    def get_all(self) -> List[dict]:
        """获取所有记录"""
        rows = self.db.execute(
            f"SELECT * FROM {self.table_name} ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]
