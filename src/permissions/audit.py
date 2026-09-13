"""审计日志 — 记录所有工具调用的完整审计轨迹。

支持：
- 记录每次工具调用的时间、工具名、动作、参数、结果、耗时、风险等级
- 通过 EventBus 自动订阅 TOOL_CALL / TOOL_RESULT 事件
- JSON 格式导出
- 日志文件按日轮转
- 内存日志查询（最近 N 条）
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import EventType, ToolCallEvent, ToolResultEvent

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """单条审计记录。"""

    timestamp: str  # ISO 格式时间字符串
    tool_name: str
    action_name: str
    function_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = ""  # "success" | "error" | "timeout" | "denied"
    output_preview: str = ""  # 输出前200字符
    error: str = ""
    duration_ms: float = 0.0
    risk_level: str = "low"
    session_id: str = ""
    completed: bool = False  # 是否已有结果

    # Phase 6 增强字段
    intent: str = ""  # 识别到的意图
    confidence: float = 0.0  # 意图置信度
    tool_tier: str = ""  # 工具集层级 (recommended/extended/full)
    consecutive_failures: int = 0  # 当时的连续失败次数
    user_input: str = ""  # 原始用户输入

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """审计日志记录器。

    用法::

        audit = AuditLogger(log_dir=Path("~/.weclaw/audit"))
        audit.connect(event_bus)  # 自动订阅事件

        # 或手动记录
        audit.log_call("shell", "run", {"command": "dir"})
        audit.log_result("shell", "run", "success", output="...", duration_ms=150)

        # 查询
        recent = audit.get_recent(10)
        audit.export_json(Path("audit.json"))
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        max_memory_entries: int = 1000,
        write_to_file: bool = True,
    ):
        """
        Args:
            log_dir: 日志文件目录
            max_memory_entries: 内存中保留的最大条目数
            write_to_file: 是否写入文件
        """
        self._log_dir = log_dir or Path.home() / ".weclaw" / "audit"
        self._max_memory = max_memory_entries
        self._write_to_file = write_to_file
        self._entries: deque[AuditEntry] = deque(maxlen=max_memory_entries)
        # 待完成的调用（tool_call 发出但 tool_result 尚未到达）
        self._pending: dict[str, AuditEntry] = {}  # tool_name_action_name_N → entry
        self._call_counter: int = 0  # 递增计数器，用于区分同名工具的并行调用
        # 统计
        self._total_calls = 0
        self._total_errors = 0
        self._total_denied = 0
        # SQLite 持久化
        self._db_path = Path.home() / ".weclaw" / "tool_audit.db"
        self._db_conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """初始化 SQLite 数据库。"""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db_conn = sqlite3.connect(
                str(self._db_path), check_same_thread=False,
            )
            self._db_conn.execute("PRAGMA journal_mode=WAL")
            self._db_conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    function_name TEXT DEFAULT '',
                    arguments TEXT DEFAULT '{}',
                    status TEXT DEFAULT '',
                    output_preview TEXT DEFAULT '',
                    error TEXT DEFAULT '',
                    duration_ms REAL DEFAULT 0,
                    risk_level TEXT DEFAULT 'low',
                    session_id TEXT DEFAULT '',
                    intent TEXT DEFAULT '',
                    confidence REAL DEFAULT 0,
                    tool_tier TEXT DEFAULT '',
                    user_input TEXT DEFAULT ''
                )
            """)
            self._db_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON tool_audit_log(timestamp)"
            )
            self._db_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON tool_audit_log(session_id)"
            )
            self._db_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_audit_log(tool_name)"
            )
            self._db_conn.commit()
        except Exception as e:
            logger.error("初始化审计日志 SQLite 失败: %s", e)
            self._db_conn = None

    # ------------------------------------------------------------------
    # EventBus 集成
    # ------------------------------------------------------------------

    def connect(self, event_bus: EventBus) -> None:
        """连接到事件总线，自动订阅工具调用/结果事件。"""
        event_bus.on(EventType.TOOL_CALL, self._on_tool_call, priority=50)
        event_bus.on(EventType.TOOL_RESULT, self._on_tool_result, priority=50)
        logger.info("审计日志已连接到事件总线")

    def _on_tool_call(self, event_type: str, data: Any) -> None:
        """处理工具调用事件。"""
        if isinstance(data, ToolCallEvent):
            entry = AuditEntry(
                timestamp=datetime.now().isoformat(),
                tool_name=data.tool_name,
                action_name=data.action_name,
                function_name=data.function_name,
                arguments=data.arguments,
                session_id=data.session_id,
            )
        elif isinstance(data, dict):
            entry = AuditEntry(
                timestamp=datetime.now().isoformat(),
                tool_name=data.get("tool_name", ""),
                action_name=data.get("action_name", ""),
                function_name=data.get("function_name", ""),
                arguments=data.get("arguments", {}),
                session_id=data.get("session_id", ""),
            )
        else:
            return

        # key 统一使用 tool_name + action_name + 递增计数器（支持并行调用同名工具）
        self._call_counter += 1
        key = f"{entry.tool_name}_{entry.action_name}_{self._call_counter}"
        self._pending[key] = entry
        self._total_calls += 1

    def _on_tool_result(self, event_type: str, data: Any) -> None:
        """处理工具结果事件。"""
        if isinstance(data, ToolResultEvent):
            tool_name = data.tool_name
            action_name = data.action_name
            status = data.status
            output = data.output
            error = data.error
            duration_ms = data.duration_ms
            session_id = data.session_id
        elif isinstance(data, dict):
            tool_name = data.get("tool_name", "")
            action_name = data.get("action_name", "")
            status = data.get("status", "")
            output = data.get("output", "")
            error = data.get("error", "")
            duration_ms = data.get("duration_ms", 0)
            session_id = data.get("session_id", "")
        else:
            return

        # 从 _pending 中查找匹配的 entry（支持并行调用同名工具）
        # 优先精确匹配（带 counter），然后前缀匹配（取最早的）
        prefix = f"{tool_name}_{action_name}_"
        entry = None
        for k in list(self._pending.keys()):
            if k.startswith(prefix):
                # 取最早加入的（字典保持插入顺序）
                entry = self._pending.pop(k)
                break

        if entry is None:
            # 没有匹配的 call，创建新记录
            entry = AuditEntry(
                timestamp=datetime.now().isoformat(),
                tool_name=tool_name,
                action_name=action_name,
                session_id=session_id,
            )

        entry.status = status or "unknown"
        entry.output_preview = output[:200] if output else ""
        entry.error = error
        entry.duration_ms = duration_ms
        entry.completed = True

        if status == "error":
            self._total_errors += 1
        elif status == "denied":
            self._total_denied += 1

        self._entries.append(entry)

        if self._write_to_file:
            self._write_entry(entry)

        self._persist_entry(entry)

    # ------------------------------------------------------------------
    # 手动记录
    # ------------------------------------------------------------------

    def log_call(
        self,
        tool_name: str,
        action_name: str,
        arguments: dict[str, Any] | None = None,
        risk_level: str = "low",
        session_id: str = "",
    ) -> AuditEntry:
        """手动记录一次工具调用（无结果）。"""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            tool_name=tool_name,
            action_name=action_name,
            arguments=arguments or {},
            risk_level=risk_level,
            session_id=session_id,
        )
        self._call_counter += 1
        self._pending[f"{tool_name}_{action_name}_{self._call_counter}"] = entry
        self._total_calls += 1
        return entry

    def log_result(
        self,
        tool_name: str,
        action_name: str,
        status: str,
        output: str = "",
        error: str = "",
        duration_ms: float = 0.0,
        session_id: str = "",
    ) -> AuditEntry:
        """手动记录一次工具结果。"""
        key = f"{tool_name}_{action_name}_"
        entry = None
        for k in list(self._pending.keys()):
            if k.startswith(key):
                entry = self._pending.pop(k)
                break
        if entry is None:
            entry = AuditEntry(
                timestamp=datetime.now().isoformat(),
                tool_name=tool_name,
                action_name=action_name,
                session_id=session_id,
            )

        entry.status = status
        entry.output_preview = output[:200] if output else ""
        entry.error = error
        entry.duration_ms = duration_ms
        entry.completed = True

        if status == "error":
            self._total_errors += 1
        elif status == "denied":
            self._total_denied += 1

        self._entries.append(entry)

        if self._write_to_file:
            self._write_entry(entry)

        self._persist_entry(entry)
        return entry

    # ------------------------------------------------------------------
    # SQLite 持久化
    # ------------------------------------------------------------------

    def _persist_entry(self, entry: AuditEntry) -> None:
        """将完成的审计记录写入 SQLite。"""
        if self._db_conn is None:
            return
        try:
            args_json = json.dumps(entry.arguments, ensure_ascii=False, default=str)
            self._db_conn.execute(
                """INSERT INTO tool_audit_log
                   (timestamp, tool_name, action_name, function_name, arguments,
                    status, output_preview, error, duration_ms, risk_level,
                    session_id, intent, confidence, tool_tier, user_input)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.timestamp, entry.tool_name, entry.action_name,
                    entry.function_name, args_json, entry.status,
                    entry.output_preview, entry.error, entry.duration_ms,
                    entry.risk_level, entry.session_id, entry.intent,
                    entry.confidence, entry.tool_tier, entry.user_input,
                ),
            )
            self._db_conn.commit()
        except Exception as e:
            logger.error("写入审计记录到 SQLite 失败: %s", e)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_recent(self, count: int = 20) -> list[AuditEntry]:
        """获取最近 N 条审计记录。"""
        entries = list(self._entries)
        return entries[-count:] if len(entries) > count else entries

    def get_by_tool(self, tool_name: str) -> list[AuditEntry]:
        """按工具名查询。"""
        return [e for e in self._entries if e.tool_name == tool_name]

    def get_by_session(self, session_id: str) -> list[AuditEntry]:
        """按会话查询。"""
        return [e for e in self._entries if e.session_id == session_id]

    def get_errors(self) -> list[AuditEntry]:
        """获取所有错误记录。"""
        return [e for e in self._entries if e.status in ("error", "denied")]

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def total_errors(self) -> int:
        return self._total_errors

    @property
    def total_denied(self) -> int:
        return self._total_denied

    def get_stats(self) -> dict[str, Any]:
        """获取审计统计。"""
        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "total_denied": self._total_denied,
            "entries_in_memory": len(self._entries),
            "pending_calls": len(self._pending),
        }

    def query(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        tool_name: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        keyword: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """多维度查询审计记录。

        Args:
            date_from: 起始日期 (ISO 格式, 含)
            date_to: 结束日期 (ISO 格式, 含)
            tool_name: 工具名精确匹配
            status: 状态精确匹配
            session_id: 会话 ID 精确匹配
            keyword: 关键词模糊匹配 (tool_name/action_name/arguments/output_preview/user_input)
            limit: 返回条数上限
            offset: 偏移量

        Returns:
            匹配记录列表 (按时间倒序)
        """
        if self._db_conn is None:
            return []

        conditions: list[str] = []
        params: list[Any] = []

        if date_from:
            conditions.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            # 如果 date_to 只是日期（无时间），补上当天 23:59:59
            if len(date_to) <= 10:
                date_to = f"{date_to}T23:59:59"
            conditions.append("timestamp <= ?")
            params.append(date_to)
        if tool_name:
            conditions.append("tool_name = ?")
            params.append(tool_name)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if keyword:
            conditions.append(
                "(tool_name LIKE ? OR action_name LIKE ? OR arguments LIKE ? "
                "OR output_preview LIKE ? OR user_input LIKE ? OR error LIKE ?)"
            )
            kw = f"%{keyword}%"
            params.extend([kw] * 6)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = (
            f"SELECT * FROM tool_audit_log WHERE {where} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        try:
            cursor = self._db_conn.execute(sql, params)
            columns = [desc[0] for desc in cursor.description]
            rows = []
            for row in cursor.fetchall():
                d = dict(zip(columns, row))
                # 反序列化 arguments
                if isinstance(d.get("arguments"), str):
                    try:
                        d["arguments"] = json.loads(d["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                rows.append(d)
            return rows
        except Exception as e:
            logger.error("查询审计记录失败: %s", e)
            return []

    def query_count(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        tool_name: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        keyword: str | None = None,
    ) -> int:
        """与 query() 相同条件的计数查询。"""
        if self._db_conn is None:
            return 0

        conditions: list[str] = []
        params: list[Any] = []

        if date_from:
            conditions.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            if len(date_to) <= 10:
                date_to = f"{date_to}T23:59:59"
            conditions.append("timestamp <= ?")
            params.append(date_to)
        if tool_name:
            conditions.append("tool_name = ?")
            params.append(tool_name)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if keyword:
            conditions.append(
                "(tool_name LIKE ? OR action_name LIKE ? OR arguments LIKE ? "
                "OR output_preview LIKE ? OR user_input LIKE ? OR error LIKE ?)"
            )
            kw = f"%{keyword}%"
            params.extend([kw] * 6)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT COUNT(*) FROM tool_audit_log WHERE {where}"

        try:
            cursor = self._db_conn.execute(sql, params)
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error("审计记录计数查询失败: %s", e)
            return 0

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """返回所有会话列表（含调用次数、时间范围）。"""
        if self._db_conn is None:
            return []
        try:
            cursor = self._db_conn.execute("""
                SELECT session_id,
                       COUNT(*) as call_count,
                       MIN(timestamp) as first_call,
                       MAX(timestamp) as last_call,
                       SUM(CASE WHEN status = 'error' OR status = 'denied' THEN 1 ELSE 0 END) as error_count
                FROM tool_audit_log
                WHERE session_id != ''
                GROUP BY session_id
                ORDER BY last_call DESC
            """)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("获取会话列表失败: %s", e)
            return []

    def get_distinct_tool_names(self) -> list[str]:
        """返回数据库中出现过的所有工具名。"""
        if self._db_conn is None:
            return []
        try:
            cursor = self._db_conn.execute(
                "SELECT DISTINCT tool_name FROM tool_audit_log ORDER BY tool_name"
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error("获取工具名列表失败: %s", e)
            return []

    def get_stats_summary(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        """获取统计摘要（总调用数、成功率、平均耗时、Top N 工具）。"""
        if self._db_conn is None:
            return {"total": 0, "success": 0, "error": 0, "success_rate": 0, "avg_duration_ms": 0, "top_tools": []}

        conditions: list[str] = []
        params: list[Any] = []
        if date_from:
            conditions.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            if len(date_to) <= 10:
                date_to = f"{date_to}T23:59:59"
            conditions.append("timestamp <= ?")
            params.append(date_to)

        where = " AND ".join(conditions) if conditions else "1=1"

        try:
            row = self._db_conn.execute(
                f"""SELECT COUNT(*) as total,
                        SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as success,
                        SUM(CASE WHEN status IN ('error','denied') THEN 1 ELSE 0 END) as error,
                        AVG(duration_ms) as avg_ms
                   FROM tool_audit_log WHERE {where}""",
                params,
            ).fetchone()
            total, success, error, avg_ms = row or (0, 0, 0, 0)
            success = success or 0
            error = error or 0
            avg_ms = avg_ms or 0
            rate = round(success / total * 100, 1) if total > 0 else 0

            top_rows = self._db_conn.execute(
                f"""SELECT tool_name, COUNT(*) as cnt
                    FROM tool_audit_log WHERE {where}
                    GROUP BY tool_name ORDER BY cnt DESC LIMIT 10""",
                params,
            ).fetchall()

            return {
                "total": total,
                "success": success,
                "error": error,
                "success_rate": rate,
                "avg_duration_ms": round(avg_ms, 1),
                "top_tools": [(r[0], r[1]) for r in top_rows],
            }
        except Exception as e:
            logger.error("获取统计摘要失败: %s", e)
            return {"total": 0, "success": 0, "error": 0, "success_rate": 0, "avg_duration_ms": 0, "top_tools": []}

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def export_json(
        self,
        output_path: Path,
        date_from: str | None = None,
        date_to: str | None = None,
        tool_name: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        keyword: str | None = None,
    ) -> int:
        """按筛选条件导出为 JSON 文件。

        Returns:
            导出的记录数
        """
        rows = self.query(
            date_from=date_from, date_to=date_to,
            tool_name=tool_name, status=status,
            session_id=session_id, keyword=keyword,
            limit=100000,
        )
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            logger.info("导出 %d 条审计记录到 %s", len(rows), output_path)
        except Exception as e:
            logger.error("导出审计日志 JSON 失败: %s", e)
            return 0
        return len(rows)

    def export_csv(
        self,
        output_path: Path,
        date_from: str | None = None,
        date_to: str | None = None,
        tool_name: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        keyword: str | None = None,
    ) -> int:
        """按筛选条件导出为 CSV 文件。

        Returns:
            导出的记录数
        """
        rows = self.query(
            date_from=date_from, date_to=date_to,
            tool_name=tool_name, status=status,
            session_id=session_id, keyword=keyword,
            limit=100000,
        )
        if not rows:
            return 0

        fieldnames = [
            "id", "timestamp", "tool_name", "action_name", "function_name",
            "arguments", "status", "output_preview", "error", "duration_ms",
            "risk_level", "session_id", "intent", "confidence", "tool_tier",
            "user_input",
        ]
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    # arguments 序列化为字符串
                    if isinstance(row.get("arguments"), dict):
                        row["arguments"] = json.dumps(row["arguments"], ensure_ascii=False)
                    writer.writerow(row)
            logger.info("导出 %d 条审计记录到 CSV %s", len(rows), output_path)
        except Exception as e:
            logger.error("导出审计日志 CSV 失败: %s", e)
            return 0
        return len(rows)

    def get_session_ids_in_filter(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        tool_name: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        keyword: str | None = None,
    ) -> list[str]:
        """获取筛选条件命中的所有不重复 session_id。"""
        if self._db_conn is None:
            return []

        conditions: list[str] = []
        params: list[Any] = []
        if date_from:
            conditions.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            if len(date_to) <= 10:
                date_to = f"{date_to}T23:59:59"
            conditions.append("timestamp <= ?")
            params.append(date_to)
        if tool_name:
            conditions.append("tool_name = ?")
            params.append(tool_name)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if keyword:
            conditions.append(
                "(tool_name LIKE ? OR action_name LIKE ? OR arguments LIKE ? "
                "OR output_preview LIKE ? OR user_input LIKE ? OR error LIKE ?)"
            )
            kw = f"%{keyword}%"
            params.extend([kw] * 6)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""SELECT DISTINCT session_id FROM tool_audit_log
                  WHERE {where} AND session_id != ''"""
        try:
            cursor = self._db_conn.execute(sql, params)
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error("获取筛选 session_id 失败: %s", e)
            return []

    # ------------------------------------------------------------------
    # 文件输出
    # ------------------------------------------------------------------

    def _write_entry(self, entry: AuditEntry) -> None:
        """将单条记录追加写入日志文件。"""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self._log_dir / f"audit-{today}.jsonl"

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("写入审计日志失败: %s", e)

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._db_conn is not None:
            try:
                self._db_conn.close()
                logger.debug("审计日志 SQLite 连接已关闭")
            except Exception:
                pass
            self._db_conn = None

    def clear(self) -> None:
        """清空内存中的审计记录。"""
        self._entries.clear()
        self._pending.clear()
        self._total_calls = 0
        self._total_errors = 0
        self._total_denied = 0
