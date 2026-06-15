"""ExperienceStore — 跨会话经验积累核心模块。

功能：
- 记录/检索 WeClaw 历史诊断经验和工具试错经验
- SQLite 存储结构化元数据，ChromaDB 存储语义向量
- 自动 fallback 到 SQLite LIKE 全文检索（当 ChromaDB/Embedder 不可用时）
- 三层过滤：outcome 过滤 → abstract_pattern 去重 → 时间衰减加权

依赖：
- src/core/rag/embedder.py (Embedder)
- src/core/rag/vector_store.py (VectorStore)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 默认存储路径
_DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".weclaw")
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DATA_DIR, "experiences.db")
_DEFAULT_VECTOR_DIR = os.path.join(_DEFAULT_DATA_DIR, "experience_vectors")
_COLLECTION_NAME = "weclaw_experiences"

# 经验数量上限（LRU 淘汰）
_MAX_EXPERIENCES = 500


@dataclass
class Experience:
    """单条经验记录。"""

    id: int = 0
    trigger: str = ""
    diagnosis: str = ""
    fix_summary: str = ""
    abstract_pattern: str = ""
    outcome: str = "pending"
    source_type: str = "manual"
    related_files: list[str] = field(default_factory=list)
    session_id: str = ""
    tool_names: list[str] = field(default_factory=list)
    hit_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    # recall 时附加字段（不写入 SQLite）
    similarity: float = 0.0


class ExperienceStore:
    """经验存储与检索引擎。"""

    def __init__(
        self,
        db_path: str = "",
        vector_db_dir: str = "",
    ):
        """初始化 ExperienceStore。

        Args:
            db_path: SQLite 数据库路径（默认 ~/.weclaw/experiences.db）
            vector_db_dir: ChromaDB 向量数据库目录（默认 ~/.weclaw/experience_vectors）
        """
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._vector_db_dir = vector_db_dir or _DEFAULT_VECTOR_DIR

        self._embedder = None
        self._vector_store = None
        self._db_conn: Optional[sqlite3.Connection] = None
        # E6: threading.Lock 保护 SQLite 写操作
        self._write_lock = threading.Lock()

        # 确保目录存在
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        os.makedirs(self._vector_db_dir, exist_ok=True)

        # 初始化 SQLite
        self._init_db()

    # ------------------------------------------------------------------
    # 懒加载属性
    # ------------------------------------------------------------------

    @property
    def embedder(self):
        """延迟加载 Embedder 模型。"""
        if self._embedder is None:
            try:
                from src.core.rag import Embedder
                self._embedder = Embedder()
                logger.info("ExperienceStore Embedder 懒加载成功")
            except Exception as e:
                logger.warning("ExperienceStore Embedder 加载失败，将使用 fallback: %s", e)
        return self._embedder

    @property
    def vector_store(self):
        """延迟加载 VectorStore（E5: 传入 embedding_function=self.embedder）。"""
        if self._vector_store is None:
            if self.embedder is None:
                return None  # 无法初始化，后续 recall 走 fallback
            try:
                from src.core.rag import VectorStore
                self._vector_store = VectorStore(
                    db_path=self._vector_db_dir,
                    collection_name=_COLLECTION_NAME,
                    embedding_function=self.embedder,
                )
                logger.info("ExperienceStore VectorStore 懒加载成功")
            except Exception as e:
                logger.warning("ExperienceStore VectorStore 加载失败，将使用 fallback: %s", e)
        return self._vector_store

    # ------------------------------------------------------------------
    # E12: 后台预热
    # ------------------------------------------------------------------

    def _warmup_embedder(self) -> None:
        """后台预热 Embedder 模型，避免首次 recall 延迟。

        由 gui_app 初始化后通过 asyncio.create_task(asyncio.to_thread(...)) 调用。
        """
        try:
            _ = self.embedder
            if self.embedder is not None:
                self.embedder.embed(["warmup"])
                logger.info("ExperienceStore Embedder 预热完成")
        except Exception as e:
            logger.warning("ExperienceStore Embedder 预热失败: %s", e)

    # ------------------------------------------------------------------
    # SQLite 初始化
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """初始化 SQLite 数据库（E6: check_same_thread=False + WAL 模式）。"""
        try:
            self._db_conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
            )
            self._db_conn.execute("PRAGMA journal_mode=WAL")
            self._db_conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger TEXT NOT NULL,
                    diagnosis TEXT NOT NULL,
                    fix_summary TEXT NOT NULL,
                    abstract_pattern TEXT,
                    outcome TEXT DEFAULT 'pending',
                    source_type TEXT DEFAULT 'manual',
                    related_files TEXT,
                    session_id TEXT,
                    tool_names TEXT,
                    hit_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            self._db_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exp_outcome ON experiences(outcome)"
            )
            self._db_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exp_created ON experiences(created_at)"
            )
            self._db_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exp_source ON experiences(source_type)"
            )
            self._db_conn.commit()
            logger.info("ExperienceStore SQLite 初始化成功: %s", self._db_path)
        except Exception as e:
            logger.error("ExperienceStore SQLite 初始化失败: %s", e)
            self._db_conn = None

    # ------------------------------------------------------------------
    # 写入操作
    # ------------------------------------------------------------------

    async def record(
        self,
        trigger: str,
        diagnosis: str,
        fix_summary: str,
        abstract_pattern: str = "",
        outcome: str = "success",
        source_type: str = "manual",
        files: Optional[list[str]] = None,
        session_id: str = "",
        tools: Optional[list[str]] = None,
    ) -> Optional[Experience]:
        """记录一条经验（E3: async def，内部用 asyncio.to_thread 卸载阻塞操作）。

        Args:
            trigger: 触发原因（如 "tool_audit错误率上升"）
            diagnosis: 诊断结论（如 "stock_query浮点解析bug"）
            fix_summary: 修复摘要（如 "添加float_or_none()安全转换"）
            abstract_pattern: 抽象模式（如 "外部数据未做防御性解析"）
            outcome: 结果 success / failure / pending
            source_type: 来源类型 self_correction / tool_retry / user_feedback / manual
            files: 相关文件列表
            session_id: 关联会话 ID
            tools: 涉及工具名列表

        Returns:
            创建的 Experience 对象，失败时返回 None
        """
        if not self._db_conn:
            logger.warning("SQLite 未初始化，无法记录经验")
            return None

        # E7: abstract_pattern 自动生成（调用者未提供时）
        if not abstract_pattern:
            abstract_pattern = await self._generate_abstract_pattern(
                trigger, diagnosis, fix_summary
            )

        now = datetime.now().isoformat()
        files = files or []
        tools = tools or []

        # 将阻塞的 SQLite/ChromaDB 操作卸载到线程池
        exp = await asyncio.to_thread(
            self._write_experience_sync,
            trigger, diagnosis, fix_summary, abstract_pattern,
            outcome, source_type, files, session_id, tools, now,
        )
        return exp

    def _write_experience_sync(
        self,
        trigger: str,
        diagnosis: str,
        fix_summary: str,
        abstract_pattern: str,
        outcome: str,
        source_type: str,
        files: list[str],
        session_id: str,
        tools: list[str],
        now: str,
    ) -> Optional[Experience]:
        """同步写入经验到 SQLite + ChromaDB（在线程池中执行）。"""
        with self._write_lock:
            # 写入 SQLite
            try:
                cursor = self._db_conn.execute(
                    """INSERT INTO experiences
                       (trigger, diagnosis, fix_summary, abstract_pattern,
                        outcome, source_type, related_files, session_id,
                        tool_names, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        trigger, diagnosis, fix_summary, abstract_pattern,
                        outcome, source_type,
                        json.dumps(files, ensure_ascii=False),
                        session_id,
                        json.dumps(tools, ensure_ascii=False),
                        now, now,
                    ),
                )
                self._db_conn.commit()
                exp_id = cursor.lastrowid
            except Exception as e:
                logger.error("写入经验到 SQLite 失败: %s", e)
                return None

            # 写入 ChromaDB（可选，失败不阻断）
            chroma_id = f"exp_{exp_id}"
            text = f"{trigger} {diagnosis} {fix_summary} {abstract_pattern}"
            try:
                if self.vector_store is not None:
                    self.vector_store.add_documents(
                        documents=[text],
                        metadatas=[{
                            "experience_id": exp_id,
                            "outcome": outcome,
                            "source_type": source_type,
                            "created_at": now,
                        }],
                        ids=[chroma_id],
                    )
            except Exception as e:
                logger.warning("写入经验到 ChromaDB 失败（不影响功能）: %s", e)

            # LRU 淘汰：超出上限时删除最旧的经验
            self._enforce_max_experiences()

            return Experience(
                id=exp_id,
                trigger=trigger,
                diagnosis=diagnosis,
                fix_summary=fix_summary,
                abstract_pattern=abstract_pattern,
                outcome=outcome,
                source_type=source_type,
                related_files=files,
                session_id=session_id,
                tool_names=tools,
                created_at=now,
                updated_at=now,
            )

    def _enforce_max_experiences(self) -> None:
        """LRU 淘汰：超出 _MAX_EXPERIENCES 时删除最旧的经验。"""
        try:
            count = self._db_conn.execute(
                "SELECT COUNT(*) FROM experiences"
            ).fetchone()[0]
            if count > _MAX_EXPERIENCES:
                to_delete = count - _MAX_EXPERIENCES
                rows = self._db_conn.execute(
                    f"SELECT id FROM experiences ORDER BY created_at ASC LIMIT {to_delete}"
                ).fetchall()
                for (row_id,) in rows:
                    self._delete_single_sync(row_id)
                logger.info("LRU 淘汰: 删除 %d 条旧经验", to_delete)
        except Exception as e:
            logger.warning("LRU 淘汰检查失败: %s", e)

    # ------------------------------------------------------------------
    # 检索操作
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.6,
    ) -> list[Experience]:
        """检索最相似的历史经验（E13: 三层过滤 + E15: 相似度阈值）。

        注意：此方法为同步阻塞操作，调用方应使用 asyncio.to_thread 包裹。

        Args:
            query: 查询文本
            top_k: 返回数量上限
            min_similarity: 最低相似度阈值（0-1）

        Returns:
            经验列表，按相似度加权分数排序
        """
        if not query or not query.strip():
            return []

        # 尝试向量检索
        results = self._vector_recall(query, top_k * 3)  # 多取一些用于后续过滤

        # Fallback: ChromaDB/Embedder 不可用时使用 SQLite LIKE 全文检索
        if results is None:
            results = self._sqlite_fallback_recall(query, top_k * 3)

        if not results:
            return []

        # E13: 三层过滤
        results = self._filter_by_outcome(results)          # (1) 过滤 outcome=failure
        results = self._deduplicate_by_pattern(results)      # (2) 按 abstract_pattern 去重
        results = self._apply_time_decay(results)            # (3) 时间衰减加权
        results = self._filter_by_similarity(results, min_similarity)

        # 更新命中计数
        for exp in results[:top_k]:
            self._increment_hit_count(exp.id)

        return results[:top_k]

    def _vector_recall(self, query: str, top_k: int) -> Optional[list[Experience]]:
        """向量检索（内部方法）。返回 None 表示不可用。"""
        if self.vector_store is None:
            return None
        try:
            search_results = self.vector_store.query(query, n_results=top_k)
            if not search_results:
                return []

            experiences = []
            for sr in search_results:
                exp_id = sr.metadata.get("experience_id")
                if exp_id is None:
                    continue
                exp = self._get_experience_by_id(exp_id)
                if exp is None:
                    continue
                # ChromaDB distance → similarity (L2 distance，越小越好)
                # 转换为 0-1 相似度分数
                exp.similarity = max(0.0, 1.0 - sr.distance / 2.0)
                experiences.append(exp)
            return experiences
        except Exception as e:
            logger.warning("向量检索失败，将使用 fallback: %s", e)
            return None

    def _sqlite_fallback_recall(self, query: str, top_k: int) -> list[Experience]:
        """SQLite LIKE 全文检索（E10: fallback 策略）。"""
        if not self._db_conn:
            return []
        try:
            kw = f"%{query}%"
            rows = self._db_conn.execute(
                """SELECT * FROM experiences
                   WHERE (trigger LIKE ? OR diagnosis LIKE ? OR fix_summary LIKE ?
                          OR abstract_pattern LIKE ?)
                   ORDER BY created_at DESC LIMIT ?""",
                (kw, kw, kw, kw, top_k),
            ).fetchall()
            return [self._row_to_experience(row, similarity=0.7) for row in rows]
        except Exception as e:
            logger.warning("SQLite fallback 检索失败: %s", e)
            return []

    def _filter_by_outcome(self, experiences: list[Experience]) -> list[Experience]:
        """E13-(1): 过滤 outcome=failure 的经验。"""
        return [e for e in experiences if e.outcome != "failure"]

    def _deduplicate_by_pattern(self, experiences: list[Experience]) -> list[Experience]:
        """E13-(2): 按 abstract_pattern 去重，相同 pattern 只保留最新一条。"""
        seen_patterns: set[str] = set()
        result = []
        for exp in experiences:
            pattern = exp.abstract_pattern.strip()
            if pattern and pattern in seen_patterns:
                continue
            if pattern:
                seen_patterns.add(pattern)
            result.append(exp)
        return result

    def _apply_time_decay(self, experiences: list[Experience]) -> list[Experience]:
        """E13-(3): 时间衰减加权，30天内权重更高。"""
        now = datetime.now()
        for exp in experiences:
            try:
                created = datetime.fromisoformat(exp.created_at)
                days_ago = (now - created).days
                if days_ago <= 30:
                    decay = 1.0
                elif days_ago <= 90:
                    decay = 0.8
                else:
                    decay = 0.6
                exp.similarity = exp.similarity * decay
            except (ValueError, TypeError):
                pass
        # 按加权分数排序
        experiences.sort(key=lambda e: e.similarity, reverse=True)
        return experiences

    def _filter_by_similarity(
        self, experiences: list[Experience], threshold: float
    ) -> list[Experience]:
        """E15: 过滤低于相似度阈值的经验。"""
        return [e for e in experiences if e.similarity >= threshold]

    # ------------------------------------------------------------------
    # 列表/删除/更新
    # ------------------------------------------------------------------

    def list_experiences(
        self,
        limit: int = 20,
        outcome: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> list[Experience]:
        """列出经验（按时间倒序）。"""
        if not self._db_conn:
            return []

        conditions = []
        params: list[Any] = []
        if outcome:
            conditions.append("outcome = ?")
            params.append(outcome)
        if source_type:
            conditions.append("source_type = ?")
            params.append(source_type)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        try:
            rows = self._db_conn.execute(
                f"SELECT * FROM experiences WHERE {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
            return [self._row_to_experience(row) for row in rows]
        except Exception as e:
            logger.error("列出经验失败: %s", e)
            return []

    def forget(self, experience_id: int) -> bool:
        """删除指定经验（从 SQLite + ChromaDB 同步删除）。"""
        if not self._db_conn:
            return False
        with self._write_lock:
            return self._delete_single_sync(experience_id)

    def _delete_single_sync(self, experience_id: int) -> bool:
        """同步删除单条经验（在 _write_lock 内调用）。"""
        try:
            self._db_conn.execute(
                "DELETE FROM experiences WHERE id = ?", (experience_id,)
            )
            self._db_conn.commit()
        except Exception as e:
            logger.error("删除经验 SQLite 记录失败: %s", e)
            return False

        # 从 ChromaDB 删除（使用 exp_{id} 格式 ID）
        try:
            if self.vector_store is not None:
                self.vector_store.delete_by_id(f"exp_{experience_id}")
        except Exception as e:
            logger.warning("删除经验 ChromaDB 记录失败（不影响功能）: %s", e)

        return True

    def update_outcome(self, experience_id: int, outcome: str) -> bool:
        """更新经验结果。"""
        if not self._db_conn:
            return False
        with self._write_lock:
            try:
                self._db_conn.execute(
                    "UPDATE experiences SET outcome = ?, updated_at = ? WHERE id = ?",
                    (outcome, datetime.now().isoformat(), experience_id),
                )
                self._db_conn.commit()
                return True
            except Exception as e:
                logger.error("更新经验结果失败: %s", e)
                return False

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _get_experience_by_id(self, exp_id: int) -> Optional[Experience]:
        """按 ID 获取单条经验。"""
        if not self._db_conn:
            return None
        try:
            row = self._db_conn.execute(
                "SELECT * FROM experiences WHERE id = ?", (exp_id,)
            ).fetchone()
            if row:
                return self._row_to_experience(row)
        except Exception as e:
            logger.warning("获取经验 %d 失败: %s", exp_id, e)
        return None

    def _row_to_experience(self, row: tuple, similarity: float = 0.0) -> Experience:
        """将 SQLite 行转为 Experience 对象。"""
        # 列顺序：id, trigger, diagnosis, fix_summary, abstract_pattern,
        #          outcome, source_type, related_files, session_id, tool_names,
        #          hit_count, created_at, updated_at
        def _parse_json_list(val: str) -> list[str]:
            if not val:
                return []
            try:
                result = json.loads(val)
                return result if isinstance(result, list) else []
            except (json.JSONDecodeError, TypeError):
                return []

        return Experience(
            id=row[0],
            trigger=row[1],
            diagnosis=row[2],
            fix_summary=row[3],
            abstract_pattern=row[4] or "",
            outcome=row[5] or "pending",
            source_type=row[6] or "manual",
            related_files=_parse_json_list(row[7]),
            session_id=row[8] or "",
            tool_names=_parse_json_list(row[9]),
            hit_count=row[10] or 0,
            created_at=row[11] or "",
            updated_at=row[12] or "",
            similarity=similarity,
        )

    def _increment_hit_count(self, exp_id: int) -> None:
        """增加经验命中计数。"""
        if not self._db_conn:
            return
        try:
            self._db_conn.execute(
                "UPDATE experiences SET hit_count = hit_count + 1, updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), exp_id),
            )
            self._db_conn.commit()
        except Exception as e:
            logger.debug("更新 hit_count 失败: %s", e)

    async def _generate_abstract_pattern(
        self, trigger: str, diagnosis: str, fix_summary: str
    ) -> str:
        """E7: 自动生成 abstract_pattern（轻量模板，不依赖 LLM）。

        使用关键词提取 + 模板组合，避免 LLM 调用开销。
        """
        # 从 trigger + diagnosis + fix_summary 中提取关键信息生成抽象 pattern
        parts = []
        # 提取错误类型关键词
        error_keywords = [
            "超时", "timeout", "错误", "error", "异常", "exception",
            "失败", "fail", "拒绝", "denied", "崩溃", "crash",
            "浮点", "float", "解析", "parse", "连接", "connect",
            "权限", "permission", "编码", "encoding", "内存", "memory",
        ]
        combined = f"{trigger} {diagnosis} {fix_summary}".lower()
        matched = [kw for kw in error_keywords if kw in combined]
        if matched:
            parts.append(":".join(matched[:3]))

        # 提取工具名
        import re
        tool_pattern = re.search(r'(\w+)_(\w+)', combined)
        if tool_pattern:
            parts.append(f"{tool_pattern.group(1)}工具问题")

        # 提取修复动作关键词
        fix_keywords = [
            "防御性", "容错", "重试", "fallback", "降级", "安全转换",
            "异常处理", "类型检查", "参数校验", "默认值", "空值检查",
        ]
        fix_matched = [kw for kw in fix_keywords if kw in combined]
        if fix_matched:
            parts.append(f"修复:{'+'.join(fix_matched[:2])}")

        if parts:
            return " | ".join(parts)

        # 兜底：取 diagnosis 前 60 字符
        return diagnosis[:60] if diagnosis else ""

    # ------------------------------------------------------------------
    # 资源清理
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭 SQLite 连接。"""
        if self._db_conn is not None:
            try:
                self._db_conn.close()
                logger.debug("ExperienceStore SQLite 连接已关闭")
            except Exception:
                pass
            self._db_conn = None

    def __del__(self) -> None:
        self.close()
