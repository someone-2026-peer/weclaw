"""会话管理器 — 管理对话历史、多会话切换、上下文窗口截断。

支持：
- 多会话管理（创建/切换/删除）
- 对话历史 messages 列表管理
- 上下文窗口自动截断（保留 system prompt + 最近消息）
- 会话元数据（标题、创建时间等）

Phase 4.4 增强：
- 持久化存储集成（可选）
- 自动保存消息到 SQLite
- 自动标题生成

Phase 4.7 增强：
- 智能截断：保留完整对话轮次（user+assistant 为一轮）
- tool_calls 完整性：不截断 tool_calls 和对应 tool 结果
- 消息数量硬上限：防止内存无限增长

Phase 5 增强：
- 上下文压缩：使用辅助 LLM 对早期消息生成摘要，替代简单截断
- 向后兼容：辅助模型未配置时回退到原有截断策略

Phase 6+ 优化（长对话截断优化 v2）：
- 修复 _try_compress 调用链：token_threshold 独立于 max_tokens 生效
- 统一 Token 估算：委托给 token_utils（含 reasoning_content）
- 安全压缩缓存：复合键 (msg_count, sp_hash, limit) 避免无效重压缩
- 预留输出 token 空间：output_reserve 参数防止小窗口模型 API 报错
- System Prompt 膨胀监控：超过 8K tokens 时记录警告
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from src.core.token_utils import estimate_tokens as _unified_estimate_tokens
from src.core.token_utils import estimate_chars_per_token

if TYPE_CHECKING:
    from src.core.storage import ChatStorage
    from src.core.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)

# 默认最大消息数量硬上限
# 【v2.31.1 提升】100 → 500，适配 DeepSeek 1M 等大上下文模型
DEFAULT_MAX_MESSAGE_COUNT = 500


@dataclass
class Session:
    """单个对话会话。"""

    id: str
    title: str = "新对话"
    created_at: datetime = field(default_factory=datetime.now)
    messages: list[dict[str, Any]] = field(default_factory=list)
    model_key: str = ""  # 此会话使用的模型
    total_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "local"

    # ── 压缩缓存（Phase 6+ 新增，不参与序列化）──────────────────────────────
    # 缓存 get_messages() 的压缩/截断结果，避免 ReAct 循环中重复压缩
    _compressed_cache: list[dict[str, Any]] | None = field(default=None, repr=False)
    # 复合缓存键：(消息数量, system_prompt_hash, token_limit)
    _cache_key: tuple[int, int, int] = field(default=(0, 0, 0), repr=False)
    # 【v4.23.0】孤儿 tool_call 清理标记（DB 加载后一次性清理）
    _orphans_cleaned: bool = field(default=False, repr=False)
    # ── v4.27.0 DeepSeek 缓存优化（不参与序列化）─────────────────────────────
    # 前缀稳定性保护：上次发送给 API 的消息数量与前缀 hash
    _last_prefix_msg_count: int = field(default=0, repr=False)
    _last_sent_prefix_hash: str = field(default="", repr=False)
    # DeepSeek 缓存命中统计（累计，用于审计与 UI 展示）
    cache_hit_tokens: int = field(default=0, repr=False)
    cache_miss_tokens: int = field(default=0, repr=False)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def has_system_prompt(self) -> bool:
        return bool(self.messages) and self.messages[0].get("role") == "system"


class SessionManager:
    """会话管理器。

    用法::

        mgr = SessionManager(context_window=64000)
        session = mgr.create_session(title="测试对话")
        mgr.add_message(role="user", content="你好")
        messages = mgr.get_messages()  # 自动截断
    """

    def __init__(
        self,
        context_window: int = 128000,
        max_sessions: int = 50,
        system_prompt: str = "",
        max_message_count: int = DEFAULT_MAX_MESSAGE_COUNT,
        storage: ChatStorage | None = None,
        context_compressor: ContextCompressor | None = None,
    ):
        """
        Args:
            context_window: 上下文窗口大小（token 数估算用字符数/3近似）
            max_sessions: 最大会话数量
            system_prompt: 默认 system prompt
            max_message_count: 最大消息数量硬上限（防止内存无限增长）
            storage: 可选的持久化存储实例
            context_compressor: 可选的上下文压缩器（Phase 5 新增）
        """
        self._sessions: dict[str, Session] = {}
        self._current_id: str = ""
        self._context_window = context_window
        self._max_sessions = max_sessions
        self._system_prompt = system_prompt
        self._max_message_count = max_message_count
        self._storage = storage
        self._compressor = context_compressor
        # ★ Task 10: 异步压缩 Task 管理（按 session_id 索引）
        self._pending_summary_tasks: dict[str, Any] = {}  # asyncio.Task

        # 自动创建第一个会话
        self.create_session(title="默认对话")

    def set_context_window(self, context_window: int) -> None:
        """动态更新上下文窗口大小。

        【v3.27.1 新增】支持模型切换时动态调整截断阈值。
        解决问题：DeepSeek V4 有 1M context_window，但默认只使用 128000 导致过早截断。

        Args:
            context_window: 新的上下文窗口大小（token数）
        """
        if context_window > 0 and context_window != self._context_window:
            old_value = self._context_window
            self._context_window = context_window
            logger.info(
                "上下文窗口已更新: %d → %d tokens",
                old_value, context_window
            )

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    def create_session(self, title: str = "新对话", model_key: str = "", source: str = "local") -> Session:
        """创建新会话并切换到它。
        
        Args:
            title: 会话标题
            model_key: 模型标识
            source: 会话来源标识，"local" 或 "pwa:{user_id}"
        """
        session_id = str(uuid.uuid4())[:8]
        session = Session(id=session_id, title=title, model_key=model_key, source=source)

        # 自动添加 system prompt
        if self._system_prompt:
            session.messages.append({
                "role": "system",
                "content": self._system_prompt,
            })

        self._sessions[session_id] = session
        self._current_id = session_id

        # 【多Agent日志】创建会话后自动设置session_id上下文
        try:
            from src.core.logging_config import set_current_session_id
            set_current_session_id(session_id)
        except ImportError:
            pass  # 启动早期可能无法导入

        # 清理过多的旧会话
        if len(self._sessions) > self._max_sessions:
            self._cleanup_oldest()

        logger.info("创建会话: %s (%s, source=%s)", session_id, title, source)

        # 异步保存到存储（fire-and-forget）
        if self._storage:
            import asyncio
            from src.core.storage import StoredSession
            stored = StoredSession(
                id=session_id,
                title=title,
                model_key=model_key,
                created_at=session.created_at,
                updated_at=session.created_at,
                source=source,
            )
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._storage.save_session(stored))
                else:
                    loop.run_until_complete(self._storage.save_session(stored))
            except Exception as e:
                logger.warning("保存会话到存储失败: %s", e)

        return session

    def switch_session(self, session_id: str) -> Session:
        """切换到指定会话。"""
        if session_id not in self._sessions:
            raise ValueError(f"会话不存在: {session_id}")
        self._current_id = session_id

        # ★ F1: 切换会话时重置压缩器状态，防止跨会话泄漏
        if self._compressor:
            self._compressor.reset()

        # 【多Agent日志】切换会话后自动设置session_id上下文
        try:
            from src.core.logging_config import set_current_session_id
            set_current_session_id(session_id)
        except ImportError:
            pass

        logger.info("切换到会话: %s", session_id)
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话。"""
        if session_id not in self._sessions:
            return False

        # ★ Task 10: 取消待处理的异步压缩 Task
        self._cancel_pending_task(session_id)
        del self._sessions[session_id]

        # 如果删除的是当前会话，切换到最近的
        if self._current_id == session_id:
            if self._sessions:
                self._current_id = list(self._sessions.keys())[-1]
            else:
                # 没有会话了，创建一个新的
                self.create_session()

        logger.info("删除会话: %s", session_id)
        return True

    def update_system_prompt(self, new_prompt: str, session_id: str = "") -> None:
        """更新当前（或指定）会话的 System Prompt。

        用于动态构建 System Prompt，根据用户意图注入不同的扩展模块。

        Args:
            new_prompt: 新的 System Prompt 内容
            session_id: 指定会话ID（默认用当前会话）
        """
        session = self._get_session(session_id)
        # 更新内部存储
        self._system_prompt = new_prompt
        # 更新会话中的 system 消息
        if session.messages and session.messages[0].get("role") == "system":
            session.messages[0]["content"] = new_prompt
        else:
            # 如果没有 system 消息，在开头插入
            session.messages.insert(0, {"role": "system", "content": new_prompt})

        # ★ Phase 6+ 膨胀监控：System Prompt 随对话进展持续增长
        # 技能上下文、文件路径上下文、工具选择指引会累积，长程对话中可能膨胀
        sp_tokens_est = len(new_prompt) // 2  # 粗略估算（中英文混合）
        if sp_tokens_est > 15000:
            logger.warning(
                "System Prompt 膨胀警告: ~%d tokens (%d 字符), "
                "会话 %s, 建议检查技能/文件路径上下文注入",
                sp_tokens_est, len(new_prompt), session.id,
            )
        # ★ 缓存自动失效：sp_hash 在 get_messages 的 cache_key 中会检测到变化
        logger.debug("已更新 System Prompt (长度: %d, ~%d tokens)", len(new_prompt), sp_tokens_est)

    @property
    def current_session(self) -> Session:
        """获取当前会话。"""
        return self._sessions[self._current_id]

    @property
    def current_session_id(self) -> str:
        return self._current_id

    def list_sessions(self) -> list[Session]:
        """列出所有会话，按创建时间降序。"""
        return sorted(
            self._sessions.values(),
            key=lambda s: s.created_at,
            reverse=True,
        )

    def get_or_create_session(
        self,
        session_id: str,
        source: str = "local",
        title: str = "新对话",
        model_key: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """获取或创建指定 session_id 的会话（不切换 _current_id）。

        用于 PWA 会话隔离：在 PWA 请求处理前，查找或创建专属会话，
        但不改变桌面端的 _current_id，避免 GUI 闪烁。

        Args:
            session_id: 会话 ID
            source: 来源标识，如 "pwa:{user_id}"
            title: 新会话的标题（仅在创建时使用）
            model_key: 模型标识
            metadata: 会话元数据

        Returns:
            找到或新建的 Session 实例
        """
        if session_id in self._sessions:
            return self._sessions[session_id]

        session = Session(
            id=session_id,
            title=title,
            model_key=model_key,
            source=source,
            metadata=metadata or {},
        )
        if self._system_prompt:
            session.messages.insert(0, {
                "role": "system",
                "content": self._system_prompt,
            })
        self._sessions[session_id] = session
        logger.info("创建PWA专属会话: %s (source=%s)", session_id, source)

        # 异步保存到存储
        if self._storage:
            import asyncio
            from src.core.storage import StoredSession
            stored = StoredSession(
                id=session_id,
                title=title,
                model_key=model_key,
                created_at=session.created_at,
                updated_at=session.created_at,
                source=source,
                metadata=metadata or {},
            )
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._storage.save_session(stored))
                else:
                    loop.run_until_complete(self._storage.save_session(stored))
            except Exception as e:
                logger.warning("保存PWA会话到存储失败: %s", e)

        return session

    # ------------------------------------------------------------------
    # 消息管理
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        session_id: str = "",
        **extra: Any,
    ) -> None:
        """向当前（或指定）会话添加消息。

        Args:
            role: "user" | "assistant" | "system" | "tool"
            content: 消息内容
            session_id: 指定会话ID（默认用当前会话）
            **extra: 额外字段（如 tool_calls, tool_call_id）
        """
        session = self._get_session(session_id)
        msg: dict[str, Any] = {"role": role, "content": content}
        msg.update(extra)
        session.messages.append(msg)

        # 【v4.23.0】安全阀：仅在消息数远超上限时触发
        # 防止 session.messages 无限增长影响 UI 渲染/Curator/DB 加载
        if len(session.messages) > self._max_message_count * 2:
            self._enforce_message_limit(session)

        # 异步保存到存储（fire-and-forget）
        self._save_message_to_db(session, role, content, extra)

    def add_tool_message(
        self,
        tool_call_id: str,
        content: str,
        session_id: str = "",
    ) -> None:
        """添加工具结果消息。"""
        self.add_message(
            role="tool",
            content=content,
            session_id=session_id,
            tool_call_id=tool_call_id,
        )

    def add_message_batch(
        self,
        messages: list[tuple[str, str, dict[str, Any]]],
        session_id: str = "",
    ) -> None:
        """批量添加消息，中间不触发 _enforce_message_limit。

        用于 ReAct 步骤内保证 assistant + tool_results 的原子性，
        避免中间状态被安全阀截断破坏 tool 配对。

        【v4.23.0 新增】解决 P1: ReAct 步骤内截断破坏 tool 配对。

        Args:
            messages: [(role, content, extra_dict), ...] 列表
            session_id: 指定会话ID
        """
        if not messages:
            return
        session = self._get_session(session_id)
        try:
            for role, content, extra in messages:
                msg: dict[str, Any] = {"role": role, "content": content}
                if extra:
                    msg.update(extra)
                session.messages.append(msg)
                # 逐条持久化到 DB（保持 fire-and-forget 语义）
                self._save_message_to_db(session, role, content, extra or {})
        except Exception:
            # 异常时已添加的消息不丢失，仍然保留在 session.messages 中
            logger.warning("add_message_batch 部分失败，已添加的消息保留", exc_info=True)
        finally:
            # batch 结束时统一清除压缩缓存
            session._compressed_cache = None
            session._cache_key = (0, 0, 0)

    def _save_message_to_db(
        self,
        session: Session,
        role: str,
        content: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """将单条消息异步保存到存储（fire-and-forget）。"""
        if not self._storage:
            return
        import asyncio
        from src.core.storage import StoredMessage
        extra_dict = extra or {}
        stored_msg = StoredMessage(
            id=None,
            session_id=session.id,
            role=role,
            content=content,
            tool_calls=extra_dict.get("tool_calls"),
            tool_call_id=extra_dict.get("tool_call_id"),
            attachments=extra_dict.get("attachments"),
        )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._storage.save_message(stored_msg))
            else:
                loop.run_until_complete(self._storage.save_message(stored_msg))
        except Exception as e:
            logger.warning("保存消息到存储失败: %s", e)

    def cleanup_incomplete_tool_calls(self, session_id: str = "") -> int:
        """清理未完成的 tool_calls 消息。

        当 assistant 消息包含 tool_calls 但没有对应的 tool 消息时，
        需要补全错误消息或移除不完整的 assistant 消息。

        【v2.31.1 修复】搜索范围从连续 tool 消息扩展为当前轮次内
        所有后续 tool 消息（直到下一个 user/assistant 边界），防止
        因消息结构变化导致的漏检。

        Args:
            session_id: 指定会话ID

        Returns:
            清理的消息数量
        """
        session = self._get_session(session_id)
        messages = session.messages
        if not messages:
            return 0

        cleaned = 0
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    # 收集所需的所有 tool_call_id
                    expected_ids = {tc.get("id") for tc in tool_calls if tc.get("id")}
                    if not expected_ids:
                        i += 1
                        continue

                    # 【修复】搜索所有后续 tool 消息直到下一个 user/assistant 边界
                    found_ids: set[str] = set()
                    insert_positions: list[int] = []  # 已找到的 tool 消息位置
                    j = i + 1
                    while j < len(messages):
                        nxt = messages[j]
                        nxt_role = nxt.get("role", "")
                        if nxt_role == "tool":
                            nid = nxt.get("tool_call_id", "")
                            if nid in expected_ids and nid not in found_ids:
                                found_ids.add(nid)
                                insert_positions.append(j)
                            j += 1
                        elif nxt_role in ("user", "assistant"):
                            # 到达下一个对话边界，停止搜索
                            break
                        else:
                            j += 1

                    # 检查是否所有 tool_call 都有响应
                    missing_ids = expected_ids - found_ids
                    if missing_ids:
                        logger.warning(
                            "发现未完成的 tool_calls: %s, 补全错误响应",
                            missing_ids,
                        )
                        # 在最后一个已找到的 tool 消息后插入错误响应
                        if insert_positions:
                            insert_at = max(insert_positions) + 1
                        else:
                            insert_at = i + 1  # 没有找到任何 tool 消息，插在 assistant 后
                        for call_id in sorted(missing_ids):
                            error_msg = {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": "[系统] 工具调用被中断，请重新描述您的需求。",
                            }
                            messages.insert(insert_at, error_msg)
                            insert_at += 1
                            cleaned += 1
            i += 1

        if cleaned > 0:
            logger.info("清理了 %d 条未完成的 tool_calls 消息", cleaned)

        return cleaned

    def add_assistant_message(
        self,
        content: str,
        tool_calls: list[dict] | None = None,
        session_id: str = "",
        reasoning_content: str | None = None,
    ) -> None:
        """添加助手消息（可能包含 tool_calls 和 reasoning_content）。

        Args:
            content: 消息内容
            tool_calls: 工具调用列表
            session_id: 指定会话ID
            reasoning_content: DeepSeek 等模型的思考过程内容，
                当存在 tool_calls 时必须回传给 API
        """
        extra: dict[str, Any] = {}
        if tool_calls:
            extra["tool_calls"] = tool_calls
        if reasoning_content:
            extra["reasoning_content"] = reasoning_content
        self.add_message(
            role="assistant",
            content=content,
            session_id=session_id,
            **extra,
        )

    def get_messages(
        self,
        session_id: str = "",
        max_tokens: int = 0,
        output_reserve: int = 0,
    ) -> list[dict[str, Any]]:
        """获取当前会话的消息列表（自动截断以适应上下文窗口）。

        Phase 5 增强：优先使用上下文压缩（辅助 LLM 摘要），
        压缩失败或未配置时回退到原有截断策略。

        Phase 6+ 优化：
        - 安全压缩缓存（复合键避免无效重压缩）
        - 预留输出 token 空间（output_reserve）

        Args:
            session_id: 指定会话ID
            max_tokens: 最大 token 数（0=使用默认 context_window）
            output_reserve: 预留给模型输出的 token 数（0=使用 2% 安全余量）

        Returns:
            截断后的消息列表
        """
        session = self._get_session(session_id)
        window = max_tokens or self._context_window

        # ★ Phase 6+ 预留输出空间：取 output_reserve 和 2% 中的较大值
        # 解决：小窗口模型（如 GLM-4-Flash 128K 窗口 + 8K 输出）
        #   仅 2% 余量 = 2560 tokens < 8000 tokens 输出需求 → API 报错
        reserve = max(output_reserve or 0, int(window * 0.02))
        limit = window - reserve

        messages = session.messages

        # 【v4.23.0】一次性清理 DB 加载的孤儿 tool_call
        # 解决：旧代码截断遗留的 assistant(tool_calls) 无对应 tool 结果，
        # 导致 _validate_message_structure 每步重复检测并添加占位消息
        if not session._orphans_cleaned and len(messages) > 1:
            all_tool_result_ids: set[str] = set()
            for _m in messages:
                if _m.get("role") == "tool":
                    _tid = _m.get("tool_call_id", "")
                    if _tid:
                        all_tool_result_ids.add(_tid)

            _orphan_count = 0
            _to_remove: list[int] = []
            for _idx, _m in enumerate(messages):
                if _m.get("role") == "assistant" and _m.get("tool_calls"):
                    _valid_tcs = [
                        tc for tc in _m["tool_calls"]
                        if (tc.get("id", "") if isinstance(tc, dict) else "") in all_tool_result_ids
                    ]
                    _orphan_count += len(_m["tool_calls"]) - len(_valid_tcs)
                    if _valid_tcs:
                        if len(_valid_tcs) < len(_m["tool_calls"]):
                            messages[_idx] = dict(_m)
                            messages[_idx]["tool_calls"] = _valid_tcs
                    else:
                        _content = str(_m.get("content", "") or "")
                        if _content.strip():
                            _cleaned = dict(_m)
                            _cleaned.pop("tool_calls", None)
                            messages[_idx] = _cleaned
                        else:
                            _to_remove.append(_idx)

            if _to_remove:
                for _idx in reversed(_to_remove):
                    messages.pop(_idx)
            if _orphan_count > 0 or _to_remove:
                session._compressed_cache = None
                session._cache_key = (0, 0, 0)
                logger.info("一次性清理 %d 个孤儿 tool_call（来自 DB）", _orphan_count)
            session._orphans_cleaned = True

        # ★ Phase 6+ 安全压缩缓存（复合键：消息数量 + system_prompt_hash + limit）
        current_count = len(messages)
        sp_hash = hash(messages[0]["content"]) if (
            messages and messages[0].get("role") == "system"
        ) else 0
        cache_key = (current_count, sp_hash, limit)

        if (session._compressed_cache is not None
                and session._cache_key == cache_key):
            logger.debug(
                "使用压缩缓存 (count=%d, limit=%d)，跳过重新估算/压缩",
                current_count, limit,
            )
            # 【v4.23.0】缓存命中：直接返回副本，跳过验证
            # 缓存内容已是上一次 _validate_message_structure 的输出，
            # 再次调用可能误删缓存中的合法占位消息
            return list(session._compressed_cache)

        # 估算 token（统一使用 token_utils）
        estimated_tokens = self._estimate_tokens(messages)

        # ★ v4.23.0 Phase 1: tool 结果分级裁剪（零 LLM 成本的廉价预处理）
        # 在压缩/截断之前，先用三级 Pass 裁剪旧 tool 结果，可能避免昂贵的 LLM 压缩
        # ★ v4.27.0 前缀稳定性保护：保护已发送给 API 的消息前缀不被修改
        _prefix_protect_count = session._last_prefix_msg_count if session._last_sent_prefix_hash else 0
        if self._compressor and estimated_tokens > limit:
            protect_tail = int(limit * 0.3)  # 保护最近 30% 的消息不裁剪
            pruned = self._compressor.prune_tool_results_advanced(
                messages, protect_tail_tokens=protect_tail,
                preserve_prefix_count=_prefix_protect_count,
            )
            pruned_tokens = self._estimate_tokens(pruned)
            if pruned_tokens < estimated_tokens:
                logger.info(
                    "Phase 1 tool 裁剪: %d -> %d tokens (节省 %d)",
                    estimated_tokens, pruned_tokens, estimated_tokens - pruned_tokens,
                )
                messages = pruned
                estimated_tokens = pruned_tokens

        if estimated_tokens <= limit:
            result = list(messages)
        else:
            # ★ Task 10: 检查是否有已完成的异步压缩结果
            _async_result = self._check_async_result(session.id, messages)
            if _async_result is not None:
                result = _async_result
            else:
                # Phase 5: 优先尝试上下文压缩（异步方法需要同步包装）
                if self._compressor:
                    compressed = self._try_compress(messages, limit)
                    if compressed is not None:
                        result = compressed
                    else:
                        # 压缩失败，回退到截断 + 调度异步任务
                        logger.warning(
                            "消息超限，触发截断: 估算 %d tokens > 限制 %d tokens",
                            estimated_tokens, limit,
                        )
                        result = self._truncate_messages(messages, limit)
                        # ★ Task 10: 截断后调度异步压缩（下一轮可用）
                        self._schedule_async_compress(session.id, messages, limit)
                else:
                    # 回退到原有截断策略
                    logger.warning(
                        "消息超限，触发截断: 估算 %d tokens > 限制 %d tokens",
                        estimated_tokens, limit,
                    )
                    result = self._truncate_messages(messages, limit)

        # 【v2.31.1 修复】返回前最终验证：确保消息结构对 LLM API 合法
        # ★ Task 3: 截断通知注入 — 防止模型对丢失上下文一无所知
        _n_dropped = len(messages) - len(result)
        if _n_dropped > 0 and result and result[0].get("role") == "system":
            try:
                from src.config import get_config
                _notice_enabled = get_config("compressor.truncation_notice_enabled", True)
            except Exception:
                _notice_enabled = True
            if _notice_enabled:
                _notice = (
                    f"\n\n[上下文提示] 为适应模型窗口限制，早期 {_n_dropped} 条消息已被截断。"
                    "请基于当前可见的对话内容继续。"
                )
                _sp_content = result[0].get("content", "")
                if _notice not in _sp_content:
                    result[0] = dict(result[0])  # 浅拷贝避免修改原始 system 消息
                    result[0]["content"] = _sp_content + _notice

        result = self._validate_message_structure(result)

        # ★ v4.27.0 DeepSeek 优化：剥离非工具轮次的 reasoning_content
        # DeepSeek API 会静默忽略这些字段，发送它们浪费 token
        try:
            from src.config import get_config
            _strip_enabled = get_config("compressor.strip_unused_reasoning", True)
        except Exception:
            _strip_enabled = True
        if _strip_enabled:
            result = self._strip_unused_reasoning(result)

        # ★ Phase 6+ 缓存压缩结果（避免 ReAct 循环中重复压缩）
        session._compressed_cache = result
        session._cache_key = cache_key

        # ★ v4.27.0 前缀稳定性保护：记录本次发送的消息前缀信息
        try:
            from src.config import get_config
            _prefix_enabled = get_config("compressor.prefix_stability_protection", True)
        except Exception:
            _prefix_enabled = True
        if _prefix_enabled:
            session._last_prefix_msg_count = len(result)
            session._last_sent_prefix_hash = hashlib.md5(
                json.dumps(result[:10], ensure_ascii=False, default=str).encode()
            ).hexdigest()

        return result

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """估算消息列表的 token 数量。

        Phase 6+ 优化：委托给 token_utils.estimate_tokens()，
        确保 SessionManager 与 ContextCompressor 使用相同算法。
        额外计入 reasoning_content（DeepSeek 思考内容）。
        """
        return _unified_estimate_tokens(messages)

    @staticmethod
    def _strip_unused_reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """剥离非工具调用轮次的 reasoning_content。

        DeepSeek API 文档明确规定：两个 user 消息之间如果模型未进行工具调用，
        则中间 assistant 的 reasoning_content 在后续轮次传入时会被静默忽略。
        发送这些字段只会浪费 token。

        规则：
        - assistant 消息有 tool_calls → 保留 reasoning_content（API 要求必须回传）
        - assistant 消息无 tool_calls → 移除 reasoning_content（API 会忽略）

        v4.27.0: 仅修改副本，不影响 session 存储的原始消息。
        """
        result = []
        for msg in messages:
            if (msg.get("role") == "assistant"
                    and not msg.get("tool_calls")
                    and "reasoning_content" in msg):
                cleaned = dict(msg)
                del cleaned["reasoning_content"]
                result.append(cleaned)
            else:
                result.append(msg)
        return result

    def clear_messages(self, session_id: str = "") -> None:
        """清空当前会话的消息（保留 system prompt）。"""
        session = self._get_session(session_id)
        # ★ Task 10: 取消待处理的异步压缩 Task
        self._cancel_pending_task(session.id)
        system_msg = None
        if session.has_system_prompt:
            system_msg = session.messages[0]

        session.messages.clear()
        session.total_tokens = 0
        # ★ Phase 6+ 清除压缩缓存（消息列表已重置）
        session._compressed_cache = None
        session._cache_key = (0, 0, 0)

        if system_msg:
            session.messages.append(system_msg)

        logger.info("清空会话消息: %s", session.id)

    def update_title(self, title: str, session_id: str = "") -> None:
        """更新会话标题。"""
        session = self._get_session(session_id)
        session.title = title

    def update_tokens(self, tokens: int, session_id: str = "") -> None:
        """更新会话累计 token 数。"""
        session = self._get_session(session_id)
        session.total_tokens += tokens

    # ------------------------------------------------------------------
    # 上下文压缩（Phase 5 新增）
    # ------------------------------------------------------------------

    def set_compressor(self, compressor: ContextCompressor | None) -> None:
        """设置或更新上下文压缩器。

        允许在运行时动态注入压缩器（例如在辅助模型配置加载后）。

        Args:
            compressor: 上下文压缩器实例，None 表示禁用压缩
        """
        self._compressor = compressor
        if compressor:
            logger.info("上下文压缩器已启用")
        else:
            logger.info("上下文压缩器已禁用，将使用截断策略")

    def set_compression_threshold(self, threshold: int) -> None:
        """更新上下文压缩触发阈值（委托给 compressor）。

        Phase 6+ 新增：供 gui_app._on_model_changed 在模型切换时调用，
        避免从 UI 层穿透 private 属性 (_compressor) 访问压缩器。

        同时清除所有会话的压缩缓存（阈值变化后缓存不再有效）。

        Args:
            threshold: 新的压缩触发阈值（token 数）
        """
        if self._compressor:
            old = self._compressor._token_threshold
            self._compressor.set_token_threshold(threshold)
            if threshold != old:
                logger.info("压缩阈值已更新: %d → %d tokens", old, threshold)
                # 仅在阈值变化时清除压缩缓存
                for sess in self._sessions.values():
                    sess._compressed_cache = None
                    sess._cache_key = (0, 0, 0)

    def manual_compress(
        self,
        focus_topic: str = "",
        session_id: str = "",
    ) -> dict[str, Any] | None:
        """Task 7: 手动触发上下文压缩。

        仅在 ReAct 循环未激活时可用，防止压缩期间 add_message
        追加新消息后整体替换导致消息丢失。

        Args:
            focus_topic: 可选的焦点主题引导
            session_id: 指定会话ID

        Returns:
            压缩统计信息 dict，失败时返回 None
        """
        if not self._compressor:
            logger.info("手动压缩失败: 未配置压缩器")
            return None

        session = self._get_session(session_id)
        messages = session.messages

        if len(messages) < 5:
            logger.info("手动压缩跳过: 消息数不足 (%d)", len(messages))
            return None

        try:
            # 使用同步包装调用异步压缩
            compressed = self._try_compress(messages, self._context_window)
            if compressed is None:
                return None

            # 更新会话消息
            session.messages = compressed
            session._compressed_cache = None
            session._cache_key = (0, 0, 0)

            stats = self._compressor.last_stats
            return {
                "original_messages": stats.original_messages if stats else len(messages),
                "compressed_messages": stats.compressed_messages if stats else len(compressed),
                "original_tokens": stats.original_tokens if stats else 0,
                "compressed_tokens": stats.compressed_tokens if stats else 0,
                "compression_ratio": round(stats.compression_ratio * 100, 1) if stats else 0,
                "focus_topic": focus_topic or "",
            }
        except Exception as e:
            logger.warning("手动压缩失败: %s", e)
            return None

    def _try_compress(
        self,
        messages: list[dict[str, Any]],
        token_limit: int,
    ) -> list[dict[str, Any]] | None:
        """尝试使用上下文压缩器压缩消息。

        由于 get_messages() 是同步方法，需要在事件循环中运行异步压缩。
        压缩失败时返回 None，由调用方回退到截断策略。

        Phase 6+ 修复：needs_compression 不再传入 token_limit，
        让压缩器使用自身的 token_threshold（软阈值）判断是否需要压缩。
        这修复了 token_limit（=context_window*0.98）覆盖 token_threshold
        导致压缩机制形同虚设的问题。

        Args:
            messages: 原始消息列表
            token_limit: token 上限（用于 compress 的目标上限，非预检条件）

        Returns:
            压缩后的消息列表，失败时返回 None
        """
        if not self._compressor:
            return None

        try:
            import asyncio

            # ★ Phase 6+ 修复：不传 token_limit，使用 compressor 自身的
            #   token_threshold（软阈值）作为预检条件。
            #   此前传入 token_limit（=context_window*0.98）覆盖了
            #   self._token_threshold，导致 32K 阈值从未生效。
            if not self._compressor.needs_compression(messages):
                return None

            # 在事件循环中运行异步压缩
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 已在事件循环中（GUI/远程场景），创建 task
                import concurrent.futures
                # 使用新线程中的事件循环来运行异步压缩，避免嵌套 await
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(self._run_compress_sync, messages, token_limit)
                    result = future.result(timeout=60)  # 最长等待 60 秒
                    return result
            else:
                return loop.run_until_complete(
                    self._compressor.compress(messages, token_limit)
                )
        except Exception as e:
            logger.warning("上下文压缩失败，将回退到截断策略: %s", e)
            return None

    def _run_compress_sync(
        self,
        messages: list[dict[str, Any]],
        token_limit: int,
    ) -> list[dict[str, Any]]:
        """在新线程的事件循环中同步运行异步压缩。"""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._compressor.compress(messages, token_limit)
            )
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # 截断策略
    # ------------------------------------------------------------------

    def _truncate_messages(
        self,
        messages: list[dict[str, Any]],
        token_limit: int,
    ) -> list[dict[str, Any]]:
        """截断消息列表以适应 token 限制。

        Phase 4.7 增强策略：
        1. 保留第一条 system prompt
        2. 保留完整对话轮次（user + assistant + tool 消息组）
        3. 不截断 tool_calls 和对应的 tool 结果
        """
        if not messages:
            return []

        result = []
        system_msg = None

        # 抽出 system prompt
        if messages[0].get("role") == "system":
            system_msg = messages[0]
            messages = messages[1:]

        # ★ Phase 6+ 修复：使用与 _estimate_tokens 一致的动态系数
        #   替代原来硬编码的 token_limit * 2（第三套不一致的系数）
        chars_per_tok = estimate_chars_per_token(messages)
        remaining_chars = int(token_limit * chars_per_tok)
        if system_msg:
            remaining_chars -= len(str(system_msg.get("content", "")))

        # 按对话轮次分组（user + assistant + tools 为一轮）
        rounds = self._group_message_rounds(messages)

        # 从后向前保留完整的对话轮次
        for round_msgs in reversed(rounds):
            # 【修复】将 tool_calls 参数计入字符开销，与 _estimate_tokens 保持一致
            round_chars = 0
            for m in round_msgs:
                content = str(m.get("content", "") or "")
                round_chars += len(content)
                # 计入 tool_calls 参数的字符开销
                for tc in m.get("tool_calls", []):
                    if isinstance(tc, dict):
                        func = tc.get("function", {}) if isinstance(tc, dict) else {}
                        args = str(func.get("arguments", ""))
                        round_chars += len(func.get("name", "")) + len(args)
            if remaining_chars - round_chars < 0:
                break
            result = round_msgs + result
            remaining_chars -= round_chars

        # 插入 system prompt
        if system_msg:
            result.insert(0, system_msg)

        truncated_count = len(messages) - (len(result) - (1 if system_msg else 0))
        if truncated_count > 0:
            logger.info("截断了 %d 条旧消息以适应上下文窗口", truncated_count)

        return result

    def _validate_message_structure(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """验证并修复消息结构完整性。

        确保消息列表满足 OpenAI API 的要求：
        1. 以 system 或 user 消息开始
        2. 每个 assistant 的 tool_calls 都有对应的 tool 结果消息
        3. 每个 tool 结果消息都有对应的 assistant tool_call
        4. 跳过开头的孤立 tool 消息

        【v2.31.1 修复】补齐缺失 tool 结果的占位消息，防止
        "insufficient tool messages following tool_calls message" 错误。

        【v4.22.1 修复】增加全局 tool 结果搜索，解决截断后 tool 结果
        不在 assistant 紧邻后面（被其他消息间隔）导致的配对断裂问题。
        原实现仅收集 assistant 后面连续的 tool 消息，当截断或消息重排
        导致 tool 结果出现在非连续位置时，会错误地添加占位消息并遗留
        孤立 tool 消息，触发 API "tool without tool_calls" 错误。
        """
        if not messages:
            return messages

        # ---------- 第一遍：收集全局 tool_call_id 配对信息 ----------
        all_call_ids: set[str] = set()
        all_result_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    cid = tc.get("id", "") if isinstance(tc, dict) else ""
                    if cid:
                        all_call_ids.add(cid)
            elif msg.get("role") == "tool":
                cid = msg.get("tool_call_id", "")
                if cid:
                    all_result_ids.add(cid)

        # ★ v4.22.1: 预扫描 — 为每个 assistant 的 tool_calls 定位非连续位置的 tool 结果
        #   解决：截断后 tool 结果可能被 user 消息或其他 assistant 间隔，
        #   导致连续收集策略遗漏，错误添加占位消息并遗留孤立 tool 消息。
        consumed_ids: set[str] = set()
        extras_by_pos: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                tc_ids_this = set()
                for tc in msg.get("tool_calls", []):
                    cid = tc.get("id", "") if isinstance(tc, dict) else ""
                    if cid:
                        tc_ids_this.add(cid)
                if tc_ids_this:
                    extras: list[tuple[int, dict[str, Any]]] = []
                    for j in range(i + 1, len(messages)):
                        m2 = messages[j]
                        if m2.get("role") == "tool":
                            tid = m2.get("tool_call_id", "")
                            if tid in tc_ids_this and tid not in consumed_ids:
                                consumed_ids.add(tid)
                                extras.append((j, m2))
                        elif m2.get("role") == "assistant" and m2.get("tool_calls"):
                            break  # 不越过下一个有 tool_calls 的 assistant
                    if extras:
                        extras_by_pos[i] = extras

        # ---------- 第二遍：重建消息列表，修复不完整的配对 ----------
        result: list[dict[str, Any]] = []
        # 构建需跳过的 tool 消息索引集（已被预扫描消费，会在 assistant 后重新插入）
        skip_tool_indices: set[int] = set()
        for pos, extras_list in extras_by_pos.items():
            for idx, _ in extras_list:
                skip_tool_indices.add(idx)

        i = 0
        while i < len(messages):
            # 跳过已被预扫描消费的 tool 结果（它们会在 assistant 后面被重新插入）
            if i in skip_tool_indices:
                i += 1
                continue

            msg = messages[i]
            role = msg.get("role", "")

            # 跳过开头不是 system/user 的消息（孤立 tool 等）
            if not result and role not in ("system", "user"):
                i += 1
                continue

            # 处理 assistant 消息
            if role == "assistant":
                _assistant_pos = i  # 【v4.23.0】记录 assistant 真实位置
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    # 收集该 assistant 的 tool_call_ids
                    tc_ids: list[str] = []
                    for tc in tool_calls:
                        cid = tc.get("id", "") if isinstance(tc, dict) else ""
                        if cid:
                            tc_ids.append(cid)

                    if not tc_ids:
                        # 没有有效 tool_call_id，按普通消息处理
                        result.append(msg)
                        i += 1
                        continue

                    # 添加 assistant 消息
                    result.append(msg)
                    i += 1

                    # 收集后续连续匹配的 tool 消息（排除已被预扫描消费的）
                    found_ids: set[str] = set()
                    while i < len(messages):
                        if i in skip_tool_indices:
                            i += 1
                            continue
                        next_msg = messages[i]
                        if next_msg.get("role") == "tool":
                            nid = next_msg.get("tool_call_id", "")
                            if nid in tc_ids and nid not in consumed_ids:
                                result.append(next_msg)
                                found_ids.add(nid)
                                i += 1
                            elif nid and nid not in all_call_ids:
                                # 孤立的 tool 结果（没有匹配的 assistant），跳过
                                i += 1
                            else:
                                # 属于其他 assistant 的 tool 结果，停止收集
                                break
                        else:
                            break

                    # 【v4.26.0 统一策略】孤儿 tool_call 保留 + stub result（替代剥离）
                    # 必须在 extras 插入之前执行，确保 result[-1] 指向 assistant
                    # 但需预先考虑 extras 中将找到的 ID，避免误处理
                    _extras_ids: set[str] = set()
                    if _assistant_pos in extras_by_pos:
                        for _, _em in extras_by_pos[_assistant_pos]:
                            _eid = _em.get("tool_call_id", "")
                            if _eid:
                                _extras_ids.add(_eid)
                    missing_ids = [cid for cid in tc_ids if cid not in found_ids and cid not in _extras_ids]
                    if missing_ids:
                        # ★ Task 11: 保留孤儿 tool_call，添加 stub result
                        for _mid in missing_ids:
                            result.append({
                                "role": "tool",
                                "tool_call_id": _mid,
                                "content": "[该工具结果已在上下文处理中被移除]",
                            })
                        found_ids.update(missing_ids)
                        logger.info(
                            "补充 %d 个孤儿 tool_call 的 stub result: %s",
                            len(missing_ids), missing_ids,
                        )

                    # ★ v4.22.1: 插入非连续位置的 tool 结果（从预扫描中获取）
                    # 【v4.23.0 修复】使用 _assistant_pos 替代 i-1，避免 skip 导致的偏移
                    if _assistant_pos in extras_by_pos:
                        for _, extra_msg in extras_by_pos[_assistant_pos]:
                            result.append(extra_msg)
                            found_ids.add(extra_msg.get("tool_call_id", ""))
                else:
                    result.append(msg)
                    i += 1
            elif role == "tool":
                # 孤立的 tool 消息（前面没有 assistant），跳过
                i += 1
            else:
                result.append(msg)
                i += 1

        return result

    def _group_message_rounds(self, messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """将消息按对话轮次分组。

        一轮对话 = user消息 + (assistant消息 + tool_calls + tool结果)

        【修复】当用户插话导致 tool 结果跨轮次时，将 tool 结果合并到
        对应的 assistant 所在轮次，确保截断时 assistant + tool 结果作为整体保留。
        """
        # 1. 先收集所有 tool 结果（按 tool_call_id 索引）
        tool_results: dict[str, dict] = {}
        for msg in messages:
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id", "")
                if tid:
                    tool_results[tid] = msg

        # 2. 按轮次分组（增强版）
        rounds: list[list[dict[str, Any]]] = []
        current_round: list[dict[str, Any]] = []
        # 记录已预收集到轮次中的 tool 结果，避免重复添加
        collected_tool_ids: set[str] = set()

        for msg in messages:
            role = msg.get("role", "")

            if role == "user":
                # 新的一轮开始
                if current_round:
                    rounds.append(current_round)
                current_round = [msg]
            elif role == "assistant":
                current_round.append(msg)
                # 【修复】预收集该 assistant 的所有 tool 结果到同一轮
                for tc in msg.get("tool_calls", []):
                    if isinstance(tc, dict):
                        tid = tc.get("id", "")
                        if tid and tid in tool_results and tid not in collected_tool_ids:
                            current_round.append(tool_results[tid])
                            collected_tool_ids.add(tid)
            elif role == "tool":
                tid = msg.get("tool_call_id", "")
                # 如果 tool 结果已作为预收集加入，则跳过
                if tid not in collected_tool_ids:
                    current_round.append(msg)
            else:
                # 其他消息加入当前轮
                current_round.append(msg)

        # 最后一轮
        if current_round:
            rounds.append(current_round)

        return rounds

    def _enforce_message_limit(self, session: Session) -> None:
        """强制执行消息数量限制，保留完整的对话轮次。"""
        if len(session.messages) <= self._max_message_count:
            return

        # 保留 system prompt
        system_msg = None
        messages = session.messages
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            messages = messages[1:]

        # 按轮次分组
        rounds = self._group_message_rounds(messages)

        # 计算需要保留的轮次数
        target_count = self._max_message_count - (1 if system_msg else 0)
        kept_rounds = []
        current_count = 0

        # 从后向前保留轮次
        for round_msgs in reversed(rounds):
            if current_count + len(round_msgs) > target_count:
                break
            kept_rounds.insert(0, round_msgs)
            current_count += len(round_msgs)

        # 重建消息列表
        new_messages: list[dict[str, Any]] = []
        if system_msg:
            new_messages.append(system_msg)
        for round_msgs in kept_rounds:
            new_messages.extend(round_msgs)

        # 【v2.31.1 修复】截断后验证消息结构完整性，防止 tool_calls 配对断裂
        # 【增强】使用 _validate_message_structure 确保配对完整
        validated_messages = self._validate_message_structure(new_messages)
        session.messages = validated_messages
        # ★ Phase 6+ 清除压缩缓存（消息列表已被替换）
        session._compressed_cache = None
        session._cache_key = (0, 0, 0)
        logger.info("会话 %s 消息数量超限，已截断至 %d 条", session.id, len(new_messages))

    # ------------------------------------------------------------------
    # 持久化相关（Phase 4.4）
    # ------------------------------------------------------------------

    async def load_history(self, limit: int = 10) -> list[Session]:
        """从存储加载最近的会话历史。

        只加载元数据，不加载消息内容。

        Args:
            limit: 加载的会话数量

        Returns:
            加载的会话列表
        """
        if not self._storage:
            return []

        from src.core.storage import StoredSession

        stored_sessions = await self._storage.list_sessions(limit=limit)
        loaded = []

        for stored in stored_sessions:
            # 如果已经在内存中，跳过
            if stored.id in self._sessions:
                continue

            # 创建 Session 对象（不加载消息）
            session = Session(
                id=stored.id,
                title=stored.title,
                model_key=stored.model_key,
                created_at=stored.created_at,
                messages=[],  # 消息在需要时按需加载
                total_tokens=stored.total_tokens,
                metadata=stored.metadata,
            )
            self._sessions[stored.id] = session
            loaded.append(session)

        if loaded:
            logger.info("从存储加载了 %d 个历史会话", len(loaded))

        return loaded

    async def load_session_messages(self, session_id: str) -> None:
        """从存储加载指定会话的消息。"""
        if not self._storage:
            return

        session = self._get_session(session_id)
        if session.messages:  # 已经加载过
            return

        stored_msgs = await self._storage.load_messages(session_id)
        for stored in stored_msgs:
            msg = stored.to_dict()
            # 跳过 system prompt（已有）
            if msg.get("role") == "system" and session.has_system_prompt:
                continue
            session.messages.append(msg)

        logger.info("加载会话 %s 的 %d 条消息", session_id, len(stored_msgs))

    def generate_title(self, session_id: str = "") -> str:
        """根据首条用户消息生成会话标题。

        规则：取用户首条消息前 20 字符 + "..."
        """
        session = self._get_session(session_id)

        # 查找第一条用户消息
        for msg in session.messages:
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                if len(content) <= 20:
                    title = content
                else:
                    title = content[:20] + "..."
                # 更新标题
                session.title = title
                # 异步更新存储
                if self._storage:
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(
                                self._storage.update_session_title(session.id, title)
                            )
                        else:
                            loop.run_until_complete(
                                self._storage.update_session_title(session.id, title)
                            )
                    except Exception as e:
                        logger.warning("更新会话标题失败: %s", e)
                return title

        return session.title

    async def export_session(self, session_id: str = "", format: str = "markdown") -> str:
        """导出会话内容。"""
        if not self._storage:
            # 从内存导出
            session = self._get_session(session_id)
            lines = [
                f"# {session.title}",
                "",
                f"> 创建时间: {session.created_at.strftime('%Y-%m-%d %H:%M')}",
                "",
                "---",
                "",
            ]
            for msg in session.messages:
                role_label = {
                    "system": "⚙️ System",
                    "user": "👤 User",
                    "assistant": "🤖 Assistant",
                    "tool": "🔧 Tool",
                }.get(msg.get("role", ""), msg.get("role", ""))
                lines.append(f"### {role_label}")
                lines.append("")
                lines.append(str(msg.get("content", "")))
                lines.append("")
            return "\n".join(lines)

        sid = session_id or self._current_id
        return await self._storage.export_session(sid, format)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_session(self, session_id: str = "") -> Session:
        """获取指定会话或当前会话。"""
        sid = session_id or self._current_id
        if sid not in self._sessions:
            raise ValueError(f"会话不存在: {sid}")
        return self._sessions[sid]

    def _cancel_pending_task(self, session_id: str) -> None:
        """Task 10: 取消指定会话的待处理异步压缩 Task。"""
        task = self._pending_summary_tasks.pop(session_id, None)
        if task is not None:
            try:
                import asyncio
                if isinstance(task, asyncio.Task) and not task.done():
                    task.cancel()
                    logger.debug("取消待处理异步压缩 Task: %s", session_id)
            except Exception:
                pass

    def _check_async_result(
        self,
        session_id: str,
        current_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Task 10: 检查已完成的异步压缩 Task 结果。

        快照 hash 保护：比较当前消息与 Task 启动时的快照，
        不一致则丢弃结果。

        Returns:
            压缩后的消息列表，无可用结果时返回 None
        """
        import asyncio
        import hashlib

        task = self._pending_summary_tasks.pop(session_id, None)
        if task is None:
            return None

        if not isinstance(task, asyncio.Task):
            return None

        if not task.done():
            # 还没完成，放回去
            self._pending_summary_tasks[session_id] = task
            return None

        try:
            result = task.result()
        except Exception:
            return None

        if result is None:
            return None

        # 快照 hash 验证
        snapshot = getattr(task, '_snapshot', None)
        if snapshot:
            msg_count = len(current_messages)
            msg_hash = hashlib.md5(
                str([(m.get("role"), str(m.get("content", ""))[:100]) for m in current_messages]).encode()
            ).hexdigest()[:12]
            if snapshot != (msg_count, msg_hash):
                logger.debug("异步压缩结果快照不匹配，丢弃")
                return None

        logger.info("使用异步压缩结果 (session=%s)", session_id)
        return result

    def _schedule_async_compress(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        token_limit: int,
    ) -> None:
        """Task 10: 调度异步后台压缩 Task。

        当前轮使用截断结果（不阻塞），下一轮 get_messages 时
        检查 Task 是否完成，完成则使用摘要结果。
        """
        if not self._compressor:
            return

        # 取消已有的待处理 Task
        self._cancel_pending_task(session_id)

        try:
            import asyncio
            import hashlib

            # 快照 hash 保护
            msg_count = len(messages)
            msg_hash = hashlib.md5(
                str([(m.get("role"), str(m.get("content", ""))[:100]) for m in messages]).encode()
            ).hexdigest()[:12]
            snapshot = (msg_count, msg_hash)

            async def _run():
                return await self._compressor.compress_async(messages, token_limit)

            loop = asyncio.get_event_loop()
            task = loop.create_task(_run())
            task._snapshot = snapshot  # type: ignore[attr-defined]
            task._target_session = session_id  # type: ignore[attr-defined]

            def _on_done(t: asyncio.Task) -> None:
                try:
                    result = t.result()
                    if result is not None:
                        logger.info("异步压缩完成: %s", session_id)
                    else:
                        logger.debug("异步压缩返回 None: %s", session_id)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning("异步压缩 Task 异常: %s", e)

            task.add_done_callback(_on_done)
            self._pending_summary_tasks[session_id] = task
            logger.debug("已调度异步压缩 Task: %s (snapshot=%s)", session_id, snapshot)
        except Exception as e:
            logger.warning("调度异步压缩失败: %s", e)

    def _cleanup_oldest(self) -> None:
        """清理最旧的会话以保持在限制内。"""
        sessions = sorted(self._sessions.values(), key=lambda s: s.created_at)
        while len(self._sessions) > self._max_sessions:
            oldest = sessions.pop(0)
            if oldest.id != self._current_id:
                del self._sessions[oldest.id]
                logger.info("自动清理旧会话: %s", oldest.id)
