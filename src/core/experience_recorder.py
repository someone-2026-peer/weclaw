"""ExperienceAutoRecorder — Phase 2：日常工具调用经验自动采集中间件。

通过 EventBus 订阅 TOOL_CALL + TOOL_RESULT 事件对，
自动检测"工具连续失败 → 重试成功"模式，生成经验记录。

设计：
- 同一 session_id 内，同一 tool_name 连续失败 >= 2 次后成功 = 一次试错经验
- 仅在相邻时间窗口内匹配，不跨 session
- 轻量 prompt 模板生成 trigger + diagnosis + fix_summary + abstract_pattern
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import EventType, ToolCallEvent, ToolResultEvent

logger = logging.getLogger(__name__)

# 触发阈值：连续失败次数达到此值后成功才记录经验
_MIN_FAILURES_BEFORE_SUCCESS = 2

# 时间窗口：超过此时间窗口的失败不计入连续失败（秒）
_TIME_WINDOW_SECONDS = 300  # 5 分钟

# 单次会话内同一工具的最大跟踪失败数（避免无限累积）
_MAX_TRACKED_FAILURES = 10


class _ToolFailureTracker:
    """跟踪单个工具的连续失败状态。"""

    def __init__(self, tool_name: str, session_id: str):
        self.tool_name = tool_name
        self.session_id = session_id
        self.failures: list[dict[str, Any]] = []  # [{error, timestamp, action_name, arguments}]
        self.last_failure_time: float = 0.0

    def add_failure(self, error: str, action_name: str, arguments: dict) -> None:
        now = datetime.now().timestamp()
        # 如果距离上次失败超过时间窗口，重置
        if self.last_failure_time and (now - self.last_failure_time) > _TIME_WINDOW_SECONDS:
            self.failures.clear()
        self.failures.append({
            "error": error[:500],
            "timestamp": now,
            "action_name": action_name,
            "arguments": {k: str(v)[:100] for k, v in (arguments or {}).items()},
        })
        self.last_failure_time = now
        # 限制累积
        if len(self.failures) > _MAX_TRACKED_FAILURES:
            self.failures = self.failures[-_MAX_TRACKED_FAILURES:]

    def check_and_reset(self) -> list[dict[str, Any]] | None:
        """检查是否达到触发阈值。达到则返回失败列表并重置。"""
        if len(self.failures) >= _MIN_FAILURES_BEFORE_SUCCESS:
            result = list(self.failures)
            self.failures.clear()
            return result
        return None

    def reset(self) -> None:
        self.failures.clear()


class ExperienceAutoRecorder:
    """日常工具调用经验自动采集器。"""

    def __init__(self, event_bus: EventBus, experience_store):
        """初始化并订阅事件。

        Args:
            event_bus: 事件总线
            experience_store: ExperienceStore 实例
        """
        self._store = experience_store
        self._event_bus = event_bus

        # key: f"{session_id}:{tool_name}"
        self._trackers: dict[str, _ToolFailureTracker] = defaultdict(
            lambda: _ToolFailureTracker("", "")
        )

        # 订阅事件（与 AuditLogger 同优先级，不影响主流程）
        event_bus.on(EventType.TOOL_RESULT, self._on_tool_result, priority=60)
        logger.info("ExperienceAutoRecorder 已连接到 EventBus")

    def _tracker_key(self, session_id: str, tool_name: str) -> str:
        return f"{session_id}:{tool_name}"

    def _on_tool_result(self, event_type: str, data: Any) -> None:
        """处理工具结果事件。"""
        if isinstance(data, ToolResultEvent):
            tool_name = data.tool_name
            status = data.status
            error = data.error
            action_name = data.action_name
            session_id = data.session_id
        elif isinstance(data, dict):
            tool_name = data.get("tool_name", "")
            status = data.get("status", "")
            error = data.get("error", "")
            action_name = data.get("action_name", "")
            session_id = data.get("session_id", "")
        else:
            return

        if not tool_name or not session_id:
            return

        key = self._tracker_key(session_id, tool_name)

        if status == "error":
            # 记录失败
            if key not in self._trackers:
                self._trackers[key] = _ToolFailureTracker(tool_name, session_id)
            self._trackers[key].add_failure(
                error=error,
                action_name=action_name,
                arguments=data.arguments if isinstance(data, ToolCallEvent) else {},
            )

        elif status == "success":
            # 检查是否有连续失败历史
            tracker = self._trackers.get(key)
            if tracker:
                failures = tracker.check_and_reset()
                if failures:
                    # 触发自动经验记录
                    asyncio.create_task(
                        self._generate_and_record(
                            tool_name=tool_name,
                            action_name=action_name,
                            session_id=session_id,
                            failures=failures,
                        )
                    )

    async def _generate_and_record(
        self,
        tool_name: str,
        action_name: str,
        session_id: str,
        failures: list[dict[str, Any]],
    ) -> None:
        """从试错事件对生成经验并记录。"""
        try:
            # 构建 trigger
            failure_count = len(failures)
            trigger = f"{tool_name}.{action_name} 连续失败 {failure_count} 次后重试成功"

            # 构建 diagnosis（从失败错误中提取关键信息）
            error_samples = [f["error"][:100] for f in failures[:3]]
            diagnosis = f"工具 {tool_name} 调用出错: {'; '.join(error_samples)}"

            # 构建 fix_summary
            fix_summary = f"经过 {failure_count} 次重试后成功执行 {tool_name}.{action_name}"

            # 构建 abstract_pattern
            abstract_pattern = self._generate_pattern(tool_name, failures)

            await self._store.record(
                trigger=trigger,
                diagnosis=diagnosis,
                fix_summary=fix_summary,
                abstract_pattern=abstract_pattern,
                outcome="success",
                source_type="tool_retry",
                session_id=session_id,
                tools=[tool_name],
            )
            logger.info(
                "自动记录工具试错经验: %s (session=%s, failures=%d)",
                tool_name, session_id[:8], failure_count,
            )
        except Exception as e:
            logger.warning("自动经验记录失败: %s", e)

    @staticmethod
    def _generate_pattern(tool_name: str, failures: list[dict[str, Any]]) -> str:
        """从失败信息生成抽象模式。"""
        # 提取常见错误类型
        error_texts = " ".join(f["error"].lower() for f in failures[:3])

        patterns = []

        # 超时类
        if any(kw in error_texts for kw in ["timeout", "超时", "timed out"]):
            patterns.append("超时重试")

        # 连接/网络类
        if any(kw in error_texts for kw in ["connection", "connect", "连接", "network", "网络"]):
            patterns.append("连接重试")

        # 参数错误类
        if any(kw in error_texts for kw in ["invalid", "参数", "argument", "parameter"]):
            patterns.append("参数修正")

        # 权限类
        if any(kw in error_texts for kw in ["permission", "denied", "权限", "拒绝"]):
            patterns.append("权限问题")

        # 编码/解析类
        if any(kw in error_texts for kw in ["encoding", "parse", "decode", "编码", "解析"]):
            patterns.append("编码/解析容错")

        if not patterns:
            patterns.append(f"{tool_name}工具重试成功")

        return f"工具试错模式: {'+'.join(patterns)}"

    def disconnect(self) -> None:
        """清理跟踪器。"""
        self._trackers.clear()
        logger.debug("ExperienceAutoRecorder 已断开")
