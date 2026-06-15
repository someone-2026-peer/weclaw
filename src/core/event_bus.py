"""异步事件总线 — 核心的发布-订阅消息系统。

提供模块间松耦合通信。Agent 推理的每一步都通过事件总线广播，
UI / 审计日志 / 其他模块通过订阅事件来响应。

特性：
- 基于 asyncio，支持异步回调
- 支持订阅优先级（数值越小越先调用）
- 支持一次性订阅（once=True）
- 支持通配符订阅（"*" 接收所有事件）
- 线程安全的同步发布方法 emit_sync()
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# 事件回调类型：同步或异步函数，接受 event_type 和 data
EventCallback = Callable[..., Any]


@dataclass(order=True)
class _Subscription:
    """内部订阅记录（按 priority 排序）。"""

    priority: int
    callback: EventCallback = field(compare=False)
    once: bool = field(default=False, compare=False)
    _id: int = field(default=0, compare=False)


class EventBus:
    """异步事件总线。

    用法::

        bus = EventBus()

        async def on_tool_call(event_type, data):
            print(f"工具被调用: {data}")

        bus.on("tool_call", on_tool_call)
        await bus.emit("tool_call", {"tool": "shell", "action": "run"})
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[_Subscription]] = defaultdict(list)
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # 订阅
    # ------------------------------------------------------------------

    def on(
        self,
        event_type: str,
        callback: EventCallback,
        priority: int = 100,
        once: bool = False,
    ) -> int:
        """订阅事件。

        Args:
            event_type: 事件类型名称，或 "*" 订阅所有事件
            callback: 回调函数（同步或异步）
            priority: 优先级（数值越小越先调用，默认100）
            once: 是否只触发一次

        Returns:
            订阅 ID，可用于取消订阅
        """
        self._next_id += 1
        sub = _Subscription(
            priority=priority,
            callback=callback,
            once=once,
            _id=self._next_id,
        )
        self._subscribers[event_type].append(sub)
        # 保持按优先级排序
        self._subscribers[event_type].sort()
        logger.debug("订阅事件 '%s' (id=%d, priority=%d)", event_type, sub._id, priority)
        return sub._id

    def once(self, event_type: str, callback: EventCallback, priority: int = 100) -> int:
        """订阅事件（只触发一次）。"""
        return self.on(event_type, callback, priority=priority, once=True)

    def off(self, event_type: str, subscription_id: int) -> bool:
        """取消订阅。

        Args:
            event_type: 事件类型
            subscription_id: 订阅 ID

        Returns:
            是否成功取消
        """
        subs = self._subscribers.get(event_type, [])
        for i, sub in enumerate(subs):
            if sub._id == subscription_id:
                subs.pop(i)
                logger.debug("取消订阅 '%s' (id=%d)", event_type, subscription_id)
                return True
        return False

    def off_all(self, event_type: str = "") -> int:
        """取消所有订阅。

        Args:
            event_type: 事件类型。为空则清除所有事件的订阅。

        Returns:
            取消的订阅数量
        """
        if event_type:
            count = len(self._subscribers.get(event_type, []))
            self._subscribers[event_type] = []
            return count
        else:
            count = sum(len(subs) for subs in self._subscribers.values())
            self._subscribers.clear()
            return count

    # ------------------------------------------------------------------
    # 发布
    # ------------------------------------------------------------------

    async def emit(self, event_type: str, data: Any = None) -> int:
        """异步发布事件。

        按优先级顺序调用所有订阅者。同时也触发通配符 "*" 的订阅者。

        Args:
            event_type: 事件类型
            data: 事件数据

        Returns:
            被通知的订阅者数量
        """
        notified = 0
        to_remove: list[tuple[str, int]] = []

        # 收集匹配的订阅者：精确匹配 + 通配符
        all_subs: list[tuple[str, _Subscription]] = []
        for et in [event_type, "*"]:
            for sub in self._subscribers.get(et, []):
                all_subs.append((et, sub))

        # 按优先级排序
        all_subs.sort(key=lambda x: x[1].priority)

        for et, sub in all_subs:
            try:
                result = sub.callback(event_type, data)
                # 如果回调是协程，await 它
                if asyncio.iscoroutine(result):
                    await result
                notified += 1
            except Exception as e:
                logger.error("事件回调异常 '%s' → %s: %s", event_type, sub.callback, e)

            if sub.once:
                to_remove.append((et, sub._id))

        # 清理一次性订阅
        for et, sid in to_remove:
            self.off(et, sid)

        logger.debug("事件 '%s' 已通知 %d 个订阅者", event_type, notified)
        return notified

    def emit_sync(self, event_type: str, data: Any = None) -> None:
        """同步发布事件（在已有事件循环中调度异步 emit）。

        适用于在同步代码中触发事件。
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(event_type, data))
        except RuntimeError:
            # 没有运行中的事件循环，直接运行
            asyncio.run(self.emit(event_type, data))

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def subscriber_count(self, event_type: str = "") -> int:
        """获取订阅者数量。"""
        if event_type:
            return len(self._subscribers.get(event_type, []))
        return sum(len(subs) for subs in self._subscribers.values())

    def event_types(self) -> list[str]:
        """获取所有已注册的事件类型。"""
        return [et for et, subs in self._subscribers.items() if subs]
