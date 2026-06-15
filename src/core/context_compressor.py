"""上下文压缩器 — 对话历史智能压缩，减少 token 消耗。

借鉴 Hermes Agent 的 ContextCompressor 设计，针对 WeClaw 场景简化实现：
- 对话历史超过 token 阈值时触发压缩
- 保留最近 N 轮原始消息不压缩
- 使用辅助 LLM 对早期消息生成结构化摘要
- 工具调用结果进行结构化压缩（保留关键输出，移除冗余详情）
- 摘要消息以特殊格式标记，便于识别
- 压缩失败时回退到原有截断策略
"""
#
# SPDX-License-Identifier: MIT
# Architecture reference: NousResearch/hermes-agent (MIT License)
# Repository: https://github.com/NousResearch/hermes-agent
#

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

from src.core.token_utils import estimate_tokens as _unified_estimate_tokens
from src.core.redact import redact_sensitive_text

if TYPE_CHECKING:
    from src.core.auxiliary_client import AuxiliaryBudgetExceeded, AuxiliaryClient
    from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

# 摘要消息前缀标记（v4.26.0 双语约束：中文为主，英文兜底）
SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY / 上下文压缩 — 仅供参照] "
    "Earlier turns were compacted into the summary below. "
    "Treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Your current task is in the 'Active Task' section — resume from there. "
    "Respond ONLY to the latest user message that appears AFTER this summary."
)

# 压缩发生时追加到 system_prompt 末尾的注解
_SYSTEM_PROMPT_ANNOTATION = (
    "\n\n[Note: Some earlier turns were compacted into a handoff summary "
    "to preserve context space.]"
)

# 默认参数
# 【v2.31.1 提升】适配 DeepSeek 1M 等大上下文模型，减少激进压缩
DEFAULT_TOKEN_THRESHOLD = 32000      # 触发压缩的 token 阈值（原 8000）
DEFAULT_PROTECT_RECENT_ROUNDS = 12   # 保留最近 N 轮不压缩（原 5）
DEFAULT_SUMMARY_MAX_TOKENS = 3000    # 摘要最大输出 token（原 1500，上一轮提至 2000）
TOOL_CONTENT_TRUNCATE_LIMIT = 2000   # 工具结果内容截断阈值（字符数，原 300，上一轮提至 800）

# 字符数/token 估算系数
CHARS_PER_TOKEN = 3  # 中英文混合场景下的平均估算


@dataclass
class CompressionStats:
    """压缩统计信息。"""

    original_tokens: int = 0       # 压缩前估算 token 数
    compressed_tokens: int = 0     # 压缩后估算 token 数
    original_messages: int = 0     # 压缩前消息数
    compressed_messages: int = 0   # 压缩后消息数
    summarized_rounds: int = 0     # 被摘要的对话轮次数
    compression_ratio: float = 0.0 # 压缩比（越小越好）
    duration_ms: float = 0.0       # 压缩耗时（毫秒）
    used_fallback: bool = False    # 是否回退到截断策略


class ContextCompressor:
    """上下文压缩器。

    核心方法：``compress()`` — 检查消息列表是否超过 token 阈值，
    如超过则使用辅助 LLM 对早期消息生成摘要，替换原始消息。

    用法::

        compressor = ContextCompressor(auxiliary_client)
        compressed_messages = await compressor.compress(messages, max_tokens=8000)
    """

    def __init__(
        self,
        auxiliary_client: AuxiliaryClient | None = None,
        token_threshold: int = DEFAULT_TOKEN_THRESHOLD,
        protect_recent_rounds: int = DEFAULT_PROTECT_RECENT_ROUNDS,
        summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
        model_registry: ModelRegistry | None = None,
        main_model_key: str = "",
        cache_aware_compression: bool = False,
    ):
        """
        Args:
            auxiliary_client: 辅助 LLM 客户端实例（为 None 则只做工具结果裁剪，不做LLM摘要）
            token_threshold: 触发压缩的 token 阈值
            protect_recent_rounds: 保留最近 N 轮对话不压缩
            summary_max_tokens: 摘要输出的最大 token 数
            model_registry: 模型注册表（用于主模型重试 fallback）
            main_model_key: 主模型标识（用于主模型重试 fallback）
            cache_aware_compression: v4.27.0 缓存感知压缩开关（当 DeepSeek 缓存命中成本 < 压缩成本时延迟压缩）
        """
        self._client = auxiliary_client
        self._token_threshold = token_threshold
        self._protect_recent_rounds = protect_recent_rounds
        self._summary_max_tokens = summary_max_tokens
        # ★ Task 1: 主模型重试依赖注入
        self._model_registry = model_registry
        self._main_model_key = main_model_key
        # ★ Task 5: token-budget 尾部保护（基于 limit 计算，而非 threshold）
        self._tail_token_budget: int = 0  # 由 set_token_threshold 或 compress() 中动态设置
        # 迭代摘要：保存上一次压缩的摘要，用于增量更新
        self._previous_summary: str | None = None
        # 压缩计数
        self._compression_count: int = 0
        # 最近一次统计
        self._last_stats: CompressionStats | None = None
        # 摘要失败冷却（防止频繁重试）
        self._failure_cooldown_until: float = 0.0
        # 【v4.23.0】反抖动保护：连续无效压缩时跳过（借鉴 Hermes _ineffective_compression_count）
        self._ineffective_compression_count: int = 0
        self._last_compression_savings_pct: float = 0.0
        # ★ v4.27.0 缓存感知压缩：当 DeepSeek 缓存命中成本 < 压缩成本时延迟压缩
        self._cache_aware_compression: bool = cache_aware_compression

        logger.info(
            "上下文压缩器初始化: threshold=%d tokens, protect_rounds=%d, "
            "has_auxiliary=%s, has_main_model_fallback=%s",
            token_threshold, protect_recent_rounds,
            "是" if auxiliary_client else "否",
            "是" if model_registry and main_model_key else "否",
        )

    @property
    def last_stats(self) -> CompressionStats | None:
        """最近一次压缩的统计信息。"""
        return self._last_stats

    @property
    def compression_count(self) -> int:
        """累计压缩次数。"""
        return self._compression_count

    # ------------------------------------------------------------------
    # v4.27.0 缓存感知压缩决策
    # ------------------------------------------------------------------

    def _should_skip_for_cache(self, token_count: int, threshold: int) -> bool:
        """当 DeepSeek 缓存命中成本 < 压缩成本时，延迟压缩。

        DeepSeek 缓存命中价 = 正常价 1/10（v4-flash: 0.1元 vs 1元/百万token）。
        当缓存命中率 >60% 时，保留完整前缀（走缓存）的成本低于调用辅助模型压缩。

        仅当 cache_aware_compression 配置启用时生效。

        Args:
            token_count: 当前估算 token 数
            threshold: 压缩阈值

        Returns:
            True 表示应跳过压缩
        """
        if not self._cache_aware_compression:
            return False

        # 超限太多则无论如何都要压缩（超过阈值的 1.5 倍）
        if token_count > threshold * 1.5:
            return False

        # DeepSeek v4-flash 缓存命中价: 0.1元/百万token
        _CACHE_HIT_PRICE = 0.1  # 元/百万token
        # DeepSeek v4-flash 正常价: 1元/百万token
        _NORMAL_PRICE = 1.0  # 元/百万token
        # 压缩额外开销: 辅助模型调用固定成本
        _COMPRESS_FIXED_COST = 0.002  # 元

        # 假设缓存命中率 70%（保守估计）
        _ASSUMED_HIT_RATE = 0.7

        cache_cost_per_m = _CACHE_HIT_PRICE * _ASSUMED_HIT_RATE + _NORMAL_PRICE * (1 - _ASSUMED_HIT_RATE)
        expected_cache_cost = token_count * cache_cost_per_m / 1_000_000
        compress_cost = token_count * _NORMAL_PRICE / 1_000_000 + _COMPRESS_FIXED_COST

        return expected_cache_cost < compress_cost

    # ------------------------------------------------------------------
    # 核心压缩入口
    # ------------------------------------------------------------------

    async def compress(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 0,
        focus_topic: str = "",
    ) -> list[dict[str, Any]]:
        """压缩消息列表以适应 token 限制。

        算法（v4.26.0 重写）：
        1. 估算当前 token 数，未超阈值则直接返回
        2. 分离 system prompt 和对话消息
        3. 使用 token-budget 尾部保护确定分割点（替代固定轮次）
        4. 用户消息锚定（确保最后一条 user 消息在尾部）
        5. 对早期消息进行工具结果裁剪
        6. 三级容灾摘要：辅助 LLM → 主模型重试 → 静态 fallback
        7. 组装压缩后的消息列表

        Args:
            messages: 原始消息列表
            max_tokens: token 上限（0=使用默认阈值）
            focus_topic: 可选的焦点主题引导

        Returns:
            压缩后的消息列表
        """
        start_time = time.monotonic()
        threshold = max_tokens or self._token_threshold

        # 估算当前 token 数
        original_tokens = self.estimate_tokens(messages)

        # 未超阈值，无需压缩
        if original_tokens <= threshold:
            return messages

        # ★ v4.27.0: 缓存感知 - 如果缓存成本更低，允许适度超限
        if self._should_skip_for_cache(original_tokens, threshold):
            logger.info(
                "缓存感知压缩: 跳过压缩 (缓存成本 < 压缩成本, tokens=%d)",
                original_tokens,
            )
            return messages

        logger.info(
            "触发上下文压缩: 估算 %d tokens > 阈值 %d tokens (%d 条消息)",
            original_tokens, threshold, len(messages),
        )

        stats = CompressionStats(
            original_tokens=original_tokens,
            original_messages=len(messages),
        )

        # 分离 system prompt
        system_msg = None
        work_messages = list(messages)
        if work_messages and work_messages[0].get("role") == "system":
            system_msg = work_messages[0]
            work_messages = work_messages[1:]

        # ★ Task 5: 使用 token-budget 尾部保护确定分割点
        # 动态计算 tail_budget（基于 limit/threshold，而非固定轮次）
        if self._tail_token_budget <= 0:
            self._tail_token_budget = int(threshold * 0.15)

        head_end = 0  # 头部保护区域（system prompt 已分离，从 0 开始）
        cut_idx = self._find_tail_cut_by_tokens(
            work_messages, head_end, self._tail_token_budget,
        )

        # ★ Task 6: 用户消息锚定
        cut_idx = self._ensure_last_user_in_tail(work_messages, cut_idx, head_end)

        # 安全检查：确保有足够的消息可以压缩
        if cut_idx <= head_end or cut_idx >= len(work_messages):
            logger.info("消息不足，无法压缩 (cut_idx=%d, total=%d)", cut_idx, len(work_messages))
            return messages

        early_messages = work_messages[:cut_idx]
        tail_messages = work_messages[cut_idx:]
        stats.summarized_rounds = len(self._group_rounds(early_messages))

        # Phase 1: 工具结果裁剪（无需 LLM，降低后续摘要输入成本）
        pruned_early = self._prune_tool_results(early_messages)

        # Phase 2: 三级容灾摘要生成
        summary = None
        _summary_source = "none"

        # Level 1: 辅助 LLM
        if self._client and not self._is_in_cooldown():
            try:
                summary = await self._generate_summary(pruned_early, focus_topic=focus_topic)
                if summary:
                    _summary_source = "auxiliary"
            except Exception as e:
                is_budget_exceeded = (
                    type(e).__name__ == "AuxiliaryBudgetExceeded"
                )
                if is_budget_exceeded:
                    now = datetime.now()
                    end_of_day = datetime(now.year, now.month, now.day, 23, 59, 59)
                    cooldown_secs = max(60, (end_of_day - now).total_seconds())
                    self._failure_cooldown_until = time.monotonic() + cooldown_secs
                    logger.warning(
                        "辅助模型预算耗尽，尝试主模型重试 (%.0f 秒冷却)",
                        cooldown_secs,
                    )
                else:
                    logger.warning("辅助模型摘要失败，尝试主模型重试: %s", e)

        # Level 2: 主模型重试
        if not summary:
            summary = await self._try_main_model_summary(pruned_early, focus_topic=focus_topic)
            if summary:
                _summary_source = "main_model"

        # Level 3: 静态 fallback
        if not summary:
            summary = self._generate_static_fallback(early_messages, len(early_messages))
            _summary_source = "static"
            stats.used_fallback = True
            # 主模型重试也失败后才进入冷却
            self._failure_cooldown_until = time.monotonic() + 60
            logger.info("使用静态 fallback 摘要")

        # Phase 3: 组装压缩后的消息列表
        compressed = []

        # 添加 system prompt（★ Task 9: 压缩注解）
        if system_msg:
            sp = dict(system_msg)  # 浅拷贝
            if summary:  # 压缩成功，追加注解
                sp_content = str(sp.get("content", ""))
                if _SYSTEM_PROMPT_ANNOTATION not in sp_content:
                    sp["content"] = sp_content + _SYSTEM_PROMPT_ANNOTATION
            compressed.append(sp)

        # 添加摘要消息
        if summary:
            summary_role = self._choose_summary_role(
                system_msg, [tail_messages] if tail_messages else [],
            )
            compressed.append({
                "role": summary_role,
                "content": f"{SUMMARY_PREFIX}\n{summary}",
            })

        # 添加尾部消息
        compressed.extend(tail_messages)

        # 验证消息结构完整性
        compressed = self._validate_tool_pairs(compressed)

        # 统计
        stats.compressed_tokens = self.estimate_tokens(compressed)
        stats.compressed_messages = len(compressed)
        stats.compression_ratio = (
            stats.compressed_tokens / stats.original_tokens
            if stats.original_tokens > 0 else 1.0
        )
        stats.duration_ms = (time.monotonic() - start_time) * 1000
        self._last_stats = stats
        self._compression_count += 1

        logger.info(
            "压缩完成 #%d: %d -> %d 条消息, %d -> %d tokens (压缩比 %.1f%%), "
            "耗时 %.0fms, 摘要来源=%s",
            self._compression_count,
            stats.original_messages, stats.compressed_messages,
            stats.original_tokens, stats.compressed_tokens,
            stats.compression_ratio * 100,
            stats.duration_ms,
            _summary_source,
        )

        # ★ v4.23.0 反抖动：更新压缩节省百分比和连续无效计数
        savings_pct = (
            1.0 - (stats.compressed_tokens / stats.original_tokens)
            if stats.original_tokens > 0 else 0.0
        )
        self._last_compression_savings_pct = savings_pct
        if savings_pct < 0.1:
            self._ineffective_compression_count += 1
            logger.debug(
                "反抖动：压缩节省 %.1f%% < 10%%，连续无效计数=%d",
                savings_pct * 100, self._ineffective_compression_count,
            )
        else:
            self._ineffective_compression_count = 0

        return compressed

    async def compress_async(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 0,
    ) -> list[dict[str, Any]] | None:
        """Task 10: 异步后台压缩入口。

        由 SessionManager 在后台 Task 中调用，不阻塞主对话循环。
        返回压缩结果或 None（失败时）。

        Returns:
            压缩后的消息列表，失败时返回 None
        """
        try:
            result = await self.compress(messages, max_tokens)
            return result
        except Exception as e:
            logger.warning("异步压缩失败: %s", e)
            return None

    def set_token_threshold(self, threshold: int) -> None:
        """动态更新压缩触发阈值。

        Phase 6+ 新增：供 SessionManager.set_compression_threshold() 调用，
        模型切换时同步更新压缩阈值。

        v4.26.0 增强：同步更新 _tail_token_budget。

        Args:
            threshold: 新的压缩触发阈值（token 数）
        """
        if threshold > 0 and threshold != self._token_threshold:
            old = self._token_threshold
            self._token_threshold = threshold
            # ★ Task 5: 同步更新尾部 token 预算
            self._tail_token_budget = int(threshold * 0.15)
            logger.info(
                "压缩阈值已更新: %d → %d tokens (tail_budget=%d)",
                old, threshold, self._tail_token_budget,
            )

    def needs_compression(self, messages: list[dict[str, Any]], max_tokens: int = 0) -> bool:
        """检查消息列表是否需要压缩。

        Phase 6+ 说明：当 max_tokens=0 时使用自身的 token_threshold
        （软阈值），由 SessionManager._try_compress 调用时不传 max_tokens，
        确保压缩器的独立阈值生效。

        【v4.23.0】反抖动保护：连续 2 次压缩节省 < 10% 时跳过压缩。

        Args:
            messages: 消息列表
            max_tokens: token 上限（0=使用默认阈值，即 self._token_threshold）

        Returns:
            True 表示需要压缩
        """
        # ★ v4.23.0 反抖动：连续无效压缩时跳过
        if self._ineffective_compression_count >= 2:
            logger.debug(
                "反抖动：连续 %d 次压缩无效（上次节省 %.1f%%），跳过",
                self._ineffective_compression_count,
                self._last_compression_savings_pct * 100,
            )
            return False
        threshold = max_tokens or self._token_threshold
        return self.estimate_tokens(messages) > threshold

    # ------------------------------------------------------------------
    # Token 估算
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(messages: list[dict[str, Any]]) -> int:
        """估算消息列表的 token 数量。

        Phase 6+ 优化：委托给 token_utils.estimate_tokens()，
        确保 SessionManager 与 ContextCompressor 使用相同算法，
        并计入 reasoning_content（DeepSeek 思考内容）。
        """
        return _unified_estimate_tokens(messages)

    # ------------------------------------------------------------------
    # 对话轮次分组
    # ------------------------------------------------------------------

    @staticmethod
    def _group_rounds(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """将消息按对话轮次分组。

        一轮对话 = user 消息 + (assistant 消息 + tool_calls + tool 结果)
        """
        rounds: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                if current:
                    rounds.append(current)
                current = [msg]
            else:
                current.append(msg)

        if current:
            rounds.append(current)

        return rounds

    # ------------------------------------------------------------------
    # 工具结果裁剪（无需 LLM 的预处理）
    # ------------------------------------------------------------------

    def _prune_tool_results(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """裁剪工具调用结果，减少 token 消耗。

        对超长的 tool 结果内容进行结构化压缩：
        - 保留关键输出的头尾部分
        - 用简短摘要替换中间内容
        - 同时截断 assistant 消息中过长的 tool_call arguments
        """
        # 构建 tool_call_id -> (tool_name, args) 映射
        call_id_map: dict[str, tuple[str, str]] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    if isinstance(tc, dict):
                        cid = tc.get("id", "")
                        fn = tc.get("function", {})
                        call_id_map[cid] = (
                            fn.get("name", "unknown"),
                            fn.get("arguments", ""),
                        )

        result = []
        for msg in messages:
            msg = dict(msg)  # 浅拷贝避免修改原始数据

            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > TOOL_CONTENT_TRUNCATE_LIMIT:
                    # 生成工具结果摘要
                    call_id = msg.get("tool_call_id", "")
                    tool_name, tool_args = call_id_map.get(call_id, ("unknown", ""))
                    summary = self._summarize_tool_result(tool_name, tool_args, content)
                    msg["content"] = summary

            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                # 截断过长的 tool_call arguments
                new_tcs = []
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        args = fn.get("arguments", "")
                        if len(args) > 500:
                            args = self._truncate_json_args(args)
                            tc = {**tc, "function": {**fn, "arguments": args}}
                    new_tcs.append(tc)
                msg["tool_calls"] = new_tcs

            result.append(msg)

        return result

    def prune_tool_results_advanced(
        self,
        messages: list[dict[str, Any]],
        protect_tail_tokens: int = 0,
        preserve_prefix_count: int = 0,
    ) -> list[dict[str, Any]]:
        """三级 Pass 工具结果裁剪（无需 LLM 的廉价预处理）。

        【v4.23.0 新增】移植 Hermes 的三级 Pass 设计：
        - Pass 1: 去重 — 相同内容的 tool 结果只保留最新
        - Pass 2: 信息性摘要替换 — 将旧 tool 结果替换为一行摘要
        - Pass 3: 大参数截断 — 截断旧 assistant 消息中 tool_calls 的 arguments

        【v4.27.0 新增】前缀稳定性保护：
        preserve_prefix_count 指定前 N 条消息不得修改，
        以保护 DeepSeek 上下文缓存的前缀匹配。

        Args:
            messages: 消息列表
            protect_tail_tokens: 保护尾部消息的 token 预算（0=不保护）
            preserve_prefix_count: 保护前缀消息数量（0=不保护）

        Returns:
            裁剪后的消息列表（新列表，不修改原始数据）
        """
        if not messages:
            return messages

        result = [dict(m) for m in messages]  # 浅拷贝

        # 计算尾部保护边界（从后向前累积 token 预算）
        protect_start = len(result)
        if protect_tail_tokens > 0:
            accumulated = 0
            for i in range(len(result) - 1, -1, -1):
                msg_tokens = self.estimate_tokens([result[i]])
                if accumulated + msg_tokens > protect_tail_tokens:
                    break
                accumulated += msg_tokens
                protect_start = i

        # 构建 tool_call_id -> (tool_name, args) 映射
        call_id_map: dict[str, tuple[str, str]] = {}
        for msg in result:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    if isinstance(tc, dict):
                        cid = tc.get("id", "")
                        fn = tc.get("function", {})
                        call_id_map[cid] = (
                            fn.get("name", "unknown"),
                            fn.get("arguments", ""),
                        )

        # Pass 1: 去重 — 从后向前，相同内容的 tool 结果只保留最新
        # ★ v4.27.0: 前缀保护区内的消息跳过裁剪
        _prune_start = preserve_prefix_count  # 前缀保护区右边界
        seen_hashes: dict[str, int] = {}  # hash → 最新消息索引
        for i in range(len(result) - 1, -1, -1):
            if i >= protect_start:
                continue  # 保护区域内的不去重
            if i < _prune_start:
                continue  # ★ v4.27.0: 前缀保护区内跳过
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 200:
                continue
            h = hashlib.md5(content.encode()).hexdigest()[:12]
            if h in seen_hashes:
                result[i] = {
                    **msg,
                    "content": "[重复结果已清除，与后续同名工具调用结果相同]",
                }
            else:
                seen_hashes[h] = i

        # Pass 2: 信息性摘要替换 — 保护区域外的旧 tool 结果替换为摘要
        for i in range(max(_prune_start, 0), protect_start):  # ★ v4.27.0: 从前缀保护区右边界开始
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) <= TOOL_CONTENT_TRUNCATE_LIMIT:
                continue
            # 已经是摘要的不重复处理
            if content.startswith("[") and "→" in content[:50]:
                continue
            call_id = msg.get("tool_call_id", "")
            tool_name, tool_args = call_id_map.get(call_id, ("unknown", ""))
            summary = self._summarize_tool_result(tool_name, tool_args, content)
            result[i] = {**msg, "content": summary}

        # Pass 3: 大参数截断 — 保护区域外的 assistant 消息中 tool_calls 的 arguments
        for i in range(max(_prune_start, 0), protect_start):  # ★ v4.27.0: 从前缀保护区右边界开始
            msg = result[i]
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            new_tcs = []
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "")
                    if len(args) > 500:
                        args = self._truncate_json_args(args)
                        tc = {**tc, "function": {**fn, "arguments": args}}
                new_tcs.append(tc)
            result[i] = {**msg, "tool_calls": new_tcs}

        return result

    @staticmethod
    def _summarize_tool_result(
        tool_name: str, tool_args: str, content: str
    ) -> str:
        """生成工具结果的简短摘要。

        【v4.23.0 增强】为 WeClaw 高频工具添加专门分支，提高摘要信息量。
        """
        content_len = len(content)
        line_count = content.count("\n") + 1 if content.strip() else 0

        try:
            args = json.loads(tool_args) if tool_args else {}
        except (json.JSONDecodeError, TypeError):
            args = {}

        # WeClaw 高频工具专门分支
        if tool_name == "search_history":
            query = str(args.get("query", args.get("keyword", "")))[:30]
            return f"[search_history] 搜索\"{query}\" → {line_count} 条结果"
        elif tool_name == "read_file":
            path = str(args.get("file_path", args.get("path", "")))[-40:]
            return f"[read_file] 读取 {path} ({content_len:,} 字符)"
        elif tool_name in ("shell", "terminal"):
            cmd = str(args.get("command", ""))[:50]
            exit_code = "exit 0" if "成功" in content or "success" in content.lower() else "exit ?"
            return f"[shell] 执行 `{cmd}` → {exit_code}, {line_count} 行输出"
        elif tool_name == "web_search":
            query = str(args.get("query", args.get("keyword", "")))[:30]
            return f"[web_search] 搜索\"{query}\" → {line_count} 条结果"
        elif tool_name == "write_file":
            path = str(args.get("file_path", args.get("path", "")))[-40:]
            return f"[write_file] 写入 {path} ({line_count} 行)"
        elif tool_name == "image_generator":
            prompt = str(args.get("prompt", ""))[:40]
            return f"[image_generator] 生成图片 \"{prompt}\""
        elif tool_name == "stock_query":
            symbol = str(args.get("symbol", args.get("code", "")))
            return f"[stock_query] 查询 {symbol} → {content_len:,} 字符"
        elif tool_name == "paper_lifecycle":
            action = str(args.get("action", ""))
            return f"[paper_lifecycle] {action} → {content_len:,} 字符"
        elif tool_name == "quant_trading":
            action = str(args.get("action", ""))
            return f"[quant_trading] {action} → {content_len:,} 字符"
        elif tool_name == "ocr" or tool_name.startswith("document_"):
            path = str(args.get("file_path", args.get("path", "")))[-40:]
            return f"[{tool_name}] 处理 {path} → {content_len:,} 字符"
        else:
            # 通用 fallback
            preview = content[:150].replace("\n", " ").strip()
            if len(content) > 150:
                preview += "..."
            first_arg = ""
            for k, v in list(args.items())[:2]:
                sv = str(v)[:40]
                first_arg += f" {k}={sv}"
            return (
                f"[{tool_name}]{first_arg} -> {content_len:,} 字符, "
                f"{line_count} 行\n预览: {preview}"
            )

    @staticmethod
    def _truncate_json_args(args: str, head_chars: int = 200) -> str:
        """截断工具调用参数中的长字符串值，保持 JSON 有效性。"""
        try:
            parsed = json.loads(args)
        except (ValueError, TypeError):
            return args

        def _shrink(obj: Any) -> Any:
            if isinstance(obj, str) and len(obj) > head_chars:
                return obj[:head_chars] + "...[截断]"
            if isinstance(obj, dict):
                return {k: _shrink(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_shrink(v) for v in obj]
            return obj

        return json.dumps(_shrink(parsed), ensure_ascii=False)

    # ------------------------------------------------------------------
    # 消息序列化
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_messages(messages: list[dict[str, Any]]) -> str:
        """将消息列表序列化为文本格式，供 LLM 摘要使用。"""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content") or ""

            if isinstance(content, list):
                # 多模态内容：提取文本部分
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and part.get("text"):
                        text_parts.append(part["text"])
                content = "\n".join(text_parts)

            # 截断超长内容
            if len(content) > 4000:
                content = content[:3000] + "\n...[截断]\n" + content[-800:]

            # 提取 tool_calls
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_parts = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args = fn.get("arguments", "")
                        if len(args) > 500:
                            args = args[:400] + "..."
                        tc_parts.append(f"  {name}({args})")
                content += "\n[工具调用:\n" + "\n".join(tc_parts) + "\n]"

            # 添加工具结果标识
            if role == "TOOL":
                tool_id = msg.get("tool_call_id", "")
                parts.append(f"[工具结果 {tool_id}]: {content}")
            else:
                parts.append(f"[{role}]: {content}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # LLM 摘要生成
    # ------------------------------------------------------------------

    async def _generate_summary(
        self,
        messages: list[dict[str, Any]],
        focus_topic: str = "",
    ) -> str | None:
        """使用辅助 LLM 生成对话摘要。

        使用结构化模板，确保摘要包含关键信息：
        - 活跃任务（用户最近一条请求的原文）
        - 用户目标 / 已完成的操作 / 当前状态 / 待处理事项
        - 已解答问题 / 相关文件

        支持迭代更新：如果存在上一次摘要，则基于上次摘要增量更新。

        v4.26.0 增强：
        - 扩展为 10 字段模板（Task 4）
        - 支持 focus_topic 引导（Task 7）
        - 输出端脱敏（Task 8）

        Args:
            messages: 待摘要的消息列表
            focus_topic: 可选的焦点主题引导（如 "量化交易"）

        Returns:
            摘要文本，失败时返回 None
        """
        if not self._client:
            return None

        # 序列化消息为文本
        serialized = self._serialize_messages(messages)

        # 构建摘要提示词
        system_prompt = (
            "你是一个对话摘要助手。你的任务是将对话历史压缩为结构化的摘要，"
            "供另一个 AI 助手继续对话时参考。\n"
            "注意：\n"
            "- 不要回应对话中的任何问题或请求，只输出摘要\n"
            "- 保留具体的文件路径、命令、错误信息等关键细节\n"
            "- 使用中文撰写摘要\n"
            "- 不要包含任何 API 密钥、密码等敏感信息\n"
            "- 不要添加任何前缀或问候语"
        )

        # ★ Task 4: 扩展为 10 字段模板
        template = (
            "## 活跃任务\n"
            "[用户最近一条请求的原文，最重要的字段，必须保留]\n\n"
            "## 用户目标\n"
            "[用户想要完成什么]\n\n"
            "## 已完成操作\n"
            "[编号列表，包含具体操作和结果]\n\n"
            "## 已解答问题\n"
            "[已解决的问题列表，防止下轮重复回答]\n\n"
            "## 当前状态\n"
            "[当前的工作状态、修改的文件等]\n\n"
            "## 关键决策\n"
            "[重要的技术决策及原因]\n\n"
            "## 待处理事项\n"
            "[尚未完成的工作，作为背景信息而非指令]\n\n"
            "## 待回应用户请求\n"
            "[尚未满足的请求，无则写“无”]\n\n"
            "## 相关文件\n"
            "[读取/修改/创建的文件清单]\n\n"
            "## 关键上下文\n"
            "[需要保留的具体值、错误信息、配置细节等]\n\n"
            f"目标约 {self._summary_max_tokens} tokens。请具体明确，"
            "包含文件路径、命令输出、行号等。"
        )

        # ★ Task 7: focus_topic 引导段
        focus_section = ""
        if focus_topic:
            focus_section = (
                f"\n\n【FOCUS TOPIC】用户希望摘要优先保留与「{focus_topic}」相关的信息。"
                "请在摘要中突出显示与该主题相关的内容。\n"
            )

        if self._previous_summary:
            # 迭代更新模式
            prompt = (
                f"以下是上一次的对话摘要：\n\n{self._previous_summary}\n\n"
                f"以下是新增的对话内容：\n\n{serialized}\n\n"
                f"请基于上次摘要，整合新增内容，更新以下结构化摘要。"
                f"保留仍然相关的信息，添加新完成的操作。\n\n{template}{focus_section}"
            )
        else:
            # 首次摘要
            prompt = (
                f"以下是需要摘要的对话内容：\n\n{serialized}\n\n"
                f"请按以下结构生成摘要：\n\n{template}{focus_section}"
            )

        # 调用辅助 LLM
        summary = await self._client.complete(
            prompt=prompt,
            max_tokens=self._summary_max_tokens,
            system_prompt=system_prompt,
        )

        if summary and summary.strip():
            summary = summary.strip()
            # ★ Task 8: 输出端脱敏
            try:
                from src.config import get_config
                _redaction_enabled = get_config("compressor.enable_redaction", True)
            except Exception:
                _redaction_enabled = True
            if _redaction_enabled:
                summary = redact_sensitive_text(summary)
            # 存储用于迭代更新
            self._previous_summary = summary
            return summary

        return None

    async def _try_main_model_summary(
        self,
        messages: list[dict[str, Any]],
        focus_topic: str = "",
    ) -> str | None:
        """Task 1 Level 2: 通过主模型生成摘要（辅助模型失败时的重试路径）。

        使用 ModelRegistry.chat() 调用主模型，确保 CostTracker 费用记录完整。
        独立超时 30s，含限流检测（429/503 直接返回 None）。

        Returns:
            摘要文本，失败时返回 None
        """
        if not self._model_registry or not self._main_model_key:
            return None

        try:
            from src.config import get_config
            _fallback_enabled = get_config("compressor.enable_main_model_fallback", True)
        except Exception:
            _fallback_enabled = True
        if not _fallback_enabled:
            return None

        try:
            import asyncio
            serialized = self._serialize_messages(messages)
            prompt = (
                f"请将以下对话历史压缩为结构化摘要（包含：活跃任务、用户目标、"
                f"已完成操作、当前状态、待处理事项、相关文件、关键上下文）：\n\n{serialized}"
            )
            if focus_topic:
                prompt += f"\n\n请优先保留与「{focus_topic}」相关的信息。"

            response = await asyncio.wait_for(
                self._model_registry.chat(
                    self._main_model_key,
                    messages=[
                        {"role": "system", "content": "你是对话摘要助手，只输出摘要。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self._summary_max_tokens,
                ),
                timeout=30,
            )
            # 提取文本内容
            content = ""
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
            elif isinstance(response, str):
                content = response

            if content and content.strip():
                logger.info("主模型重试摘要成功")
                return content.strip()
        except asyncio.TimeoutError:
            logger.warning("主模型重试摘要超时 (30s)")
        except Exception as e:
            # 限流错误（429/503）直接返回 None
            err_str = str(e)
            if "429" in err_str or "503" in err_str or "rate" in err_str.lower():
                logger.warning("主模型重试摘要被限流: %s", e)
            else:
                logger.warning("主模型重试摘要失败: %s", e)
        return None

    def _generate_static_fallback(
        self,
        messages: list[dict[str, Any]],
        n_dropped: int,
    ) -> str:
        """Task 1 Level 3: 生成静态 fallback 文本（所有 LLM 摘要都失败时）。

        从被压缩消息中提取关键元信息（文件路径、工具名），
        生成无 LLM 依赖的结构化标记。

        Returns:
            静态 fallback 文本
        """
        import re as _re
        files: list[str] = []
        tool_names: list[str] = []
        for msg in messages:
            content = str(msg.get("content", "") or "")
            for m in _re.finditer(
                r'[\w/\\.-]+\.(?:py|js|ts|md|txt|json|toml|yaml|yml)\b', content
            ):
                f = m.group(0)
                if f not in files and len(files) < 5:
                    files.append(f)
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict):
                    name = tc.get("function", {}).get("name", "")
                    if name and name not in tool_names and len(tool_names) < 5:
                        tool_names.append(name)

        lines = [
            "[上下文压缩 — 静态回退]",
            f"- 已移除 {n_dropped} 条早期消息",
        ]
        if files:
            lines.append(f"- 最近操作文件: {', '.join(files)}")
        if tool_names:
            lines.append(f"- 最后工具调用: {', '.join(tool_names)}")
        lines.append("请基于可见的后续消息继续。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Token-budget 尾部保护（Task 5/6）
    # ------------------------------------------------------------------

    def _find_tail_cut_by_tokens(
        self,
        messages: list[dict[str, Any]],
        head_end: int,
        tail_budget: int,
    ) -> int:
        """从后向前累积 token，确定尾部保护的切割点。

        Args:
            messages: 消息列表（不含 system prompt）
            head_end: 头部保护区域的结束索引
            tail_budget: 尾部 token 预算

        Returns:
            切割点索引（该索引及之后的消息保留在尾部）
        """
        accumulated = 0
        cut_idx = len(messages)
        soft_ceiling = int(tail_budget * 1.5)  # 防单消息过大

        for i in range(len(messages) - 1, head_end, -1):
            msg_tokens = self.estimate_tokens([messages[i]])
            if accumulated + msg_tokens > soft_ceiling:
                break
            accumulated += msg_tokens
            cut_idx = i
            if accumulated >= tail_budget:
                break

        # 确保至少保留 3 条消息
        min_keep = max(head_end + 1, len(messages) - 3)
        cut_idx = min(cut_idx, min_keep)

        # 对齐到 user 消息边界（轮次起点）
        cut_idx = self._align_boundary_backward(messages, cut_idx, head_end)

        return cut_idx

    @staticmethod
    def _align_boundary_backward(
        messages: list[dict[str, Any]],
        idx: int,
        head_end: int,
    ) -> int:
        """向前对齐到 user 消息边界（轮次起点）。

        确保 early_rounds / recent_rounds 分割点是完整轮次。
        """
        while idx > head_end and idx < len(messages):
            if messages[idx].get("role") == "user":
                break
            idx -= 1
        # 不能进入 head 保护区域
        return max(idx, head_end + 1)

    @staticmethod
    def _ensure_last_user_in_tail(
        messages: list[dict[str, Any]],
        cut_idx: int,
        head_end: int,
    ) -> int:
        """Task 6: 确保最后一条 user 消息在尾部区域。

        从后向前查找最后一条 user 消息，若不在尾部则将 cut_idx 前移。

        Returns:
            调整后的切割点索引
        """
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx >= 0 and last_user_idx < cut_idx:
            # user 消息在被压缩区域，前移 cut_idx
            new_cut = last_user_idx
            # 不能进入 head 保护区域
            new_cut = max(new_cut, head_end + 1)
            # 对齐到轮次边界
            while new_cut > head_end and new_cut < len(messages):
                if messages[new_cut].get("role") == "user":
                    break
                new_cut -= 1
            new_cut = max(new_cut, head_end + 1)
            logger.debug(
                "用户消息锚定: cut_idx %d → %d (last_user=%d)",
                cut_idx, new_cut, last_user_idx,
            )
            return new_cut
        return cut_idx
        """将消息列表序列化为易于摘要的文本格式。"""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content") or ""

            if isinstance(content, list):
                # 多模态内容，只提取文本
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and part.get("text"):
                        text_parts.append(part["text"])
                content = "\n".join(text_parts)

            # 截断过长内容
            if len(content) > 4000:
                content = content[:3000] + "\n...[截断]...\n" + content[-800:]

            # 处理 tool_calls
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_parts = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args = fn.get("arguments", "")
                        if len(args) > 500:
                            args = args[:400] + "..."
                        tc_parts.append(f"  {name}({args})")
                content += "\n[工具调用:\n" + "\n".join(tc_parts) + "\n]"

            # 处理 tool 结果
            if role == "TOOL":
                tool_id = msg.get("tool_call_id", "")
                parts.append(f"[工具结果 {tool_id}]: {content}")
            else:
                parts.append(f"[{role}]: {content}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _choose_summary_role(
        system_msg: dict[str, Any] | None,
        recent_rounds: list[list[dict[str, Any]]],
    ) -> str:
        """选择摘要消息的角色，避免与相邻消息冲突。

        OpenAI API 要求消息角色交替出现（user/assistant），
        摘要消息需要选择合适的角色以保持结构合法。
        """
        # 最近轮次的第一条消息通常是 user
        first_recent_role = "user"
        if recent_rounds and recent_rounds[0]:
            first_recent_role = recent_rounds[0][0].get("role", "user")

        # 如果最近轮次第一条是 user，摘要用 assistant；反之用 user
        if first_recent_role == "user":
            return "assistant"
        return "user"

    def _validate_tool_pairs(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """验证 tool_call / tool_result 配对完整性。

        确保：
        1. 每个 tool 结果消息都有对应的 assistant tool_call
        2. 每个 assistant tool_call 都有对应的 tool 结果
        """
        # 收集所有 assistant 中的 tool_call_id
        call_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    cid = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    if cid:
                        call_ids.add(cid)

        # 收集所有 tool 结果的 call_id
        result_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id", "")
                if cid:
                    result_ids.add(cid)

        # 移除孤立的 tool 结果（没有对应的 tool_call）
        orphan_results = result_ids - call_ids
        if orphan_results:
            messages = [
                m for m in messages
                if not (
                    m.get("role") == "tool"
                    and m.get("tool_call_id") in orphan_results
                )
            ]
            logger.info("移除 %d 条孤立的工具结果消息", len(orphan_results))

        # 为缺失结果的 tool_call 补充占位消息
        missing_results = call_ids - result_ids
        if missing_results:
            patched: list[dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls", []):
                        cid = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "tool_call_id": cid,
                                "content": "[工具结果已在上下文压缩中移除，详见上方摘要]",
                            })
            messages = patched
            logger.info("补充 %d 条缺失的工具结果占位消息", len(missing_results))

        return messages

    def _is_in_cooldown(self) -> bool:
        """检查是否在摘要失败冷却期内。"""
        if self._failure_cooldown_until <= 0:
            return False
        if time.monotonic() >= self._failure_cooldown_until:
            self._failure_cooldown_until = 0.0
            return False
        return True

    def reset(self) -> None:
        """重置压缩器状态（新会话时调用）。"""
        self._previous_summary = None
        self._compression_count = 0
        self._last_stats = None
        self._failure_cooldown_until = 0.0
        self._ineffective_compression_count = 0
        self._last_compression_savings_pct = 0.0

    def get_stats_summary(self) -> dict[str, Any]:
        """获取压缩统计摘要。"""
        stats = self._last_stats
        return {
            "compression_count": self._compression_count,
            "has_auxiliary": self._client is not None,
            "last_compression": {
                "original_tokens": stats.original_tokens if stats else 0,
                "compressed_tokens": stats.compressed_tokens if stats else 0,
                "compression_ratio": round(stats.compression_ratio, 3) if stats else 0,
                "duration_ms": round(stats.duration_ms, 1) if stats else 0,
                "used_fallback": stats.used_fallback if stats else False,
            } if stats else None,
        }
