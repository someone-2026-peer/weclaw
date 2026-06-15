"""统一 Token 估算模块。

解决 session.py 与 context_compressor.py 各自估算不一致的问题，
提供单一可信来源（Single Source of Truth）。

策略：
1. 优先使用 litellm.token_counter（精确，需缓存 tokenizer 可用性）
2. 回退：字符比例混合估算（中文 1.5字/token, 英文 4字符/token）
3. ★ 计入 reasoning_content（DeepSeek 等推理模型的思考内容）
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 模块级缓存：litellm tokenizer 可用性（model_key -> bool）
# 首次调用后缓存结果，避免每次都尝试加载 tokenizer
_litellm_available: dict[str, bool] = {}


def estimate_tokens(messages: list[dict[str, Any]], model: str = "") -> int:
    """统一 token 估算。

    两处调用方（SessionManager._estimate_tokens / ContextCompressor.estimate_tokens）
    均应委托给本函数，确保估算结果一致。

    Args:
        messages: OpenAI 格式的消息列表
        model: 模型标识（用于 litellm.token_counter，为空时跳过精确估算）

    Returns:
        估算的 token 总数
    """
    # 优先尝试 litellm 精确估算（带缓存保护）
    if model and _litellm_available.get(model, True):
        try:
            import litellm  # type: ignore
            result = litellm.token_counter(model=model, messages=messages)
            _litellm_available[model] = True
            return int(result)
        except Exception:
            # 标记该模型不可用，后续跳过
            _litellm_available[model] = False
            logger.debug("litellm.token_counter 对模型 %s 不可用，回退字符估算", model)

    # 回退：字符比例混合估算
    return _fallback_estimate(messages)


def _fallback_estimate(messages: list[dict[str, Any]]) -> int:
    """字符比例混合估算。

    规则：
    - 中文（CJK）字符：约 1.5 字/token
    - 英文/数字/符号：约 4 字符/token
    - tool_calls 参数：按混合中英文估算（字符数 / 2）
    - reasoning_content（DeepSeek 思考内容）：同 content 规则
    - 每条消息基础开销：4 tokens（OpenAI 格式）
    """
    total = 0
    for msg in messages:
        # ── content 字段 ──────────────────────────────────────────────────
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += _estimate_text_tokens(content)
        elif isinstance(content, list):
            # 多模态内容：提取所有文本部分
            for part in content:
                if isinstance(part, str):
                    total += _estimate_text_tokens(part)
                elif isinstance(part, dict):
                    text = part.get("text", "")
                    if text:
                        total += _estimate_text_tokens(text)

        # ── reasoning_content 字段（DeepSeek / R1 思考内容）────────────────
        # ★ v4.27.0 修正：仅工具调用轮次的 reasoning_content 被 API 计入 token
        # DeepSeek 文档：非工具调用轮次的 reasoning_content 在后续轮次传入时会被忽略
        reasoning = msg.get("reasoning_content") or ""
        if isinstance(reasoning, str) and reasoning:
            has_tool_calls = bool(msg.get("tool_calls"))
            if has_tool_calls:
                # 工具调用轮次：reasoning_content 必须回传，计入 token
                total += _estimate_text_tokens(reasoning)
            # 非工具调用轮次：API 忽略此字段，不计入 token 估算

        # ── tool_calls 参数 ──────────────────────────────────────────────
        for tc in msg.get("tool_calls", []):
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                args = str(fn.get("arguments", ""))
                # 参数通常混合中英文，保守使用 len/2
                total += len(args) // 2
                total += len(fn.get("name", ""))

        # ── 消息基础开销（OpenAI chat format 约 4 tokens）────────────────
        total += 4

    return total


def _estimate_text_tokens(text: str) -> int:
    """估算纯文本的 token 数。

    中文（CJK 统一表意文字）约 1.5 字/token；
    英文/数字/标点约 4 字符/token。
    """
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - cn_chars
    return int(cn_chars / 1.5 + other_chars / 4)


def estimate_chars_per_token(messages: list[dict[str, Any]]) -> float:
    """估算当前消息列表的平均 字符/token 比率。

    用于 _truncate_messages 的反向估算（从 token_limit 换算字符预算）。
    若无法估算，返回保守默认值 2.0。

    Returns:
        平均每 token 对应的字符数
    """
    total_chars = 0
    for msg in messages:
        content = str(msg.get("content", "") or "")
        total_chars += len(content)
        reasoning = str(msg.get("reasoning_content", "") or "")
        total_chars += len(reasoning)
        for tc in msg.get("tool_calls", []):
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                total_chars += len(str(fn.get("arguments", "")))
                total_chars += len(fn.get("name", ""))

    if total_chars == 0:
        return 2.0  # 保守默认值

    tokens = _fallback_estimate(messages)
    if tokens == 0:
        return 2.0

    ratio = total_chars / tokens
    # 钳制在合理范围：中文约 1.5，英文约 4，混合 2~3
    return max(1.5, min(4.0, ratio))
