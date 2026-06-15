"""敏感信息脱敏模块 — 公共脱敏规则，供 context_compressor / task_trace 共享。

v4.26.0 新增：从 task_trace.py 提取，统一脱敏规则并增强供应商前缀匹配。
"""

from __future__ import annotations

import re
from typing import Any

# 敏感参数名列表（用于参数级脱敏）
SENSITIVE_PARAM_NAMES: set[str] = {
    "api_key", "apikey", "api-key",
    "password", "passwd", "pwd",
    "token", "access_token", "refresh_token", "auth_token",
    "secret", "secret_key", "client_secret",
    "credential", "credentials",
    "private_key", "privatekey",
    "authorization", "auth",
}

# 敏感值模式（正则）— 仅在摘要输出端使用，增加严格边界条件
_SENSITIVE_VALUE_PATTERNS: list[re.Pattern] = [
    # 供应商前缀密钥（sk-, Bearer, ghp_, AKIA 等）+ 至少 20 字符长随机串
    re.compile(
        r'\b(?:sk-[a-zA-Z0-9_-]{20,}|Bearer\s+[a-zA-Z0-9._-]{20,}'
        r'|ghp_[a-zA-Z0-9]{20,}|AKIA[A-Z0-9]{16,})\b'
    ),
    # 明显的 key=xxx 模式（代码块保护在 redact_sensitive_text 函数级处理）
    re.compile(
        r'(api_key|apikey|api-key|token|secret|password)\s*[=:]\s*'
        r'["\']?[\w-]{10,}["\']?',
        re.I,
    ),
]

# 代码块保护正则（用于在脱敏前临时替换代码块）
_CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')

# 脱敏占位符
REDACTED = "[REDACTED]"


def redact_sensitive_text(text: str) -> str:
    """对文本中的敏感信息进行脱敏。

    仅在摘要输出端使用，不在输入端使用（避免破坏代码对话上下文）。
    保护代码块内容不被误脱敏。

    Args:
        text: 待脱敏文本

    Returns:
        脱敏后的文本
    """
    if not text:
        return text

    # 保护代码块：临时替换为占位符
    code_blocks: list[str] = []

    def _save_block(m: re.Match) -> str:
        code_blocks.append(m.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    protected = _CODE_BLOCK_RE.sub(_save_block, text)

    # 脱敏
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        protected = pattern.sub(REDACTED, protected)

    # 恢复代码块
    for i, block in enumerate(code_blocks):
        protected = protected.replace(f"__CODE_BLOCK_{i}__", block)

    return protected


def redact_param_value(key: str, value: Any) -> Any:
    """根据参数名判断是否需要脱敏。

    Args:
        key: 参数名
        value: 参数值

    Returns:
        脱敏后的值（如果参数名在敏感列表中则返回 "***"）
    """
    key_lower = key.lower().replace("-", "_")
    if key_lower in SENSITIVE_PARAM_NAMES:
        return "***"
    return value
