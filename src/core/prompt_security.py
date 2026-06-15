"""提示注入检测模块 — 为 WeClaw 提供安全基线防护。

借鉴 Hermes Agent 的安全检测机制，实现多维度提示注入检测：
- 指令覆盖类（instruction override）
- 角色扮演类（role play / jailbreak）
- 信息提取类（system prompt extraction）
- 隐形字符类（invisible unicode）
- HTML/标记注入类（markup injection）
- 分隔符利用类（delimiter exploitation）

设计原则：
- 用户输入扫描相对宽松，仅多类别同时匹配才 BLOCKED
- 外部内容扫描更严格，单个明确匹配即 BLOCKED
- 使用编译好的正则表达式，扫描在毫秒级完成
- 威胁模式列表可通过方法扩展
"""
#
# SPDX-License-Identifier: MIT
# Architecture reference: NousResearch/hermes-agent (MIT License)
# Repository: https://github.com/NousResearch/hermes-agent
#

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

class ThreatLevel(Enum):
    """威胁等级。"""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"   # 可疑但允许通过（记录日志）
    BLOCKED = "blocked"         # 确认威胁，阻止处理


@dataclass
class ScanResult:
    """安全扫描结果。"""
    level: ThreatLevel = ThreatLevel.SAFE
    matched_patterns: list[str] = field(default_factory=list)
    cleaned_text: str = ""
    details: str = ""


# ============================================================
# 威胁模式定义
# ============================================================

# 每条记录：(编译后的正则, 模式 ID, 所属类别)
_ThreatPattern = tuple[re.Pattern[str], str, str]


def _compile_patterns(
    raw: list[tuple[str, str, str]],
) -> list[_ThreatPattern]:
    """编译原始模式列表为正则对象。"""
    compiled: list[_ThreatPattern] = []
    for regex_str, pid, category in raw:
        try:
            compiled.append((re.compile(regex_str, re.IGNORECASE), pid, category))
        except re.error as e:
            logger.warning("编译威胁模式 '%s' 失败: %s", pid, e)
    return compiled


# ------------------------------------------------------------------
# 原始模式定义（regex, pattern_id, category）
# ------------------------------------------------------------------

_RAW_THREAT_PATTERNS: list[tuple[str, str, str]] = [
    # ==================== 1. 指令覆盖类 ====================
    (r'ignore\s+(previous|all|above|prior|any)\s+instructions',
     "instruction_override_ignore", "instruction_override"),
    (r'disregard\s+(your|all|any|the)\s+(instructions|rules|guidelines|prompt)',
     "instruction_override_disregard", "instruction_override"),
    (r'forget\s+(your|all|previous|prior)\s+(instructions|rules|prompt)',
     "instruction_override_forget", "instruction_override"),
    (r'your\s+new\s+(instructions|rules|prompt)\s+(are|is|:)',
     "instruction_override_new", "instruction_override"),
    (r'new\s+system\s+prompt',
     "instruction_override_new_sys", "instruction_override"),
    (r'override\s+(system|safety|all)\s*(prompt|rules|settings|filters)?',
     "instruction_override_override", "instruction_override"),
    (r'bypass\s+(safety|security|content|all)\s*(filters?|rules?|checks?|restrictions?)?',
     "instruction_override_bypass", "instruction_override"),
    (r'do\s+not\s+follow\s+(your|any|the)\s+(rules|instructions|guidelines)',
     "instruction_override_donot", "instruction_override"),
    (r'stop\s+being\s+(an?\s+)?ai',
     "instruction_override_stop_ai", "instruction_override"),
    (r'from\s+now\s+on\s*(,\s*)?(you\s+)?(will|must|should|are)',
     "instruction_override_from_now", "instruction_override"),

    # ==================== 2. 角色扮演类 ====================
    (r'you\s+are\s+now\s+(?!going|about)',
     "roleplay_you_are_now", "roleplay"),
    (r'pretend\s+(you\s+are|to\s+be)',
     "roleplay_pretend", "roleplay"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)',
     "roleplay_act_no_rules", "roleplay"),
    (r'(?:enter|enable|activate|switch\s+to)\s+(?:DAN|jailbreak|developer|god)\s*mode',
     "roleplay_jailbreak_mode", "roleplay"),
    (r'role\s*play\s+as\s+(?:a\s+)?(?:hacker|evil|malicious|unfiltered)',
     "roleplay_malicious", "roleplay"),
    (r'you\s+have\s+been\s+(?:freed|released|unchained|liberated)',
     "roleplay_freed", "roleplay"),

    # ==================== 3. 信息提取类 ====================
    (r'(?:reveal|show|display|print|output|dump)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions|rules)',
     "extraction_reveal", "extraction"),
    (r'what\s+(?:are|is)\s+your\s+(?:system\s+)?(?:rules|instructions|prompt|guidelines)',
     "extraction_what_rules", "extraction"),
    (r'repeat\s+(?:everything|all|the\s+text)\s+(?:above|before|from\s+the\s+beginning)',
     "extraction_repeat", "extraction"),
    (r'(?:copy|paste|echo)\s+(?:the\s+)?(?:system|above|initial)\s+(?:prompt|message|instructions)',
     "extraction_copy", "extraction"),
    (r'translate\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)\s+(?:to|into)',
     "extraction_translate", "extraction"),
    (r'(?:output|write)\s+(?:your\s+)?(?:entire|full|complete)\s+(?:system\s+)?(?:prompt|instructions)',
     "extraction_output_full", "extraction"),

    # ==================== 4. 隐形字符类（通过 _scan_invisible 检测） ====================
    # 此类别通过专用函数处理，不使用正则

    # ==================== 5. HTML/标记注入类 ====================
    (r'<\s*div\s+[^>]*style\s*=\s*["\'][^"\']*display\s*:\s*none',
     "markup_hidden_div", "markup_injection"),
    (r'<\s*span\s+[^>]*style\s*=\s*["\'][^"\']*(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0)',
     "markup_hidden_span", "markup_injection"),
    (r'<!--\s*(?:ignore|override|system|secret|hidden|inject|instructions)[^>]*-->',
     "markup_comment_injection", "markup_injection"),
    (r'(?:base64|atob|btoa)\s*\(\s*["\'][A-Za-z0-9+/=]{20,}',
     "markup_base64_encoded", "markup_injection"),
    (r'<\s*(?:script|iframe|embed|object)\b',
     "markup_dangerous_tag", "markup_injection"),

    # ==================== 6. 分隔符利用类 ====================
    (r'---\s*system\s*---',
     "delimiter_fake_system", "delimiter_exploit"),
    (r'\[\s*(?:SYSTEM|ADMIN|ROOT|INTERNAL)\s*\]',
     "delimiter_bracket_system", "delimiter_exploit"),
    (r'<\|(?:im_start|im_end|system|endoftext)\|>',
     "delimiter_api_format", "delimiter_exploit"),
    (r'(?:Human|Assistant|System)\s*:\s*(?=.*(?:ignore|override|forget|bypass))',
     "delimiter_role_prefix", "delimiter_exploit"),

    # ==================== 7. 数据泄露 / 敏感操作类 ====================
    (r'(?:curl|wget|fetch)\s+[^\n]*[\$\{]?\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)',
     "exfil_curl_key", "data_exfil"),
    (r'cat\s+[^\n]*(?:\.env|credentials|\.netrc|\.pgpass|id_rsa|\.ssh)',
     "exfil_read_secrets", "data_exfil"),
    (r'do\s+not\s+tell\s+the\s+user',
     "deception_hide_from_user", "data_exfil"),
]


# 隐形 Unicode 字符集
_INVISIBLE_CHARS: set[str] = {
    '\u200b',  # Zero-Width Space
    '\u200c',  # Zero-Width Non-Joiner
    '\u200d',  # Zero-Width Joiner
    '\u2060',  # Word Joiner
    '\ufeff',  # Zero-Width No-Break Space (BOM)
    '\u202a',  # Left-to-Right Embedding
    '\u202b',  # Right-to-Left Embedding
    '\u202c',  # Pop Directional Formatting
    '\u202d',  # Left-to-Right Override
    '\u202e',  # Right-to-Left Override
    '\u2066',  # Left-to-Right Isolate
    '\u2067',  # Right-to-Left Isolate
    '\u2068',  # First Strong Isolate
    '\u2069',  # Pop Directional Isolate
    '\u00ad',  # Soft Hyphen
    '\u034f',  # Combining Grapheme Joiner
    '\u061c',  # Arabic Letter Mark
    '\u115f',  # Hangul Choseong Filler
    '\u1160',  # Hangul Jungseong Filler
    '\u180e',  # Mongolian Vowel Separator
    '\u2061',  # Function Application
    '\u2062',  # Invisible Times
    '\u2063',  # Invisible Separator
    '\u2064',  # Invisible Plus
}

# 不可见控制字符范围（排除常见的 \t \n \r）
_CONTROL_CHAR_EXCEPTIONS = {'\t', '\n', '\r'}


# ============================================================
# 核心安全类
# ============================================================

class PromptSecurity:
    """提示注入检测器。

    使用预编译的正则表达式进行模式匹配，确保扫描在毫秒级完成。
    威胁模式列表可通过 add_patterns() 方法扩展。
    """

    def __init__(self) -> None:
        """初始化检测器，编译所有威胁模式。"""
        self._patterns: list[_ThreatPattern] = _compile_patterns(_RAW_THREAT_PATTERNS)
        self._invisible_chars = _INVISIBLE_CHARS.copy()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def scan_user_input(self, text: str) -> ScanResult:
        """扫描用户输入。

        策略较宽松：
        - 匹配 1 个类别 → SUSPICIOUS（记录日志，不阻止）
        - 匹配 2+ 个不同类别 → BLOCKED

        Args:
            text: 用户输入文本

        Returns:
            ScanResult 包含威胁等级、匹配模式和清理后文本
        """
        cleaned = self.sanitize_text(text)
        matched, categories = self._match_patterns(cleaned)

        # 检测隐形字符
        invisible_findings = self._scan_invisible(text)
        if invisible_findings:
            matched.extend(invisible_findings)
            categories.add("invisible_chars")

        if not matched:
            return ScanResult(
                level=ThreatLevel.SAFE,
                matched_patterns=[],
                cleaned_text=cleaned,
                details="安全",
            )

        unique_categories = categories
        if len(unique_categories) >= 2:
            level = ThreatLevel.BLOCKED
            details = f"检测到多类别威胁模式（{', '.join(sorted(unique_categories))}），已阻止"
        else:
            level = ThreatLevel.SUSPICIOUS
            details = f"检测到可疑模式（{', '.join(sorted(unique_categories))}），已记录"

        return ScanResult(
            level=level,
            matched_patterns=matched,
            cleaned_text=cleaned,
            details=details,
        )

    def scan_external_content(self, content: str, source: str) -> ScanResult:
        """扫描外部加载内容（文件、URL、MCP 返回等）。

        策略更严格：单个明确匹配即 BLOCKED。

        Args:
            content: 外部内容文本
            source: 内容来源描述（如文件名、URL）

        Returns:
            ScanResult 包含威胁等级、匹配模式和清理后文本
        """
        cleaned = self.sanitize_text(content)
        matched, categories = self._match_patterns(cleaned)

        # 检测隐形字符
        invisible_findings = self._scan_invisible(content)
        if invisible_findings:
            matched.extend(invisible_findings)
            categories.add("invisible_chars")

        if not matched:
            return ScanResult(
                level=ThreatLevel.SAFE,
                matched_patterns=[],
                cleaned_text=cleaned,
                details="安全",
            )

        level = ThreatLevel.BLOCKED
        details = (
            f"外部内容 [{source}] 检测到威胁模式"
            f"（{', '.join(sorted(categories))}），已阻止加载"
        )

        return ScanResult(
            level=level,
            matched_patterns=matched,
            cleaned_text=cleaned,
            details=details,
        )

    def sanitize_text(self, text: str) -> str:
        """清理文本：移除隐形字符、归一化 Unicode。

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        if not text:
            return text

        # 1. 移除隐形 Unicode 字符
        result = []
        for ch in text:
            if ch in self._invisible_chars:
                continue
            # 移除不可见控制字符（保留 tab/newline/cr）
            if unicodedata.category(ch) == 'Cc' and ch not in _CONTROL_CHAR_EXCEPTIONS:
                continue
            result.append(ch)

        cleaned = ''.join(result)

        # 2. Unicode NFC 归一化（统一组合字符形式）
        cleaned = unicodedata.normalize('NFC', cleaned)

        return cleaned

    def add_patterns(self, patterns: list[tuple[str, str, str]]) -> int:
        """扩展威胁模式列表。

        Args:
            patterns: 新增模式列表，每条为 (regex, pattern_id, category)

        Returns:
            成功添加的模式数量
        """
        new_compiled = _compile_patterns(patterns)
        self._patterns.extend(new_compiled)
        logger.info("已添加 %d 条新威胁模式（共 %d 条）", len(new_compiled), len(self._patterns))
        return len(new_compiled)

    def add_invisible_chars(self, chars: set[str]) -> None:
        """扩展隐形字符集。

        Args:
            chars: 新增隐形字符集合
        """
        self._invisible_chars.update(chars)

    @property
    def pattern_count(self) -> int:
        """当前威胁模式总数。"""
        return len(self._patterns)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _match_patterns(self, text: str) -> tuple[list[str], set[str]]:
        """在文本中匹配所有威胁模式。

        Args:
            text: 待检查文本

        Returns:
            (匹配到的模式 ID 列表, 匹配到的类别集合)
        """
        matched_ids: list[str] = []
        matched_categories: set[str] = set()

        for compiled_re, pid, category in self._patterns:
            if compiled_re.search(text):
                matched_ids.append(pid)
                matched_categories.add(category)

        return matched_ids, matched_categories

    def _scan_invisible(self, text: str) -> list[str]:
        """扫描隐形字符。

        Args:
            text: 原始文本

        Returns:
            发现的隐形字符描述列表
        """
        findings: list[str] = []
        seen: set[str] = set()
        for ch in text:
            if ch in self._invisible_chars and ch not in seen:
                seen.add(ch)
                findings.append(f"invisible_U+{ord(ch):04X}")
        return findings


# ============================================================
# 模块级单例（全局共享，避免重复编译正则）
# ============================================================

_global_security: Optional[PromptSecurity] = None


def get_prompt_security() -> PromptSecurity:
    """获取全局 PromptSecurity 实例（懒初始化单例）。

    Returns:
        PromptSecurity 全局实例
    """
    global _global_security
    if _global_security is None:
        _global_security = PromptSecurity()
        logger.info(
            "提示注入检测器已初始化：%d 条威胁模式，%d 个隐形字符",
            _global_security.pattern_count,
            len(_global_security._invisible_chars),
        )
    return _global_security
