"""工具暴露策略引擎 — 根据意图置信度分层暴露工具 Schema。

Phase 6 新增模块，核心功能：
1. 渐进式工具暴露：根据意图置信度从推荐集 → 扩展集 → 全量集
2. Schema 动态优先级标注：在 description 前添加 [推荐]/[备选] 引导模型
3. 工具依赖自动解析：确保依赖工具不被意外过滤
4. 自动回退：连续失败时自动升级到更大工具集

设计原则："引导而非限制，渐进而非决断"
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from src.core.prompts import (
    IntentResult,
    INTENT_TOOL_MAPPING,
    INTENT_PRIORITY_MAP,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Schema 动态优先级标注
# ------------------------------------------------------------------

def annotate_schema_priority(
    schemas: list[dict[str, Any]],
    intent_result: Any,
) -> list[dict[str, Any]]:
    """根据意图在 Schema description 前添加优先级标注。

    不删除任何工具，只在 description 前添加 [推荐] / [备选] / [禁用] 前缀，
    引导模型优先选择相关工具。

    Args:
        schemas: 原始 function calling schema 列表
        intent_result: 意图识别结果（支持对象或字典格式）

    Returns:
        标注后的 schema 列表（深拷贝，不修改原始数据）
    """
    # Phase 6+: LLM 模式优先使用 LLM 返回的推荐/禁用信息
    llm_recommended = set()
    llm_forbidden = set()
    if hasattr(intent_result, 'llm_recommended_tools'):
        llm_recommended = set(intent_result.llm_recommended_tools or [])
        llm_forbidden = set(intent_result.llm_forbidden_tools or [])

    if llm_recommended or llm_forbidden:
        # LLM 模式标注
        annotated: list[dict[str, Any]] = []
        for schema in schemas:
            func_info = schema.get("function", {})
            func_name = func_info.get("name", "")
            tool_name = _extract_tool_name(func_name)

            if tool_name in llm_recommended:
                schema = copy.deepcopy(schema)
                desc = schema["function"].get("description", "")
                schema["function"]["description"] = f"[推荐] {desc}"
            elif tool_name in llm_forbidden:
                schema = copy.deepcopy(schema)
                desc = schema["function"].get("description", "")
                schema["function"]["description"] = f"[禁用] {desc}"

            annotated.append(schema)
        return annotated

    # 规则模式标注（原有逻辑）
    # 支持对象和字典两种格式
    if hasattr(intent_result, 'primary_intent'):
        primary_intent = intent_result.primary_intent
    elif isinstance(intent_result, dict):
        primary_intent = intent_result.get("primary_intent", "")
    else:
        primary_intent = ""

    if not primary_intent:
        return schemas

    priority = INTENT_PRIORITY_MAP.get(primary_intent, {})
    recommended = set(priority.get("recommended", []))
    alternative = set(priority.get("alternative", []))

    if not recommended and not alternative:
        return schemas

    annotated: list[dict[str, Any]] = []
    for schema in schemas:
        func_info = schema.get("function", {})
        func_name = func_info.get("name", "")

        # 从 func_name 提取工具名（格式: tool_name_action_name）
        tool_name = _extract_tool_name(func_name)

        if tool_name in recommended:
            schema = copy.deepcopy(schema)
            desc = schema["function"].get("description", "")
            schema["function"]["description"] = f"[推荐] {desc}"
        elif tool_name in alternative:
            schema = copy.deepcopy(schema)
            desc = schema["function"].get("description", "")
            schema["function"]["description"] = f"[备选] {desc}"

        annotated.append(schema)

    return annotated


def _extract_tool_name(func_name: str) -> str:
    """从函数名中提取工具名。

    函数名格式为 tool_name_action_name，但工具名本身可能包含下划线
    （如 browser_use, app_control, voice_input 等）。
    """
    # 已知的多下划线工具名前缀
    known_prefixes = [
        "mcp_browserbase",
        "browser_use", "app_control", "voice_input", "voice_output",
        "datetime_tool", "chat_history", "doc_generator",
        "image_generator", "python_runner", "tool_info",
        "knowledge_rag", "batch_paper_analyzer", "user_profile",
        "course_schedule", "mind_map", "resume_builder", "data_processor",
        "coding_assistant", "data_visualization", "speech_to_text",
        "format_converter", "id_photo", "literature_search", "pdf_tool", "pdf_generator",
        "gif_maker", "ai_writer", "education_tool", "ppt_generator",
        "financial_report", "contract_generator", "family_member", "meal_menu",
        "document_scanner",  # 高拍仪文档扫描
        "music_player",  # 歌曲库
        "english_conversation",  # 英语口语练习
        "family_milestone",  # 家庭大事记
        "todo",  # 待办事项
        "remote_file_share",  # 远程文件分享
        "daily_task",  # 每日任务
        "medication",  # 用药管理
        "family_album",  # 家庭相册
        "exercise_plan",  # 运动计划
        "social_circle",  # 社交圈
        "storage_organizer",  # 收纳管理
        "insurance",  # 保险方案
        "media_capture",  # 媒体捕获
        "experience_recall",  # 经验回忆
        "poetry",  # 古诗词库
        "stock_query",  # 股票查询
        "quant_trading",  # 量化交易
        "system_monitor",  # 系统监控
        # 学术工具
        "research_landscape",
        "research_lineage",
        "idea_migration",
        "stock_photo",
        "methodology_deconstructor",
        "citation_storyteller",
        "journal_intelligence",
        "contrarian_finder",
        "paper_lifecycle",
        "qualitative_analysis",  # 定性分析（Nvivo风格）
        "local_paper_search",
        "oss_pdf_search",      # OSS PDF搜索（分层下载）
        "oss_pdf_download",    # OSS PDF下载（分层下载）
        "crawlee_tool",  # 批量网页爬取
        # 自我反思工具（P0/P1/P2）
        "tool_audit",     # 工具调用审计
        "log_viewer",     # 日志查看器
        "codebase_search",  # 代码库搜索
        "self_control",   # 自我控制
    ]
    for prefix in known_prefixes:
        if func_name.startswith(prefix + "_") or func_name == prefix:
            return prefix

    # 默认：取第一个下划线前的部分
    return func_name.split("_")[0] if "_" in func_name else func_name


# ------------------------------------------------------------------
# 渐进式工具暴露引擎
# ------------------------------------------------------------------

class ToolExposureEngine:
    """工具暴露策略引擎 — 根据意图置信度分层暴露工具。

    三层工具集：
    - 推荐工具集（~10个）：核心工具 + 意图相关工具（高置信度时使用）
    - 扩展工具集（~20个）：推荐 + 扩展工具（中置信度时使用）
    - 全量工具集（35+个）：所有已注册工具（低置信度 / 回退时使用）

    风险控制：
    - 核心工具（shell/file/screen/search）始终保留
    - 连续失败 >= 2 次自动升级到更大工具集
    - 支持配置化开关，可随时关闭
    """

    # 核心工具（始终保留在任何工具集中）
    CORE_TOOLS: set[str] = {"shell", "file", "screen", "search"}

    # 扩展工具（中置信度时额外包含）
    EXTENDED_TOOLS: set[str] = {
        "browser", "browser_use", "notify", "clipboard",
        "app_control", "calculator", "datetime_tool",
        "family_album",  # 家庭相册
        "stock_query",  # 股票行情
        "crawlee_tool",  # 批量网页爬取
        "wechat",             # v4.x: 微信消息
        "remote_file_share",  # v4.x: 远程文件分享
        # 自我反思工具（归入 recommended 层级）
        "tool_audit", "log_viewer", "codebase_search", "self_control",
        "experience_recall",  # 经验回忆
    }

    # 工具集层级
    TIER_RECOMMENDED = "recommended"  # 推荐集
    TIER_EXTENDED = "extended"        # 扩展集
    TIER_FULL = "full"                # 全量集

    def __init__(
        self,
        tool_registry: Any,
        *,
        enabled: bool = True,
        enable_annotation: bool = True,
        failures_to_upgrade: int = 2,
    ) -> None:
        """初始化。

        Args:
            tool_registry: ToolRegistry 实例
            enabled: 是否启用渐进式暴露（False 则始终全量）
            enable_annotation: 是否启用 Schema 优先级标注
            failures_to_upgrade: 连续失败多少次后自动升级工具集
        """
        self._registry = tool_registry
        self._enabled = enabled
        self._enable_annotation = enable_annotation
        self._failures_to_upgrade = failures_to_upgrade

        # 运行时状态（每次对话重置）
        self._consecutive_failures: int = 0
        self._forced_tier: str | None = None  # 被强制升级的层级
        self._deviation_count: int = 0  # 工具调用偏离次数（被前置验证拒绝）

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def get_schemas(self, intent_result: IntentResult) -> list[dict[str, Any]]:
        """根据意图结果返回分层 Schema。

        Args:
            intent_result: detect_intent_with_confidence 的返回值

        Returns:
            function calling schema 列表
        """
        if not self._enabled:
            schemas = self._registry.get_all_schemas()
            if self._enable_annotation:
                schemas = annotate_schema_priority(schemas, intent_result)
            return schemas

        tier = self._determine_tier(intent_result)
        tool_names = self._get_tool_names_for_tier(tier, intent_result)

        # 依赖解析
        tool_names = self._resolve_dependencies(tool_names)

        logger.info(
            "工具暴露策略: tier=%s, confidence=%.2f, intent=%s, tools=%d",
            tier, intent_result.confidence, intent_result.primary_intent,
            len(tool_names),
        )

        schemas = self._registry.get_schemas_by_names(tool_names)

        if self._enable_annotation:
            schemas = annotate_schema_priority(schemas, intent_result)

        return schemas

    def report_failure(self) -> tuple[str, str] | None:
        """报告工具调用失败，可能触发自动升级。

        Returns:
            如果发生层级升级，返回 (from_tier, to_tier)；否则返回 None
        """
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failures_to_upgrade:
            return self._upgrade_tier()
        return None

    def report_success(self) -> None:
        """报告工具调用成功，重置连续失败计数。"""
        self._consecutive_failures = 0

    def report_deviation(self) -> None:
        """报告一次工具调用偏离（被前置验证拒绝）。

        连续偏离 >= 2 次时，强制收敛到推荐集，防止模型继续发散。
        偏离收敛优先级 > 失败升级优先级。
        """
        self._deviation_count += 1
        if self._deviation_count >= 2:
            # 强制收敛到推荐集
            self._forced_tier = self.TIER_RECOMMENDED
            logger.warning(
                "连续偏离 %d 次，工具集强制收敛到推荐集",
                self._deviation_count,
            )

    def reset(self) -> None:
        """重置状态（新对话开始时调用）。"""
        self._consecutive_failures = 0
        self._forced_tier = None
        self._deviation_count = 0

    @property
    def current_tier(self) -> str:
        """当前生效的工具集层级（调试用）。"""
        return self._forced_tier or "auto"

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _determine_tier(self, intent_result: IntentResult) -> str:
        """根据意图置信度和运行状态确定工具集层级。"""
        # 强制层级优先
        if self._forced_tier:
            return self._forced_tier

        confidence = intent_result.confidence

        if confidence >= 0.8:
            return self.TIER_RECOMMENDED
        elif confidence >= 0.5:
            return self.TIER_EXTENDED
        else:
            return self.TIER_FULL

    def _get_tool_names_for_tier(
        self,
        tier: str,
        intent_result: IntentResult,
    ) -> set[str]:
        """获取指定层级的工具名称集合。"""
        # 核心工具始终包含
        tools = set(self.CORE_TOOLS)

        if tier == self.TIER_FULL:
            # 全量：返回所有已注册工具（含懒加载未实例化的）
            all_tools = self._registry.list_all_tool_names()
            return all_tools

        # 意图相关工具
        for intent_name in intent_result.intents:
            intent_tools = INTENT_TOOL_MAPPING.get(intent_name, [])
            tools.update(intent_tools)

        if tier == self.TIER_EXTENDED:
            # 扩展集额外包含通用扩展工具
            tools.update(self.EXTENDED_TOOLS)

        # 始终包含工具信息查询
        tools.add("tool_info")

        return tools

    def _resolve_dependencies(self, tool_names: set[str]) -> set[str]:
        """自动包含依赖工具（避免过滤掉内容源工具）。

        读取 tools.json 中的 dependencies.input_sources 字段。
        """
        all_tools = set(tool_names)
        for name in list(tool_names):
            cfg = self._registry.get_tool_config(name)
            deps = cfg.get("dependencies", {})
            # 兼容两种格式：列表 ["fredapi"] 或字典 {"input_sources": [...]}
            if isinstance(deps, dict):
                input_sources = deps.get("input_sources", [])
                all_tools.update(input_sources)
        return all_tools

    def _upgrade_tier(self) -> tuple[str, str] | None:
        """自动升级工具集层级。

        Returns:
            如果升级成功，返回 (from_tier, to_tier)；已经是最大集则返回 None
        """
        current = self._forced_tier or self.TIER_RECOMMENDED

        if current == self.TIER_RECOMMENDED:
            self._forced_tier = self.TIER_EXTENDED
            logger.warning(
                "连续失败 %d 次，工具集自动升级: %s → %s",
                self._consecutive_failures, current, self._forced_tier,
            )
            return (current, self._forced_tier)
        elif current == self.TIER_EXTENDED:
            self._forced_tier = self.TIER_FULL
            logger.warning(
                "连续失败 %d 次，工具集自动升级: %s → %s",
                self._consecutive_failures, current, self._forced_tier,
            )
            return (current, self._forced_tier)
        # TIER_FULL 已经是最大集，无需升级
        return None
