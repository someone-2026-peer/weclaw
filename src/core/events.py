"""核心事件类型定义 — Agent 推理全流程的事件枚举与数据结构。

所有事件通过 EventBus 广播，UI/审计/插件通过订阅这些事件来响应。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =====================================================================
# 事件类型常量
# =====================================================================

class EventType:
    """事件类型常量。使用字符串便于扩展和调试。"""

    # --- 用户交互 ---
    USER_INPUT = "user_input"           # 用户输入了消息
    USER_COMMAND = "user_command"        # 用户输入了 /命令

    # --- Agent 推理 ---
    AGENT_THINKING = "agent_thinking"   # Agent 开始思考（模型调用前）
    AGENT_RESPONSE = "agent_response"   # Agent 给出最终回复
    AGENT_STEP = "agent_step"           # Agent 完成一步推理
    AGENT_ERROR = "agent_error"         # Agent 推理过程出错

    # --- 工具调用 ---
    TOOL_CALL = "tool_call"             # 即将调用工具
    TOOL_RESULT = "tool_result"         # 工具返回结果
    TOOL_ERROR = "tool_error"           # 工具执行出错

    # --- 模型 ---
    MODEL_CALL = "model_call"           # 即将调用模型 API
    MODEL_RESPONSE = "model_response"   # 模型 API 返回
    MODEL_ERROR = "model_error"         # 模型 API 出错
    MODEL_USAGE = "model_usage"         # token 用量更新
    MODEL_STREAM_CHUNK = "model_stream_chunk"  # 流式文本片段
    MODEL_REASONING = "model_reasoning"  # 模型思考过程（reasoning_content）

    # --- 会话 ---
    SESSION_CREATED = "session_created"   # 新会话创建
    SESSION_SWITCHED = "session_switched" # 切换了会话
    SESSION_CLEARED = "session_cleared"   # 会话历史被清除

    # --- 文件生成 ---
    FILE_GENERATED = "file_generated"   # 工具生成了文件

    # --- 定时任务 ---
    CRON_JOB_STARTED = "cron_job_started"     # 定时任务开始执行
    CRON_JOB_FINISHED = "cron_job_finished"   # 定时任务执行完成
    CRON_JOB_ERROR = "cron_job_error"         # 定时任务执行失败

    # --- 系统 ---
    APP_STARTED = "app_started"         # 应用启动完成
    APP_SHUTDOWN = "app_shutdown"        # 应用即将关闭
    CONFIG_CHANGED = "config_changed"   # 配置项被修改

    # --- 主动陪伴 ---
    COMPANION_CARE_TRIGGERED = "companion:care_triggered"
    COMPANION_CARE_COMPLETED = "companion:care_completed"
    COMPANION_PROFILE_UPDATED = "companion:profile_updated"
    COMPANION_MOOD_DETECTED = "companion:mood_detected"

    # --- CFTA: 异步工具执行 (Chat-First, Tools-Async) ---
    DEFERRED_TOOL_STARTED = "deferred_tool_started"    # 后台异步工具开始
    DEFERRED_TOOL_RESULT = "deferred_tool_result"      # 后台异步工具结果


# =====================================================================
# 事件数据结构
# =====================================================================

@dataclass
class UserInputEvent:
    """用户输入事件数据。"""
    text: str
    session_id: str = ""


@dataclass
class AgentThinkingEvent:
    """Agent 开始思考事件数据。"""
    step: int  # 当前步骤编号（从1开始）
    max_steps: int
    model_key: str = ""
    session_id: str = ""


@dataclass
class AgentResponseEvent:
    """Agent 最终回复事件数据。"""
    content: str
    total_steps: int = 0
    total_tokens: int = 0
    tool_calls_count: int = 0
    session_id: str = ""


@dataclass
class AgentStepEvent:
    """Agent 单步推理事件数据。"""
    step: int
    step_type: str  # "tool_call" | "tool_result" | "response"
    content: str = ""
    session_id: str = ""


@dataclass
class ToolCallEvent:
    """工具调用事件数据。"""
    tool_name: str
    action_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    function_name: str = ""  # 完整函数名如 "shell_run"
    session_id: str = ""


@dataclass
class ToolResultEvent:
    """工具结果事件数据。"""
    tool_name: str
    action_name: str
    status: str = ""  # "success" | "error" | "timeout" | "denied"
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0
    session_id: str = ""
    html_image: str = ""  # 用于 GUI 直接显示的图片 HTML


@dataclass
class ModelCallEvent:
    """模型调用事件数据。"""
    model_key: str
    model_id: str = ""
    message_count: int = 0  # 消息数量
    has_tools: bool = False
    session_id: str = ""


@dataclass
class ModelResponseEvent:
    """模型响应事件数据。"""
    model_key: str
    has_tool_calls: bool = False
    content_preview: str = ""  # 回复内容前100字符
    session_id: str = ""


@dataclass
class ModelUsageEvent:
    """模型用量事件数据。"""
    model_key: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    session_id: str = ""


@dataclass
class ModelReasoningEvent:
    """模型思考过程事件数据（用于分离reasoning_content）。"""
    reasoning: str  # 思考内容
    is_delta: bool = True  # 是否是增量内容（流式）
    is_complete: bool = False  # 是否已完成
    session_id: str = ""


@dataclass
class SessionEvent:
    """会话事件数据。"""
    session_id: str
    action: str = ""  # "created" | "switched" | "cleared"
    message_count: int = 0


@dataclass
class ErrorEvent:
    """错误事件数据。"""
    source: str  # 错误来源: "agent" | "tool" | "model" | "system"
    message: str
    error_type: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""


@dataclass
class FileGeneratedEvent:
    """文件生成事件数据。"""
    file_path: str               # 文件绝对路径
    file_name: str = ""          # 文件名
    source_tool: str = ""        # 来源工具名
    source_action: str = ""      # 来源动作名
    file_size: int = 0           # 文件大小
    session_id: str = ""


@dataclass
class CronJobEvent:
    """定时任务事件数据。"""
    job_id: str                  # 任务ID
    job_type: str = "command"    # 任务类型: command/ai_task
    description: str = ""        # 任务描述
    status: str = "started"      # 状态: started/finished/error
    result: str = ""             # 执行结果（finished时）
    error: str = ""              # 错误信息（error时）
    duration_ms: float = 0.0     # 执行时长（毫秒）


@dataclass
class CompanionCareEvent:
    """主动陪伴事件数据。"""
    topic_id: str                # 关怀主题ID
    message: str = ""            # 关怀消息内容
    trigger_type: str = "scheduled"  # scheduled/contextual/reactive
    interaction_type: str = "text"   # text/voice
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompanionMoodEvent:
    """情绪检测事件数据。"""
    mood: str = "neutral"        # positive/negative/neutral
    sub_mood: str = ""           # stressed/tired/...
    confidence: float = 0.5
    source_text: str = ""


@dataclass
class DeferredToolStartedEvent:
    """CFTA: 后台异步工具开始事件。"""
    user_input: str              # 原始用户输入
    session_id: str = ""


@dataclass
class DeferredToolResultEvent:
    """CFTA: 后台异步工具执行结果事件。"""
    result_summary: str          # 工具执行结果摘要
    tool_names: list[str] = field(default_factory=list)  # 执行过的工具名列表
    session_id: str = ""
