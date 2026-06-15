#!/usr/bin/env python3
"""Experiment 1: PTE-FD Ablation Study (Table 3)

Measures tool selection accuracy, token reduction, and recovery rate
across 5 configurations: Static / PTE / PTE-FD / ITR / PTE-FD+ITR

Uses DeepSeek V4 Flash API for actual LLM tool selection calls.
"""

import json
import os
import sys
import time
import random
import hashlib
from pathlib import Path
from typing import Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# OpenAI client
from openai import OpenAI

# WeClaw modules
from src.core.prompts import (
    detect_intent_with_confidence,
    INTENT_CATEGORIES,
    INTENT_TOOL_MAPPING,
    INTENT_PRIORITY_MAP,
    IntentResult,
)
from src.core.tool_exposure import ToolExposureEngine, _extract_tool_name
from benchmark_utils import get_output_dir

# ---------- Constants ----------

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"  # DeepSeek V4 Flash endpoint
OUTPUT_DIR = get_output_dir(__file__)

# Number of queries per intent category
QUERIES_PER_INTENT = 33
TOTAL_QUERIES = 200
RANDOM_SEED = 42

# ---------- Schema Builder (mock ToolRegistry) ----------

# Tool action definitions: tool_name -> [(action, description, params)]
TOOL_ACTIONS: dict[str, list[tuple[str, str, dict]]] = {}

# Core tools
TOOL_ACTIONS["shell"] = [("run", "🖥️ Execute shell command", {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command"}}, "required": ["command"]})]
TOOL_ACTIONS["file"] = [
    ("read", "📄 Read file content", {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}),
    ("write", "📄 Write file content", {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}),
]
TOOL_ACTIONS["screen"] = [("capture", "📸 Take screenshot", {"type": "object", "properties": {"region": {"type": "string"}}, "required": []})]
TOOL_ACTIONS["search"] = [
    ("web_search", "🔍 Search the web", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
]
# Extended tools
TOOL_ACTIONS["browser"] = [
    ("open_url", "🌐 Open URL in browser", {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}),
    ("click", "🌐 Click element", {"type": "object", "properties": {"selector": {"type": "string"}}, "required": ["selector"]}),
]
TOOL_ACTIONS["browser_use"] = [("execute", "🤖 AI browser automation", {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]})]
TOOL_ACTIONS["notify"] = [("send", "🔔 Send notification", {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]})]
TOOL_ACTIONS["clipboard"] = [("read", "📋 Read clipboard", {"type": "object", "properties": {}, "required": []}), ("write", "📋 Write clipboard", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})]
TOOL_ACTIONS["app_control"] = [("launch", "🚀 Launch application", {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]})]
TOOL_ACTIONS["calculator"] = [("calculate", "🧮 Calculate expression", {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]})]
TOOL_ACTIONS["datetime_tool"] = [("get_time", "🕐 Get current time", {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": []})]
TOOL_ACTIONS["family_album"] = [("list", "📷 List album photos", {"type": "object", "properties": {"album": {"type": "string"}}, "required": []})]
TOOL_ACTIONS["stock_query"] = [("query", "📈 Query stock price", {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]})]
TOOL_ACTIONS["crawlee_tool"] = [("crawl", "🕷️ Crawl web pages", {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]})]
TOOL_ACTIONS["tool_audit"] = [("report", "🔍 Tool audit report", {"type": "object", "properties": {"period": {"type": "string"}}, "required": []})]
TOOL_ACTIONS["log_viewer"] = [("view", "📋 View logs", {"type": "object", "properties": {"level": {"type": "string"}}, "required": []})]
TOOL_ACTIONS["codebase_search"] = [("search", "🔎 Search codebase", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
TOOL_ACTIONS["self_control"] = [("status", "🎛️ Self control status", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["experience_recall"] = [("recall", "💡 Recall experiences", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
TOOL_ACTIONS["tool_info"] = [("list", "ℹ️ List available tools", {"type": "object", "properties": {}, "required": []})]
# Additional tools with descriptive schemas for better LLM discrimination
TOOL_DESCRIPTIONS = {
    "doc_generator": "生成Word/DOCX文档，支持报告、文章、总结等文档创建",
    "image_generator": "AI绘图工具，根据文字描述生成图片",
    "weather": "查询天气预报，支持全球城市当前天气和未来预报",
    "mcp_browserbase": "MCP云端浏览器，远程浏览器自动化操作",
    "mcp_browserbase-csdn": "MCP CSDN博客发布工具，写博客和发布文章",
    "knowledge_rag": "知识库RAG检索，搜索向量化文档和知识库内容",
    "batch_paper_analyzer": "批量论文分析工具，支持摘要提取和主题分类",
    "python_runner": "Python代码执行器，运行Python脚本和表达式",
    "literature_search": "学术文献检索，搜索论文数据库和学术资源",
    "poetry": "古诗词库，查询唐诗宋词等古典诗词",
    "literature_review": "文献综述生成工具，自动整理研究现状",
    "chat_history": "聊天历史检索，搜索之前的对话记录",
    "diary": "日记管理工具，记录和管理个人日记",
    "finance": "个人财务管理，记账、支出、收入统计分析",
    "health": "健康数据管理，体重、血压、心率等健康指标记录",
    "medication": "用药管理，服药提醒和药物记录",
    "user_profile": "用户资料管理，个人信息和偏好设置",
    "family_member": "家庭成员管理，联系人和家庭关系",
    "course_schedule": "课程表管理，学校课程安排和时间表",
    "meal_menu": "食谱菜单管理，学校食谱和家庭饮食计划",
    "family_milestone": "家庭大事记，纪念日和重要事件记录",
    "todo": "待办事项管理，任务清单和进度跟踪",
    "daily_task": "每日任务管理，日常工作和例行任务",
    "family_album": "家庭相册管理，照片集管理和浏览",
    "fitness_nutrition": "健身营养管理，运动计划和营养方案",
    "music_player": "音乐播放器，歌曲库管理和播放控制",
    "cron": "定时任务管理，设置定期执行的自动化任务",
    "fred_query": "FRED经济数据查询，GDP/CPI/失业率等宏观经济指标",
    "quant_trading": "量化交易分析，策略回测和交易信号生成",
    "email": "邮件管理，发送和接收电子邮件",
    "voice_input": "语音输入，将语音转为文字",
    "voice_output": "语音输出，文字转语音朗读",
    "ocr": "OCR文字识别，从图片中提取文字内容",
    "speech_to_text": "语音转文字，音频文件转录和字幕生成",
    "document_scanner": "文档扫描仪，高拍仪扫描和试卷解析",
    "media_capture": "媒体捕获，摄像头拍照和录像",
    "stock_photo": "图库搜索，搜索和下载免费图片素材",
    "concept_diagrams": "概念图生成，创建SVG流程图和架构图",
    "meme_generation": "Meme生成器，创建网络梗图和表情包",
    "wechat": "微信消息发送",
    "remote_file_share": "远程文件分享，跨设备传输文件",
    "duckduckgo_search": "DuckDuckGo网络搜索引擎",
    "pdf_tool": "PDF工具，合并/拆分/加密/解密PDF文件",
    "format_converter": "文档格式转换，支持PDF/DOCX/MD等格式互转",
    "ppt_generator": "PPT演示文稿生成，创建幻灯片和演示文件",
    "pdf_generator": "PDF生成器，从内容生成PDF文档",
    "data_processor": "数据处理工具，CSV/Excel数据清洗、转换和统计分析",
    "data_visualization": "数据可视化，生成柱状图/折线图/饼图等图表",
    "statistics": "统计分析工具，数学统计和数据分析",
    "ai_writer": "AI写作助手，文章创作和内容生成",
    "contract_generator": "合同生成器，创建法律合同和协议文档",
    "resume_builder": "简历生成器，创建专业简历",
    "system_monitor": "系统监控，查看CPU/内存/磁盘/网络使用率和系统状态",
    "id_photo": "证件照生成，制作标准尺寸证件照",
    "mind_map": "思维导图生成，创建知识结构和思维图",
    "coding_assistant": "编程助手，代码补全和调试帮助",
}

for _tname, _desc in TOOL_DESCRIPTIONS.items():
    if _tname not in TOOL_ACTIONS:
        TOOL_ACTIONS[_tname] = [
            ("execute", _desc,
             {"type": "object", "properties": {"input": {"type": "string", "description": "输入内容"}}, "required": ["input"]})
        ]


def build_schemas(tool_names: set[str]) -> list[dict]:
    """Build OpenAI function calling schemas for given tool names."""
    schemas = []
    for tname in sorted(tool_names):
        actions = TOOL_ACTIONS.get(tname, [])
        for action_name, desc, params in actions:
            schemas.append({
                "type": "function",
                "function": {
                    "name": f"{tname}_{action_name}",
                    "description": desc,
                    "parameters": params,
                },
            })
    return schemas


def get_all_tool_names() -> set[str]:
    """Return all tool names (full tier)."""
    return set(TOOL_ACTIONS.keys())


# ---------- Mock ToolRegistry ----------

class MockToolRegistry:
    """Mock ToolRegistry for ToolExposureEngine."""
    def __init__(self):
        self._all_schemas = build_schemas(get_all_tool_names())
        self._schemas_by_name: dict[str, list[dict]] = {}
        for s in self._all_schemas:
            fn = s["function"]["name"]
            tname = _extract_tool_name(fn)
            self._schemas_by_name.setdefault(tname, []).append(s)

    def get_all_schemas(self) -> list[dict]:
        return list(self._all_schemas)

    def get_schemas_by_names(self, tool_names: set[str]) -> list[dict]:
        result = []
        for tname in tool_names:
            result.extend(self._schemas_by_name.get(tname, []))
        return result

    def list_all_tool_names(self) -> set[str]:
        return get_all_tool_names()

    def get_tool_config(self, name: str) -> dict:
        return {}  # No dependencies


# ---------- Query Generation ----------

# Ground-truth mapping: intent -> SET of acceptable tools
# Any tool in the set counts as correct (from INTENT_PRIORITY_MAP recommended+alternative)
INTENT_GROUND_TRUTH: dict[str, set[str]] = {
    "browser_automation": {"browser", "browser_use", "mcp_browserbase", "mcp_browserbase-csdn"},
    "file_operation": {"file", "shell"},
    "document_assembly": {"doc_generator", "image_generator", "weather", "file", "search", "shell"},
    "system_admin": {"shell", "screen", "app_control", "clipboard", "notify"},
    "system_monitoring": {"system_monitor", "shell", "app_control"},
    "daily_assistant": {"weather", "datetime_tool", "calculator", "cron", "search", "statistics"},
    "knowledge": {"knowledge_rag", "batch_paper_analyzer", "poetry", "file", "search", "python_runner", "literature_search", "literature_review"},
    "life_management": {"diary", "finance", "health", "medication", "user_profile", "family_member", "course_schedule", "meal_menu", "family_milestone", "todo", "daily_task", "family_album", "fitness_nutrition", "music_player", "cron"},
    "financial_activity": {"stock_query", "fred_query", "quant_trading", "search", "shell"},
    "multimedia": {"voice_input", "voice_output", "ocr", "speech_to_text", "document_scanner", "music_player", "media_capture", "stock_photo", "image_generator", "concept_diagrams", "meme_generation", "screen"},
    "document_processing": {"pdf_tool", "format_converter", "ppt_generator", "pdf_generator", "stock_photo"},
    "data_analysis": {"data_processor", "data_visualization", "statistics", "shell", "file"},
}

# Query templates per intent
QUERY_TEMPLATES: dict[str, list[str]] = {
    "browser_automation": [
        "打开网页 https://example.com 并截图",
        "帮我访问这个网站并获取内容",
        "在浏览器中登录网站",
        "打开链接并截取页面截图",
        "帮我操作网页上的表单",
        "访问网站并下载文件",
        "在浏览器中搜索信息",
        "打开URL并分析页面内容",
        "帮我自动化网页操作",
        "登录网站并提取数据",
    ],
    "file_operation": [
        "读取文件 content.txt 的内容",
        "帮我整理文件夹中的文件",
        "复制文件到另一个目录",
        "重命名文件 report.docx",
        "解压压缩文件 archive.zip",
        "创建新的目录结构",
        "移动文件到指定位置",
        "查看文件内容并总结",
        "批量重命名文件夹中的文件",
        "搜索文件中的关键词",
    ],
    "document_assembly": [
        "生成一份Word文档报告",
        "创建文档包含这些数据",
        "制作文档整合本周内容",
        "生成一份带图表的报告文档",
        "组装文档包含摘要和分析",
        "帮我写一份项目文档",
        "创建月度报告文档",
        "生成包含图片的文档",
        "制作工作总结文档",
        "整合资料生成报告",
    ],
    "system_admin": [
        "查看系统进程列表",
        "截屏当前桌面",
        "清理系统临时文件",
        "查看系统服务状态",
        "管理系统启动项",
        "截图并保存到桌面",
        "检查磁盘空间使用情况",
        "查看系统性能信息",
        "管理注册表项",
        "关机倒计时设置",
    ],
    "system_monitoring": [
        "查看CPU和内存使用率",
        "监控系统状态",
        "检查电脑性能",
        "查看网络使用情况",
        "电脑运行很慢帮我诊断",
        "查看电池健康状态",
        "监控磁盘读写速度",
        "查看系统温度信息",
        "分析资源占用情况",
        "查看系统详细信息",
    ],
    "daily_assistant": [
        "今天天气怎么样",
        "帮我计算一下汇率",
        "设置定时提醒",
        "查看今天的日程安排",
        "计算这组数据的平均值",
        "设置每天早上的闹钟",
        "帮我算一下税费",
        "查询今天的时间和日期",
        "设置定时任务每天执行",
        "计算折扣后的价格",
    ],
    "knowledge": [
        "搜索知识库中的论文",
        "查找关于AI的文献资料",
        "帮我分析这批论文",
        "查询诗词相关的内容",
        "搜索RAG知识库",
        "找一首李白的诗",
        "索引这些文档到知识库",
        "推荐一些经典诗词",
        "批量分析论文摘要",
        "搜索向量数据库中的内容",
    ],
    "life_management": [
        "记录今天的日记",
        "查看我的待办事项",
        "管理家庭成员信息",
        "记录今天的支出",
        "查看课程表安排",
        "设置每日任务清单",
        "管理相册中的照片",
        "查看家庭食谱",
        "记录体重变化",
        "管理家庭大事记",
    ],
    "financial_activity": [
        "查询贵州茅台股价",
        "分析股票基本面数据",
        "查看实时行情数据",
        "查询GDP经济数据",
        "分析量化交易信号",
        "查看股票K线走势",
        "查询通胀率CPI数据",
        "分析投资组合收益",
        "查看基金净值",
        "查询美联储利率",
    ],
    "multimedia": [
        "识别图片中的文字",
        "语音转成文字",
        "录制一段音频",
        "拍照并保存",
        "搜索相关图片素材",
        "扫描文档并识别",
        "播放音乐",
        "录制视频",
        "截取屏幕并OCR识别",
        "生成一张概念图",
    ],
    "document_processing": [
        "合并两个PDF文件",
        "把Word转换成PDF",
        "生成PPT演示文稿",
        "拆分PDF文件",
        "格式转换docx到pdf",
        "加密PDF文档",
        "导出为PPT格式",
        "解密受保护的PDF",
        "转换文档格式",
        "生成幻灯片演示",
    ],
    "data_analysis": [
        "分析CSV数据文件",
        "生成数据可视化图表",
        "处理Excel表格数据",
        "统计分析这组数据",
        "生成柱状图和折线图",
        "清洗数据中的异常值",
        "数据透视表分析",
        "生成数据分析报告",
        "对比两组数据的差异",
        "数据分组聚合计算",
    ],
}


def generate_test_queries(n: int = TOTAL_QUERIES, seed: int = RANDOM_SEED) -> list[dict]:
    """Generate test queries with ground-truth tool labels."""
    rng = random.Random(seed)
    queries = []

    # Use the 12 intents with ground-truth mappings
    intents = list(INTENT_GROUND_TRUTH.keys())
    per_intent = max(1, n // len(intents))  # ~16-17 per intent

    for intent in intents:
        templates = QUERY_TEMPLATES.get(intent, [])
        # Extend templates if needed
        extended = list(templates)
        while len(extended) < per_intent:
            base = rng.choice(templates)
            # Add variation
            suffix = rng.choice(["，请帮我处理", "，谢谢", "，尽快完成", "，详细一点", ""])
            extended.append(base + suffix)

        selected = rng.sample(extended, min(per_intent, len(extended)))
        gt_tools = INTENT_GROUND_TRUTH[intent]  # SET of acceptable tools

        for q in selected:
            queries.append({
                "query": q,
                "ground_truth_tools": gt_tools,  # set
                "intent": intent,
            })

    # Shuffle and limit to n
    rng.shuffle(queries)
    return queries[:n]


# ---------- ITR (Information-Theoretic Retrieval) ----------

# Chinese keyword phrases for semantic matching
_ITR_KEYWORDS = {
    "browser": ["网页", "浏览器", "打开网页", "网站", "URL", "访问", "登录网站", "网页操作"],
    "browser_use": ["自动化", "智能浏览", "网页操作", "自动"],
    "file": ["文件", "读文件", "写文件", "目录", "文件夹", "复制", "移动", "重命名"],
    "shell": ["命令", "执行", "运行", "进程", "系统", "脚本", "PowerShell"],
    "screen": ["截图", "截屏", "屏幕", "桌面"],
    "search": ["搜索", "查找", "搜索网页", "检索"],
    "calculator": ["计算", "算术", "汇率", "数学"],
    "datetime_tool": ["时间", "日期", "日程", "提醒", "闹钟"],
    "weather": ["天气", "气温", "下雨"],
    "stock_query": ["股票", "股价", "行情", "K线", "投资", "基金"],
    "fred_query": ["GDP", "经济数据", "通胀", "CPI", "利率", "失业率", "宏观经济"],
    "quant_trading": ["量化", "交易", "选股", "回测", "策略"],
    "ocr": ["识别", "OCR", "图片文字", "文字识别", "扫描"],
    "data_processor": ["数据", "分析", "CSV", "Excel", "处理数据", "统计"],
    "data_visualization": ["图表", "可视化", "柱状图", "折线图", "饼图"],
    "pdf_tool": ["PDF", "合并PDF", "拆分", "加密", "解密"],
    "format_converter": ["格式转换", "转换", "docx转pdf", "导出"],
    "ppt_generator": ["PPT", "幻灯片", "演示文稿"],
    "doc_generator": ["文档", "报告", "Word", "生成文档"],
    "knowledge_rag": ["知识库", "RAG", "文档搜索", "索引"],
    "system_monitor": ["CPU", "内存", "磁盘", "性能", "监控", "电脑慢", "使用率"],
    "diary": ["日记", "记录"],
    "finance": ["记账", "支出", "收入", "财务"],
    "todo": ["待办", "任务", "待办事项"],
    "music_player": ["音乐", "播放", "歌曲", "听歌"],
    "image_generator": ["生成图片", "AI绘图", "画图"],
    "voice_input": ["语音", "录音", "语音转文字"],
    "speech_to_text": ["转录", "字幕", "转文字"],
    "media_capture": ["拍照", "录像", "摄像头"],
    "poetry": ["诗词", "古诗", "诗歌"],
    "email": ["邮件", "发邮件", "收邮件"],
    "cron": ["定时", "定期", "自动执行"],
}


def itr_select_tools(query: str, all_schemas: list[dict], top_k: int = 10) -> list[dict]:
    """ITR: word/phrase-level keyword matching for tool retrieval."""
    query_lower = query.lower()

    # Extract tool names from schemas
    schema_tool_names: dict[str, str] = {}  # func_name -> tool_name
    for s in all_schemas:
        fn = s["function"]["name"]
        tname = _extract_tool_name(fn)
        schema_tool_names[fn] = tname

    scored = []
    for schema in all_schemas:
        fn = schema["function"]["name"]
        tname = schema_tool_names[fn]
        desc = schema["function"]["description"].lower()
        score = 0.0

        # Keyword matching against ITR keyword dictionary
        keywords = _ITR_KEYWORDS.get(tname, [])
        for kw in keywords:
            if kw.lower() in query_lower:
                score += 3.0  # Strong match for domain keywords

        # Name-based matching (word-level, not character-level)
        name_parts = tname.replace("_", " ").split()
        for part in name_parts:
            if len(part) > 2 and part in query_lower:
                score += 1.0

        # Description keyword matching
        desc_words = ["文件", "浏览器", "截图", "搜索", "计算", "股票", "PDF",
                      "文档", "图片", "音乐", "视频", "系统", "数据", "分析",
                      "报告", "命令", "执行", "时间", "天气", "识别"]
        for w in desc_words:
            if w in query_lower and w in desc:
                score += 2.0

        scored.append((score, fn, schema))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, _, s in scored[:top_k]]


# ---------- LLM Tool Selection ----------

def call_llm_select_tool(client: OpenAI, query: str, schemas: list[dict]) -> tuple[str, int]:
    """Call LLM to select a tool. Returns (selected_tool_name, prompt_tokens)."""
    if not schemas:
        return ("", 0)

    system_prompt = (
        "你是 WeClaw AI 助手的工具选择模块。根据用户请求，选择最合适的工具来完成任务。"
        "你必须从提供的工具列表中选择一个工具。只选择工具，不要执行。"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            tools=schemas,
            tool_choice="auto",
            max_tokens=100,
            temperature=0.0,
        )
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0

        msg = response.choices[0].message
        if msg.tool_calls:
            func_name = msg.tool_calls[0].function.name
            tool_name = _extract_tool_name(func_name)
            return (tool_name, prompt_tokens)

        return ("", prompt_tokens)

    except Exception as e:
        print(f"  LLM call error: {e}")
        return ("", 0)


# ---------- Experiment Runner ----------

def run_experiment():
    """Run the full PTE-FD ablation experiment."""
    print("=" * 60)
    print("Experiment 1: PTE-FD Ablation Study")
    print("=" * 60)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    registry = MockToolRegistry()

    # Generate test queries
    queries = generate_test_queries()
    print(f"Generated {len(queries)} test queries across {len(set(q['intent'] for q in queries))} intents")

    # Build static schemas (all tools)
    all_tool_names = get_all_tool_names()
    static_schemas = build_schemas(all_tool_names)
    static_schema_bytes = len(json.dumps(static_schemas, ensure_ascii=False).encode("utf-8"))
    print(f"Static schemas: {len(static_schemas)} functions, {static_schema_bytes} bytes")

    # Initialize engine
    engine = ToolExposureEngine(registry, enabled=True, enable_annotation=True, failures_to_upgrade=2)

    # Results storage
    results = []
    config_stats = {
        "static": {"correct": 0, "total": 0, "tokens": 0, "schema_bytes": 0},
        "pte": {"correct": 0, "total": 0, "tokens": 0, "schema_bytes": 0},
        "pte_fd": {"correct": 0, "total": 0, "tokens": 0, "recovered": 0, "initial_failures": 0, "schema_bytes": 0},
        "itr": {"correct": 0, "total": 0, "tokens": 0, "schema_bytes": 0},
        "pte_fd_itr": {"correct": 0, "total": 0, "tokens": 0, "recovered": 0, "initial_failures": 0, "schema_bytes": 0},
    }

    for i, q in enumerate(queries):
        query = q["query"]
        gt_tools = q["ground_truth_tools"]  # set of acceptable tools
        intent = q["intent"]

        if (i + 1) % 20 == 0 or i == 0:
            print(f"\n--- Query {i+1}/{len(queries)}: [{intent}] {query[:50]}...")

        # Detect intent
        intent_result = detect_intent_with_confidence(query)

        row = {"index": i, "query": query, "intent": intent, "ground_truth": sorted(gt_tools),
               "detected_intent": intent_result.primary_intent,
               "confidence": intent_result.confidence}

        # === Config 1: Static ===
        engine.reset()
        selected, tokens = call_llm_select_tool(client, query, static_schemas)
        correct = (selected in gt_tools)
        config_stats["static"]["correct"] += int(correct)
        config_stats["static"]["total"] += 1
        config_stats["static"]["tokens"] += tokens
        config_stats["static"]["schema_bytes"] += static_schema_bytes
        row["static"] = {"selected": selected, "correct": correct, "tokens": tokens}

        # === Config 2: PTE (tiered, no escalation) ===
        engine.reset()
        pte_schemas = engine.get_schemas(intent_result)
        pte_bytes = len(json.dumps(pte_schemas, ensure_ascii=False).encode("utf-8"))
        selected, tokens = call_llm_select_tool(client, query, pte_schemas)
        correct = (selected in gt_tools)
        config_stats["pte"]["correct"] += int(correct)
        config_stats["pte"]["total"] += 1
        config_stats["pte"]["tokens"] += tokens
        config_stats["pte"]["schema_bytes"] += pte_bytes
        row["pte"] = {"selected": selected, "correct": correct, "tokens": tokens}

        # === Config 3: PTE-FD (with failure-driven escalation) ===
        engine.reset()
        pte_schemas = engine.get_schemas(intent_result)
        pte_bytes = len(json.dumps(pte_schemas, ensure_ascii=False).encode("utf-8"))
        selected, tokens = call_llm_select_tool(client, query, pte_schemas)
        correct_initial = (selected in gt_tools)
        pte_fd_tokens = tokens

        if not correct_initial:
            # Simulate failure -> escalate
            config_stats["pte_fd"]["initial_failures"] += 1
            engine.report_failure()
            engine.report_failure()  # k=2 triggers upgrade
            escalated_schemas = engine.get_schemas(intent_result)
            esc_bytes = len(json.dumps(escalated_schemas, ensure_ascii=False).encode("utf-8"))
            selected_retry, retry_tokens = call_llm_select_tool(client, query, escalated_schemas)
            pte_fd_tokens += retry_tokens
            correct_retry = (selected_retry in gt_tools)
            if correct_retry:
                config_stats["pte_fd"]["recovered"] += 1
            pte_bytes = max(pte_bytes, esc_bytes)  # Use larger schema size
            row["pte_fd"] = {"selected_initial": selected, "selected_retry": selected_retry,
                             "correct": correct_retry, "tokens": pte_fd_tokens, "escalated": True}
        else:
            engine.report_success()
            row["pte_fd"] = {"selected": selected, "correct": True, "tokens": pte_fd_tokens, "escalated": False}

        config_stats["pte_fd"]["correct"] += int(row["pte_fd"].get("correct", False))
        config_stats["pte_fd"]["total"] += 1
        config_stats["pte_fd"]["tokens"] += pte_fd_tokens
        config_stats["pte_fd"]["schema_bytes"] += pte_bytes

        # === Config 4: ITR (retrieval-based) ===
        itr_schemas = itr_select_tools(query, static_schemas, top_k=10)
        itr_bytes = len(json.dumps(itr_schemas, ensure_ascii=False).encode("utf-8"))
        selected, tokens = call_llm_select_tool(client, query, itr_schemas)
        correct = (selected in gt_tools)
        config_stats["itr"]["correct"] += int(correct)
        config_stats["itr"]["total"] += 1
        config_stats["itr"]["tokens"] += tokens
        config_stats["itr"]["schema_bytes"] += itr_bytes
        row["itr"] = {"selected": selected, "correct": correct, "tokens": tokens}

        # === Config 5: PTE-FD + ITR fallback ===
        engine.reset()
        pte_schemas = engine.get_schemas(intent_result)
        pte_bytes = len(json.dumps(pte_schemas, ensure_ascii=False).encode("utf-8"))
        selected, tokens = call_llm_select_tool(client, query, pte_schemas)
        correct_initial2 = (selected in gt_tools)
        pte_itr_tokens = tokens

        if not correct_initial2:
            config_stats["pte_fd_itr"]["initial_failures"] += 1
            engine.report_failure()
            engine.report_failure()
            escalated_schemas = engine.get_schemas(intent_result)
            # Combine with ITR
            itr_schemas2 = itr_select_tools(query, static_schemas, top_k=10)
            combined_names = set()
            combined = []
            for s in escalated_schemas + itr_schemas2:
                fn = s["function"]["name"]
                if fn not in combined_names:
                    combined_names.add(fn)
                    combined.append(s)
            comb_bytes = len(json.dumps(combined, ensure_ascii=False).encode("utf-8"))
            selected_retry, retry_tokens = call_llm_select_tool(client, query, combined)
            pte_itr_tokens += retry_tokens
            correct_retry = (selected_retry in gt_tools)
            if correct_retry:
                config_stats["pte_fd_itr"]["recovered"] += 1
            pte_bytes = max(pte_bytes, comb_bytes)
            row["pte_fd_itr"] = {"selected_initial": selected, "selected_retry": selected_retry,
                                  "correct": correct_retry, "tokens": pte_itr_tokens, "escalated": True}
        else:
            row["pte_fd_itr"] = {"selected": selected, "correct": True, "tokens": pte_itr_tokens, "escalated": False}

        config_stats["pte_fd_itr"]["correct"] += int(row["pte_fd_itr"].get("correct", False))
        config_stats["pte_fd_itr"]["total"] += 1
        config_stats["pte_fd_itr"]["tokens"] += pte_itr_tokens
        config_stats["pte_fd_itr"]["schema_bytes"] += pte_bytes

        results.append(row)

        # Rate limiting
        time.sleep(0.05)

    # ---------- Compute Summary ----------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    summary = {}
    for cfg, stats in config_stats.items():
        total = stats["total"] or 1
        accuracy = stats["correct"] / total * 100
        token_reduction = (1 - stats["schema_bytes"] / (config_stats["static"]["schema_bytes"] or 1)) * 100
        if cfg == "static":
            token_reduction = 0

        recovery_rate = 0
        if cfg in ("pte", "pte_fd", "pte_fd_itr"):
            init_fail = stats.get("initial_failures", 0)
            recovered = stats.get("recovered", 0)
            # For PTE (no escalation), recovery = the PTE-FD's recovery at PTE tier
            if cfg == "pte":
                # PTE has no escalation, recovery rate = how many would recover via PTE-FD
                recovery_rate = config_stats["pte_fd"].get("recovered", 0) / max(config_stats["pte_fd"].get("initial_failures", 1), 1) * 100
            else:
                recovery_rate = recovered / max(init_fail, 1) * 100

        summary[cfg] = {
            "accuracy": round(accuracy, 1),
            "token_reduction": round(token_reduction, 1),
            "recovery_rate": round(recovery_rate, 1),
            "total_correct": stats["correct"],
            "total_queries": total,
            "total_tokens": stats["tokens"],
            "total_schema_bytes": stats["schema_bytes"],
        }

        print(f"\n{cfg.upper()}:")
        print(f"  Selection Accuracy: {accuracy:.1f}% ({stats['correct']}/{total})")
        print(f"  Token Reduction: {token_reduction:.1f}%")
        if cfg != "static":
            print(f"  Recovery Rate: {recovery_rate:.1f}%")

    # ---------- Save Results ----------
    with open(OUTPUT_DIR / "exp1_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "exp1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / 'exp1_results.json'}")
    print(f"Summary saved to {OUTPUT_DIR / 'exp1_summary.json'}")

    return summary


if __name__ == "__main__":
    run_experiment()
