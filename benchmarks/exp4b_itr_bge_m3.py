#!/usr/bin/env python3
"""Experiment 4b: ITR with Multilingual BGE-m3 (dense + BM25 + cross-encoder)

ESWA fair-baseline variant of exp4. Uses BAAI/bge-m3 (multilingual, 100+
languages) as bi-encoder and BAAI/bge-reranker-v2-m3 as cross-encoder,
replacing the English-only models from the original exp4.

Pipeline:
1. Dense retrieval (BGE-m3 embeddings)
2. BM25 lexical retrieval (rank_bm25)
3. Cross-encoder reranking (BGE-reranker-v2-m3)

Compares ITR-BGE-m3 accuracy/tokens against PTE-FD and Static baselines.
Uses DeepSeek V4 API for actual LLM tool selection calls.

Isolation: This script is a copy of exp4_itr_reproduction.py with modified
model parameters. The original exp4 is untouched.
"""

import json
import os
import sys
import time
import random
import numpy as np
from pathlib import Path
from typing import Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Use HuggingFace mirror for China network
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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

# ITR dependencies
from datetime import datetime
import re

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# ---------- Constants ----------

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
OUTPUT_DIR = get_output_dir(__file__)

RANDOM_SEED = 42

# BGE-m3 model names (multilingual, replaces English-only models from exp4)
BI_ENCODER_NAME = "BAAI/bge-m3"
CROSS_ENCODER_NAME = "BAAI/bge-reranker-v2-m3"

# ITR parameters
ITR_TOP_K = 20       # Stage 1: initial candidate count
ITR_RERANK_N = 5     # Stage 2: rerank to top-N tools
ITR_CONFIDENCE_THRESHOLD = 0.7  # Confidence gate for fallback
NO_SUITABLE_TOOL = "__need_more_tools__"

# ---------- Schema Builder ----------

TOOL_ACTIONS: dict[str, list[tuple[str, str, dict]]] = {}

# Core tools
TOOL_ACTIONS["shell"] = [("run", "Execute shell command", {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]})]
TOOL_ACTIONS["file"] = [
    ("read", "Read file content", {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}),
    ("write", "Write file content", {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}),
]
TOOL_ACTIONS["screen"] = [("capture", "Take screenshot", {"type": "object", "properties": {"region": {"type": "string"}}, "required": []})]
TOOL_ACTIONS["search"] = [("web_search", "Search the web", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]

# Extended tools
TOOL_ACTIONS["browser"] = [("open_url", "Open URL in browser", {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]})]
TOOL_ACTIONS["browser_use"] = [("execute", "AI browser automation", {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]})]
TOOL_ACTIONS["notify"] = [("send", "Send notification", {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]})]
TOOL_ACTIONS["clipboard"] = [("read", "Read clipboard", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["app_control"] = [("launch", "Launch application", {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]})]
TOOL_ACTIONS["calculator"] = [("calculate", "Calculate expression", {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]})]
TOOL_ACTIONS["datetime_tool"] = [("get_time", "Get current time", {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": []})]
TOOL_ACTIONS["stock_query"] = [("query", "Query stock price", {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]})]
TOOL_ACTIONS["tool_info"] = [("list", "List available tools", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["cron"] = [("add_task", "Add scheduled task", {"type": "object", "properties": {"instruction": {"type": "string"}}, "required": ["instruction"]})]
TOOL_ACTIONS["system_monitor"] = [("check", "System status check", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["knowledge_rag"] = [("search", "Knowledge base search", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
TOOL_ACTIONS["batch_paper_analyzer"] = [("analyze", "Batch paper analysis", {"type": "object", "properties": {"papers": {"type": "string"}}, "required": ["papers"]})]
TOOL_ACTIONS["python_runner"] = [("run", "Run Python code", {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]})]
TOOL_ACTIONS["literature_search"] = [("search", "Literature search", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
TOOL_ACTIONS["poetry"] = [("search", "Search poems", {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]})]
TOOL_ACTIONS["chat_history"] = [("search", "Search chat history", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
TOOL_ACTIONS["diary"] = [("write", "Write diary", {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]})]
TOOL_ACTIONS["finance"] = [("record", "Record expense", {"type": "object", "properties": {"amount": {"type": "string"}}, "required": ["amount"]})]
TOOL_ACTIONS["health"] = [("record", "Record health data", {"type": "object", "properties": {"metric": {"type": "string"}}, "required": ["metric"]})]
TOOL_ACTIONS["medication"] = [("remind", "Medication reminder", {"type": "object", "properties": {"medication": {"type": "string"}}, "required": ["medication"]})]
TOOL_ACTIONS["user_profile"] = [("update", "Update profile", {"type": "object", "properties": {"field": {"type": "string"}}, "required": ["field"]})]
TOOL_ACTIONS["family_member"] = [("list", "List family members", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["course_schedule"] = [("view", "View schedule", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["meal_menu"] = [("view", "View meal menu", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["todo"] = [("add", "Add todo item", {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]})]
TOOL_ACTIONS["daily_task"] = [("list", "List daily tasks", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["family_album"] = [("list", "List album photos", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["fitness_nutrition"] = [("plan", "Fitness plan", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["music_player"] = [("play", "Play music", {"type": "object", "properties": {"song": {"type": "string"}}, "required": ["song"]})]
TOOL_ACTIONS["fred_query"] = [("query", "FRED data query", {"type": "object", "properties": {"series": {"type": "string"}}, "required": ["series"]})]
TOOL_ACTIONS["quant_trading"] = [("analyze", "Quantitative analysis", {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]})]
TOOL_ACTIONS["voice_input"] = [("transcribe", "Voice to text", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["voice_output"] = [("speak", "Text to speech", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})]
TOOL_ACTIONS["ocr"] = [("recognize", "OCR recognition", {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]})]
TOOL_ACTIONS["speech_to_text"] = [("transcribe", "Audio transcription", {"type": "object", "properties": {"audio_path": {"type": "string"}}, "required": ["audio_path"]})]
TOOL_ACTIONS["document_scanner"] = [("scan", "Scan document", {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]})]
TOOL_ACTIONS["media_capture"] = [("capture", "Capture media", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["stock_photo"] = [("search", "Search stock photos", {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]})]
TOOL_ACTIONS["image_generator"] = [("generate", "Generate image", {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]})]
TOOL_ACTIONS["concept_diagrams"] = [("create", "Create diagram", {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]})]
TOOL_ACTIONS["meme_generation"] = [("create", "Create meme", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["wechat"] = [("send", "Send WeChat message", {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]})]
TOOL_ACTIONS["remote_file_share"] = [("send", "Share file remotely", {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]})]
TOOL_ACTIONS["pdf_tool"] = [("merge", "Merge PDFs", {"type": "object", "properties": {"files": {"type": "string"}}, "required": ["files"]})]
TOOL_ACTIONS["format_converter"] = [("convert", "Convert format", {"type": "object", "properties": {"from_format": {"type": "string"}}, "required": ["from_format"]})]
TOOL_ACTIONS["ppt_generator"] = [("generate", "Generate PPT", {"type": "object", "properties": {"outline": {"type": "string"}}, "required": ["outline"]})]
TOOL_ACTIONS["pdf_generator"] = [("generate", "Generate PDF", {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]})]
TOOL_ACTIONS["data_processor"] = [("process", "Process data", {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]})]
TOOL_ACTIONS["data_visualization"] = [("chart", "Create chart", {"type": "object", "properties": {"data": {"type": "string"}}, "required": ["data"]})]
TOOL_ACTIONS["financial_report"] = [("generate", "Generate report", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["statistics"] = [("analyze", "Statistical analysis", {"type": "object", "properties": {"data": {"type": "string"}}, "required": ["data"]})]
TOOL_ACTIONS["ai_writer"] = [("write", "AI writing assistant", {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]})]
TOOL_ACTIONS["contract_generator"] = [("generate", "Generate contract", {"type": "object", "properties": {"type": {"type": "string"}}, "required": ["type"]})]
TOOL_ACTIONS["resume_builder"] = [("generate", "Generate resume", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["id_photo"] = [("generate", "Generate ID photo", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["mind_map"] = [("generate", "Generate mind map", {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]})]
TOOL_ACTIONS["coding_assistant"] = [("assist", "Coding assistance", {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]})]
TOOL_ACTIONS["doc_generator"] = [("generate", "Generate document", {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]})]
TOOL_ACTIONS["weather"] = [("query", "Query weather", {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})]
TOOL_ACTIONS["mcp_browserbase"] = [("execute", "Cloud browser automation", {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]})]
TOOL_ACTIONS["mcp_browserbase-csdn"] = [("publish", "Publish CSDN blog", {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]})]
TOOL_ACTIONS["literature_review"] = [("generate", "Generate literature review", {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]})]
TOOL_ACTIONS["family_milestone"] = [("record", "Record family milestone", {"type": "object", "properties": {"event": {"type": "string"}}, "required": ["event"]})]
TOOL_ACTIONS["tool_audit"] = [("report", "Tool audit report", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["log_viewer"] = [("view", "View logs", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["codebase_search"] = [("search", "Search codebase", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
TOOL_ACTIONS["self_control"] = [("status", "Self control status", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["experience_recall"] = [("recall", "Recall experiences", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
TOOL_ACTIONS["duckduckgo_search"] = [("search", "DuckDuckGo search", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
TOOL_ACTIONS["crawlee_tool"] = [("crawl", "Crawl web pages", {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]})]
TOOL_ACTIONS["oss_pdf_search"] = [("search", "Search OSS PDFs", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
TOOL_ACTIONS["oss_pdf_download"] = [("download", "Download OSS PDF", {"type": "object", "properties": {"doi": {"type": "string"}}, "required": ["doi"]})]
TOOL_ACTIONS["local_paper_search"] = [("search", "Search local papers", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
TOOL_ACTIONS["research_lineage"] = [("trace", "Trace research lineage", {"type": "object", "properties": {"concept": {"type": "string"}}, "required": ["concept"]})]
TOOL_ACTIONS["research_landscape"] = [("generate", "Generate research landscape", {"type": "object", "properties": {"field": {"type": "string"}}, "required": ["field"]})]
TOOL_ACTIONS["contrarian_finder"] = [("find", "Find contrarian papers", {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]})]
TOOL_ACTIONS["idea_migration"] = [("trace", "Trace idea migration", {"type": "object", "properties": {"concept": {"type": "string"}}, "required": ["concept"]})]
TOOL_ACTIONS["methodology_deconstructor"] = [("deconstruct", "Deconstruct methodology", {"type": "object", "properties": {"paper": {"type": "string"}}, "required": ["paper"]})]
TOOL_ACTIONS["citation_storyteller"] = [("tell", "Tell citation story", {"type": "object", "properties": {"paper": {"type": "string"}}, "required": ["paper"]})]
TOOL_ACTIONS["journal_intelligence"] = [("assess", "Assess journal", {"type": "object", "properties": {"journal": {"type": "string"}}, "required": ["journal"]})]
TOOL_ACTIONS["paper_lifecycle"] = [("manage", "Manage paper lifecycle", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["qualitative_analysis"] = [("analyze", "Qualitative analysis", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["education_tool"] = [("quiz", "Generate quiz", {"type": "object", "properties": {"subject": {"type": "string"}}, "required": ["subject"]})]
TOOL_ACTIONS["english_conversation"] = [("practice", "English conversation practice", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["study_solver"] = [("solve", "Solve study problems", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["flashcards"] = [("create", "Create flashcards", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["ai_detection"] = [("check", "Check AI detection", {"type": "object", "properties": {}, "required": []})]
TOOL_ACTIONS["research_lookup"] = [("search", "Research lookup", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
TOOL_ACTIONS["gif_maker"] = [("create", "Create GIF", {"type": "object", "properties": {}, "required": []})]


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
    return set(TOOL_ACTIONS.keys())


def get_schema_tool_names(tool_schemas: list[dict]) -> set[str]:
    """Return tool names exposed by a list of function-calling schemas."""
    names: set[str] = set()
    for schema in tool_schemas:
        function_name = schema.get("function", {}).get("name", "")
        if function_name:
            names.add(_extract_tool_name(function_name))
    return names


def is_escalation_signal(selected_tool: str, exposed_tools: set[str]) -> bool:
    """Detect non-oracle escalation signals from model output.

    Escalation is triggered only when the model says the listed tools are
    insufficient, returns no parsable tool, or selects a tool that is not in the
    currently exposed schema set. A valid but wrong tool selection does not
    trigger escalation because the benchmark must not inspect ground truth.
    """
    return (
        not selected_tool
        or selected_tool == NO_SUITABLE_TOOL
        or selected_tool not in exposed_tools
    )


# ---------- Ground Truth ----------

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

# ---------- Query Generation ----------

QUERY_TEMPLATES: dict[str, list[str]] = {
    "browser_automation": [
        "打开网页 https://example.com 并截图", "帮我访问这个网站并获取内容",
        "在浏览器中登录网站", "打开链接并截取页面截图", "帮我操作网页上的表单",
        "访问网站并下载文件", "在浏览器中搜索信息", "打开URL并分析页面内容",
        "帮我自动化网页操作", "登录网站并提取数据",
    ],
    "file_operation": [
        "读取文件 content.txt 的内容", "帮我整理文件夹中的文件",
        "复制文件到另一个目录", "重命名文件 report.docx", "解压压缩文件 archive.zip",
        "创建新的目录结构", "移动文件到指定位置", "查看文件内容并总结",
        "批量重命名文件夹中的文件", "搜索文件中的关键词",
    ],
    "document_assembly": [
        "生成一份包含天气和图片的Word文档", "将诗歌和天气组合成文档",
        "把搜索结果整合成Word文档", "创建一个包含图片的文档",
        "将内容组装成DOCX格式文档", "整合多种内容生成报告文档",
    ],
    "system_admin": [
        "帮我截个屏", "查看当前系统进程", "启动计算器应用",
        "重启电脑", "读取剪贴板内容", "发送系统通知提醒我",
        "修改系统注册表", "检查系统服务状态", "清理系统临时文件",
        "调整系统显示设置",
    ],
    "system_monitoring": [
        "查看CPU和内存使用率", "系统状态如何", "电脑是否卡顿",
        "检查磁盘空间", "查看网络连接状态", "监控系统温度和风扇",
        "电脑健康状态检查", "查看电池电量", "资源占用情况",
        "任务管理器信息",
    ],
    "daily_assistant": [
        "今天天气怎么样", "现在几点了", "帮我算一下 123*456",
        "设置每天早上8点的提醒", "明天天气如何", "帮我计算房贷月供",
        "提醒我下午3点开会", "这周天气如何", "创建一个定时任务",
        "每周一早上提醒我写周报",
    ],
    "knowledge": [
        "搜索知识库中关于机器学习的内容", "帮我查一下李白的诗",
        "搜索古诗词中含'月'的诗句", "用RAG检索相关文档",
        "帮我找一首苏轼的词", "批量分析这些论文的摘要",
        "搜索关于深度学习的论文", "查找唐诗中写春天的诗",
        "知识库检索项目文档", "帮我搜索诗经中的名句",
    ],
    "life_management": [
        "写一篇今天的日记", "记录今天花了50元", "记录今天的体重70kg",
        "添加服药提醒", "更新我的个人信息", "查看家庭成员列表",
        "查看今天的课程表", "查看学校这周的食谱", "记录家庭大事",
        "添加一个待办事项", "查看今天的任务清单", "浏览家庭相册",
        "制定健身计划", "播放一首轻音乐", "设置每天写日记的定时任务",
    ],
    "financial_activity": [
        "查询茅台的实时股价", "分析A股大盘走势", "帮我选几只科技股",
        "查询GDP增长率", "量化分析苹果公司的交易信号", "查看FRED经济数据",
        "回测这个交易策略", "搜索最近的板块轮动信号", "查看通胀率数据",
        "推荐买入的股票",
    ],
    "multimedia": [
        "把这段语音转成文字", "朗读这段文字", "识别图片中的文字",
        "转录这个音频文件", "解析这张试卷", "播放一首钢琴曲",
        "用摄像头拍张照", "搜索免费图片素材", "生成一张风景画",
        "创建一个概念图", "制作一个表情包", "截取屏幕区域",
    ],
    "document_processing": [
        "合并这些PDF文件", "把Word转成PDF", "生成一份PPT演示",
        "拆分这个PDF", "把Markdown转成DOCX", "加密这个PDF文件",
        "搜索免费图片素材用于PPT", "把HTML转成PDF",
    ],
    "data_analysis": [
        "读取这个Excel文件", "生成柱状图", "创建数据仪表盘",
        "统计分析这些数据", "处理CSV文件", "生成折线图可视化",
        "生成财务报表", "数据清洗和过滤", "制作饼图",
        "读取数据文件并统计",
    ],
}


def generate_queries(n_per_intent: int = 8, seed: int = 42) -> list[dict]:
    """Generate evaluation queries with ground-truth tools."""
    rng = random.Random(seed)
    queries = []
    for intent, templates in QUERY_TEMPLATES.items():
        gt_tools = INTENT_GROUND_TRUTH.get(intent, set())
        if not gt_tools:
            continue
        for i in range(n_per_intent):
            template = templates[i % len(templates)]
            # Add slight variation
            variations = ["请", "帮我", "我需要", "能不能", ""]
            prefix = variations[i % len(variations)]
            query_text = f"{prefix}{template}" if prefix else template
            queries.append({
                "query_id": f"{intent}_{i}",
                "query": query_text,
                "intent": intent,
                "ground_truth_tools": sorted(gt_tools),
                "primary_gt": sorted(gt_tools)[0],  # Primary ground-truth tool
            })
    return queries


# ======================================================================
# ITR Three-Stage Retrieval Pipeline
# ======================================================================

class ITRRetriever:
    """Faithful reproduction of ITR's dense+BM25+cross-encoder pipeline.
    
    Stage 1: Dense retrieval via sentence-transformers (bi-encoder)
    Stage 2: BM25 lexical retrieval
    Stage 3: Cross-encoder reranking of merged candidates
    
    Follows Franko (2025) ITR architecture:
    - Bi-encoder for fast candidate generation
    - BM25 for lexical signal
    - Cross-encoder for precise relevance scoring
    - Confidence-gated fallback: if max confidence < threshold, broaden to top-2x tools
    """

    def __init__(self, model_name: str = BI_ENCODER_NAME, cross_encoder_name: str = CROSS_ENCODER_NAME):
        print(f"Loading bi-encoder: {model_name}...")
        self.bi_encoder = SentenceTransformer(model_name)
        print(f"Loading cross-encoder: {cross_encoder_name}...")
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        
        self.all_tool_names = sorted(get_all_tool_names())
        self.tool_descriptions = self._build_tool_descriptions()
        
        # Pre-compute embeddings
        print("Computing tool embeddings...")
        self.tool_embeddings = self.bi_encoder.encode(
            self.tool_descriptions, show_progress_bar=False, normalize_embeddings=True
        )
        
        # Build BM25 index
        print("Building BM25 index...")
        tokenized_corpus = [self._tokenize(desc) for desc in self.tool_descriptions]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print("ITR Retriever initialized.")

    def _build_tool_descriptions(self) -> list[str]:
        """Build text representations of all tools for retrieval."""
        descriptions = []
        for tname in self.all_tool_names:
            actions = TOOL_ACTIONS.get(tname, [])
            parts = [f"Tool: {tname}"]
            # Add intent mapping info
            for intent, tools in INTENT_TOOL_MAPPING.items():
                if tname in tools:
                    parts.append(f"Intent: {intent}")
                    break
            for action_name, desc, _ in actions:
                parts.append(f"Action {action_name}: {desc}")
            descriptions.append(" | ".join(parts))
        return descriptions

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        # Lowercase, split on non-alphanumeric, keep tokens > 1 char
        tokens = re.findall(r'[a-zA-Z0-9\u4e00-\u9fff]+', text.lower())
        return [t for t in tokens if len(t) > 1]

    def retrieve(self, query: str, top_k: int = ITR_RERANK_N) -> tuple[set[str], float]:
        """Three-stage retrieval: dense → BM25 → cross-encoder reranking.
        
        Returns:
            (selected_tool_names, confidence_score)
        """
        # Stage 1: Dense retrieval
        query_embedding = self.bi_encoder.encode([query], normalize_embeddings=True)
        dense_scores = np.dot(query_embedding, self.tool_embeddings.T)[0]
        dense_top_indices = np.argsort(dense_scores)[::-1][:ITR_TOP_K]
        
        # Stage 2: BM25 retrieval
        tokenized_query = self._tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:ITR_TOP_K]
        
        # Merge candidates (union of dense + BM25 top-k)
        candidate_indices = set(dense_top_indices) | set(bm25_top_indices)
        
        # Stage 3: Cross-encoder reranking
        candidates = [(idx, self.tool_descriptions[idx]) for idx in candidate_indices]
        if not candidates:
            return set(), 0.0
        
        pairs = [(query, desc) for _, desc in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        
        # Sort by cross-encoder score
        scored = list(zip(candidates, ce_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Confidence = max cross-encoder score (normalized to 0-1)
        max_confidence = float(scored[0][1])
        # Normalize cross-encoder score to [0, 1] range
        confidence = 1.0 / (1.0 + np.exp(-max_confidence))  # sigmoid
        
        # Select top-k tools
        selected_indices = [idx for (idx, _), _ in scored[:top_k]]
        selected_tools = {self.all_tool_names[idx] for idx in selected_indices}
        
        # Confidence-gated fallback: if confidence is low, broaden
        if confidence < ITR_CONFIDENCE_THRESHOLD:
            # Double the tool set (like ITR's confidence-gated fallback)
            fallback_indices = [idx for (idx, _), _ in scored[:top_k * 2]]
            selected_tools = {self.all_tool_names[idx] for idx in fallback_indices}
        
        return selected_tools, confidence


# ======================================================================
# LLM-based Tool Selection
# ======================================================================

def llm_select_tool(
    client: OpenAI,
    query: str,
    tool_schemas: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 300,
) -> tuple[str, int, str]:
    """Ask LLM to select a tool given the available schemas.
    
    Returns:
        (selected_tool_name, input_tokens_used, raw_response)
    """
    system_prompt = (
        "You are a tool selection assistant. Given a user query and a list of available tools, "
        "select the SINGLE most appropriate tool. Respond with ONLY the tool function name "
        "(e.g., 'stock_query_query'). If none of the listed tools is suitable, respond exactly "
        "NEED_MORE_TOOLS. Do not include any explanation."
    )
    
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Available tools:\n{json.dumps(tool_schemas, ensure_ascii=False)}\n\nUser query: {query}"},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content.strip()
        # Extract tool name from response
        if raw.upper().strip() == "NEED_MORE_TOOLS":
            selected = NO_SUITABLE_TOOL
        else:
            selected = _extract_tool_name(raw.split("(")[0].split(".")[0].strip())
        input_tokens = resp.usage.prompt_tokens if resp.usage else 0
        return selected, input_tokens, raw
    except Exception as e:
        print(f"  LLM error: {e}")
        return "", 0, ""


# ======================================================================
# Experiment Runner
# ======================================================================

def run_experiment():
    """Run ITR vs PTE-FD vs Static comparison experiment."""
    
    if not API_KEY:
        print("ERROR: DEEPSEEK_API_KEY not set in .env")
        return
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Generate queries
    queries = generate_queries(n_per_intent=8, seed=RANDOM_SEED)
    print(f"Generated {len(queries)} queries across {len(QUERY_TEMPLATES)} intents")
    
    # Initialize ITR retriever
    skip_itr = os.getenv("WECLAW_EXP4_SKIP_ITR", "").strip() == "1"
    itr = None
    if skip_itr:
        print("Skipping ITR retriever because WECLAW_EXP4_SKIP_ITR=1.")
    else:
        itr = ITRRetriever()
    
    # Mock registry for PTE
    class MockToolRegistry:
        def get_all_schemas(self):
            return build_schemas(get_all_tool_names())
        def get_schemas_by_names(self, tool_names):
            return build_schemas(tool_names)
        def list_all_tool_names(self):
            return get_all_tool_names()
        def get_tool_config(self, name):
            return {}
    
    registry = MockToolRegistry()
    engine = ToolExposureEngine(registry, enabled=True, failures_to_upgrade=2)
    
    # Results storage
    results = {
        "static": {"correct": 0, "total": 0, "tokens": []},
        "pte_fd": {"correct": 0, "total": 0, "tokens": [], "escalated": 0, "recovered": 0},
        "itr": {"correct": 0, "total": 0, "tokens": [], "confidence_scores": []},
        "keyword": {"correct": 0, "total": 0, "tokens": []},
    }
    detail_rows = []
    
    all_tool_names = get_all_tool_names()
    all_schemas = build_schemas(all_tool_names)
    
    for i, q in enumerate(queries):
        query_text = q["query"]
        gt_tools = set(q["ground_truth_tools"])
        primary_gt = q["primary_gt"]
        
        print(f"\n[{i+1}/{len(queries)}] {query_text[:50]}...")
        
        # --- Static baseline: all tools ---
        selected_static, tokens_static, raw_static = llm_select_tool(client, query_text, all_schemas)
        static_correct = selected_static in gt_tools
        results["static"]["tokens"].append(tokens_static)
        if static_correct:
            results["static"]["correct"] += 1
        results["static"]["total"] += 1
        
        # --- PTE-FD ---
        engine.reset()
        intent_result = detect_intent_with_confidence(query_text)
        pte_schemas = engine.get_schemas(intent_result)
        pte_exposed_tools = get_schema_tool_names(pte_schemas)
        selected_pte, tokens_pte_first, raw_pte = llm_select_tool(client, query_text, pte_schemas)
        tokens_pte = tokens_pte_first
        pte_correct = selected_pte in gt_tools
        
        # Non-oracle escalation: only model-visible failure signals can trigger
        # broadening. A valid but wrong selection does not trigger escalation.
        escalated = False
        recovered = False
        escalation_signal = is_escalation_signal(selected_pte, pte_exposed_tools)
        if escalation_signal:
            # A second same-tier attempt represents the production k=2
            # consecutive-failure threshold without inspecting ground truth.
            selected_retry, tokens_retry, raw_retry = llm_select_tool(client, query_text, pte_schemas)
            tokens_pte += tokens_retry
            if is_escalation_signal(selected_retry, pte_exposed_tools):
                engine.report_failure()
                upgraded = engine.report_failure()
                if upgraded:
                    pte_fd_schemas = engine.get_schemas(intent_result)
                    selected_pte_fd, tokens_pte_fd, raw_pte_fd = llm_select_tool(client, query_text, pte_fd_schemas)
                    tokens_pte += tokens_pte_fd
                    escalated = True
                    selected_pte = selected_pte_fd
                    pte_correct = selected_pte in gt_tools
                    recovered = pte_correct
            else:
                selected_pte = selected_retry
                pte_correct = selected_pte in gt_tools
        
        results["pte_fd"]["tokens"].append(tokens_pte)
        results["pte_fd"]["escalated"] += int(escalated)
        results["pte_fd"]["recovered"] += int(recovered)
        if pte_correct:
            results["pte_fd"]["correct"] += 1
        results["pte_fd"]["total"] += 1
        
        # --- ITR (dense+BM25+cross-encoder) ---
        if itr is not None:
            itr_tools, itr_confidence = itr.retrieve(query_text, top_k=ITR_RERANK_N)
            results["itr"]["confidence_scores"].append(itr_confidence)
            itr_schemas = build_schemas(itr_tools)
            selected_itr, tokens_itr, raw_itr = llm_select_tool(client, query_text, itr_schemas)
            itr_correct = selected_itr in gt_tools
            results["itr"]["tokens"].append(tokens_itr)
            if itr_correct:
                results["itr"]["correct"] += 1
            results["itr"]["total"] += 1
        else:
            itr_tools = set()
            itr_confidence = 0.0
            selected_itr = ""
            tokens_itr = 0
            itr_correct = False
        
        # --- Keyword Retrieval baseline ---
        # Simple keyword matching (original naive baseline)
        kw_tools = set()
        for intent, keywords in INTENT_CATEGORIES.items():
            if any(kw in query_text.lower() for kw in keywords):
                kw_tools.update(INTENT_TOOL_MAPPING.get(intent, []))
        # Add core tools
        kw_tools.update(ToolExposureEngine.CORE_TOOLS)
        if not kw_tools:
            kw_tools = {"shell", "file", "search"}  # fallback
        kw_schemas = build_schemas(kw_tools)
        selected_kw, tokens_kw, raw_kw = llm_select_tool(client, query_text, kw_schemas)
        kw_correct = selected_kw in gt_tools
        results["keyword"]["tokens"].append(tokens_kw)
        if kw_correct:
            results["keyword"]["correct"] += 1
        results["keyword"]["total"] += 1
        
        # Store detail
        detail_rows.append({
            "query_id": q["query_id"],
            "query": query_text,
            "intent": q["intent"],
            "ground_truth": sorted(gt_tools),
            "static": {"selected": selected_static, "correct": static_correct, "tokens": tokens_static},
            "pte_fd": {"selected": selected_pte, "correct": pte_correct, "tokens": tokens_pte, "escalated": escalated, "recovered": recovered},
            "itr": {"selected": selected_itr, "correct": itr_correct, "tokens": tokens_itr, "confidence": round(itr_confidence, 4), "tools_count": len(itr_tools)},
            "keyword": {"selected": selected_kw, "correct": kw_correct, "tokens": tokens_kw},
        })
        
        # Rate limiting
        time.sleep(0.5)
    
    # ---------- Compute summary ----------
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS: ITR Faithful Reproduction")
    print("=" * 70)
    
    for method in ["static", "pte_fd", "itr", "keyword"]:
        r = results[method]
        label = method.upper().replace("_", "-")
        if r["total"] == 0:
            print(f"  {label:15s}: skipped")
            continue
        acc = r["correct"] / r["total"] * 100
        avg_tokens = np.mean(r["tokens"]) if r["tokens"] else 0
        print(f"  {label:15s}: Acc={acc:.1f}%, Avg tokens/q={avg_tokens:.0f}, n={r['total']}")
    
    if results["pte_fd"]["escalated"] > 0:
        rec_rate = results["pte_fd"]["recovered"] / results["pte_fd"]["escalated"] * 100
        print(f"  PTE-FD Escalation: {results['pte_fd']['escalated']} escalated, "
              f"{results['pte_fd']['recovered']} recovered ({rec_rate:.1f}%)")
    
    if results["itr"]["confidence_scores"]:
        avg_conf = np.mean(results["itr"]["confidence_scores"])
        print(f"  ITR Avg Confidence: {avg_conf:.4f}")
    
    # Save results
    output = {
        "experiment": "itr_faithful_reproduction",
        "model": MODEL,
        "n_queries": len(queries),
        "itr_params": {
            "status": "skipped" if skip_itr else "completed",
            "top_k": ITR_TOP_K,
            "rerank_n": ITR_RERANK_N,
            "confidence_threshold": ITR_CONFIDENCE_THRESHOLD,
            "bi_encoder": BI_ENCODER_NAME,
            "cross_encoder": CROSS_ENCODER_NAME,
        },
        "summary": {
            method: {
                "accuracy": r["correct"] / r["total"] * 100 if r["total"] > 0 else None,
                "avg_tokens_per_query": float(np.mean(r["tokens"])) if r["tokens"] else 0,
                "n_correct": r["correct"],
                "n_total": r["total"],
            }
            for method, r in results.items()
        },
        "details": detail_rows,
    }
    
    # Add PTE-FD escalation info
    output["summary"]["pte_fd"]["escalated"] = results["pte_fd"]["escalated"]
    output["summary"]["pte_fd"]["recovered"] = results["pte_fd"]["recovered"]
    output["summary"]["pte_fd"]["recovery_rate"] = (
        results["pte_fd"]["recovered"] / results["pte_fd"]["escalated"] * 100
        if results["pte_fd"]["escalated"] > 0 else 0
    )
    output["summary"]["itr"]["avg_confidence"] = (
        float(np.mean(results["itr"]["confidence_scores"]))
        if results["itr"]["confidence_scores"] else None
    )
    
    # Inject experiment isolation metadata
    run_id = os.getenv("WECLAW_BENCHMARK_RUN_ID", "")
    output["experiment"] = "exp4b_itr_bge_m3"
    output["run_id"] = run_id
    output["timestamp"] = datetime.now().isoformat(timespec="seconds")
    output["source_script"] = "exp4b_itr_bge_m3.py (copy of exp4 with BGE-m3)"

    out_path = OUTPUT_DIR / "exp4b_itr_bge_m3_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")
    
    # Generate LaTeX table
    print("\n--- LaTeX Table Fragment ---")
    s = output["summary"]
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\caption{ITR-BGE-m3 faithful reproduction vs.\ PTE-FD and baselines ($n{=}" + str(len(queries)) + r"$). \emph{ITR-BGE-m3}: dense (BAAI/bge-m3)+BM25+cross-encoder (BAAI/bge-reranker-v2-m3). \emph{Kw}: Keyword Retrieval (naive baseline).}")
    print(r"\label{tab:itr_bge_m3}")
    print(r"\begin{tabular}{@{}lcccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Acc.} & \textbf{Tokens/q} & \textbf{Tools/q} & \textbf{Recovery} \\")
    print(r"\midrule")
    print(f"Static (60+ tools) & {s['static']['accuracy']:.1f}\\% & {s['static']['avg_tokens_per_query']:.0f} & 60+ & --- \\\\")
    print(f"PTE-FD (ours) & \\textbf{{{s['pte_fd']['accuracy']:.1f}\\%}} & {s['pte_fd']['avg_tokens_per_query']:.0f} & ~10--20 & {s['pte_fd']['recovery_rate']:.1f}\\% \\\\")
    print(f"ITR-BGE-m3 & {s['itr']['accuracy']:.1f}\\% & {s['itr']['avg_tokens_per_query']:.0f} & {ITR_RERANK_N} & --- \\\\")
    if s["itr"]["accuracy"] is None:
        print(f"ITR-BGE-m3 & skipped & --- & --- & --- \\\\")
    print(f"Keyword Ret. & {s['keyword']['accuracy']:.1f}\\% & {s['keyword']['avg_tokens_per_query']:.0f} & 3--5 & --- \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run_experiment()
