#!/usr/bin/env python3
"""Experiment 6: End-to-End Multi-Step Desktop Tasks

Evaluates PTE-FD on realistic multi-step desktop agent workflows.
Each task requires 2-5 tool calls with real dependencies.

Uses DeepSeek V4 API for actual LLM inference.
"""

import json
import os
import sys
import time
import random
import numpy as np
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

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
MODEL = "deepseek-chat"
OUTPUT_DIR = get_output_dir(__file__)
RANDOM_SEED = 42
NO_SUITABLE_TOOL = "__need_more_tools__"

# ---------- Multi-Step Task Definitions ----------

MULTI_STEP_TASKS: list[dict] = [
    # Financial analysis workflow
    {
        "task_id": "fin_01",
        "description": "查询苹果公司股价，然后分析交易信号",
        "steps": [
            {"query": "查询苹果公司的实时股价", "ground_truth": "stock_query"},
            {"query": "分析AAPL的交易信号", "ground_truth": "quant_trading"},
        ],
        "category": "financial",
    },
    {
        "task_id": "fin_02",
        "description": "查看A股大盘并筛选科技股",
        "steps": [
            {"query": "查看A股大盘走势", "ground_truth": "stock_query"},
            {"query": "帮我筛选科技板块买入信号", "ground_truth": "quant_trading"},
        ],
        "category": "financial",
    },
    {
        "task_id": "fin_03",
        "description": "查询GDP数据并生成经济分析报告",
        "steps": [
            {"query": "查询美国GDP增长率数据", "ground_truth": "fred_query"},
            {"query": "生成经济分析Word文档", "ground_truth": "doc_generator"},
        ],
        "category": "financial",
    },
    # Academic research workflow
    {
        "task_id": "res_01",
        "description": "搜索论文并下载PDF",
        "steps": [
            {"query": "搜索关于LLM agent的论文", "ground_truth": "literature_search"},
            {"query": "检查这篇论文的PDF是否在OSS中", "ground_truth": "oss_pdf_search"},
            {"query": "下载这篇论文的PDF", "ground_truth": "oss_pdf_download"},
        ],
        "category": "research",
    },
    {
        "task_id": "res_02",
        "description": "分析研究领域并找反共识论文",
        "steps": [
            {"query": "生成强化学习领域的学术地形图", "ground_truth": "research_landscape"},
            {"query": "找关于强化学习的反共识论文", "ground_truth": "contrarian_finder"},
        ],
        "category": "research",
    },
    {
        "task_id": "res_03",
        "description": "论文项目管理和文献检索",
        "steps": [
            {"query": "创建一个新的论文写作项目", "ground_truth": "paper_lifecycle"},
            {"query": "搜索相关文献", "ground_truth": "literature_search"},
        ],
        "category": "research",
    },
    # Document creation workflow
    {
        "task_id": "doc_01",
        "description": "搜索图片并生成PPT",
        "steps": [
            {"query": "搜索AI相关的免费图片", "ground_truth": "stock_photo"},
            {"query": "生成一份关于AI技术的PPT", "ground_truth": "ppt_generator"},
        ],
        "category": "document",
    },
    {
        "task_id": "doc_02",
        "description": "读取数据并生成可视化图表",
        "steps": [
            {"query": "读取sales_data.xlsx文件", "ground_truth": "data_processor"},
            {"query": "生成销售额柱状图", "ground_truth": "data_visualization"},
        ],
        "category": "document",
    },
    {
        "task_id": "doc_03",
        "description": "转换文档格式并合并PDF",
        "steps": [
            {"query": "把report.md转成PDF格式", "ground_truth": "format_converter"},
            {"query": "合并所有PDF文件", "ground_truth": "pdf_tool"},
        ],
        "category": "document",
    },
    # Life management workflow
    {
        "task_id": "life_01",
        "description": "记录健康数据并制定健身计划",
        "steps": [
            {"query": "记录今天的体重和血压", "ground_truth": "health"},
            {"query": "制定一个减脂健身计划", "ground_truth": "fitness_nutrition"},
        ],
        "category": "life",
    },
    {
        "task_id": "life_02",
        "description": "查看课程表和食谱",
        "steps": [
            {"query": "查看今天的课程安排", "ground_truth": "course_schedule"},
            {"query": "查看学校这周的食谱", "ground_truth": "meal_menu"},
        ],
        "category": "life",
    },
    {
        "task_id": "life_03",
        "description": "添加待办并设置提醒",
        "steps": [
            {"query": "添加一个明天开会的待办事项", "ground_truth": "todo"},
            {"query": "设置明天早上9点提醒我开会", "ground_truth": "cron"},
        ],
        "category": "life",
    },
    # Multimedia workflow
    {
        "task_id": "mul_01",
        "description": "语音转文字后生成文档",
        "steps": [
            {"query": "把这段语音转成文字", "ground_truth": "voice_input"},
            {"query": "将文字内容生成Word文档", "ground_truth": "doc_generator"},
        ],
        "category": "multimedia",
    },
    {
        "task_id": "mul_02",
        "description": "截图并识别文字",
        "steps": [
            {"query": "截取屏幕", "ground_truth": "screen"},
            {"query": "识别截图中的文字", "ground_truth": "ocr"},
        ],
        "category": "multimedia",
    },
    # Browser automation workflow
    {
        "task_id": "brw_01",
        "description": "打开网页并提取信息",
        "steps": [
            {"query": "打开https://example.com网站", "ground_truth": "browser"},
            {"query": "截图保存网页内容", "ground_truth": "screen"},
        ],
        "category": "browser",
    },
    # System admin workflow
    {
        "task_id": "sys_01",
        "description": "检查系统状态并清理",
        "steps": [
            {"query": "查看CPU和内存使用率", "ground_truth": "system_monitor"},
            {"query": "清理系统临时文件", "ground_truth": "shell"},
        ],
        "category": "system",
    },
    # Knowledge + creative
    {
        "task_id": "knw_01",
        "description": "搜索诗词并生成思维导图",
        "steps": [
            {"query": "搜索李白的诗", "ground_truth": "poetry"},
            {"query": "生成唐诗思维导图", "ground_truth": "mind_map"},
        ],
        "category": "knowledge",
    },
    # Complex multi-step (5 steps)
    {
        "task_id": "cpx_01",
        "description": "完整的量化交易分析流程",
        "steps": [
            {"query": "查询茅台的实时股价", "ground_truth": "stock_query"},
            {"query": "分析600519的交易信号", "ground_truth": "quant_trading"},
            {"query": "评估这只股的风险", "ground_truth": "quant_trading"},
            {"query": "计算合适的仓位", "ground_truth": "quant_trading"},
            {"query": "生成量化分析日报", "ground_truth": "quant_trading"},
        ],
        "category": "complex",
    },
    {
        "task_id": "cpx_02",
        "description": "学术论文全流程",
        "steps": [
            {"query": "搜索关于agent的学术论文", "ground_truth": "literature_search"},
            {"query": "检查论文PDF在OSS中是否存在", "ground_truth": "oss_pdf_search"},
            {"query": "追踪agent思想谱系", "ground_truth": "research_lineage"},
            {"query": "评估适合投稿的期刊", "ground_truth": "journal_intelligence"},
        ],
        "category": "complex",
    },
]


# ---------- Tool Schema Builder ----------

# Reuse from exp4 - simplified tool schema definitions
TOOL_DESCRIPTIONS = {
    "shell": "执行Shell命令", "file": "文件读写操作", "screen": "截取屏幕",
    "search": "网络搜索", "browser": "浏览器操作", "browser_use": "AI浏览器自动化",
    "notify": "发送通知", "clipboard": "剪贴板操作", "app_control": "应用控制",
    "calculator": "计算器", "datetime_tool": "日期时间", "cron": "定时任务",
    "system_monitor": "系统监控", "stock_query": "股票行情查询",
    "quant_trading": "量化交易分析", "fred_query": "FRED经济数据",
    "knowledge_rag": "知识库检索", "batch_paper_analyzer": "批量论文分析",
    "python_runner": "Python代码执行", "literature_search": "学术文献检索",
    "poetry": "古诗词库", "chat_history": "聊天历史检索",
    "diary": "日记管理", "finance": "财务管理", "health": "健康数据",
    "medication": "用药管理", "todo": "待办事项", "daily_task": "每日任务",
    "course_schedule": "课程表管理", "meal_menu": "食谱管理",
    "family_album": "家庭相册", "fitness_nutrition": "健身营养",
    "music_player": "音乐播放器", "voice_input": "语音输入",
    "voice_output": "语音输出", "ocr": "OCR文字识别",
    "speech_to_text": "语音转文字", "document_scanner": "文档扫描",
    "media_capture": "媒体捕获", "stock_photo": "图库搜索",
    "image_generator": "AI绘图", "doc_generator": "Word文档生成",
    "ppt_generator": "PPT生成", "pdf_tool": "PDF工具",
    "format_converter": "格式转换", "pdf_generator": "PDF生成",
    "data_processor": "数据处理", "data_visualization": "数据可视化",
    "financial_report": "财务报表", "ai_writer": "AI写作",
    "mind_map": "思维导图", "oss_pdf_search": "OSS PDF搜索",
    "oss_pdf_download": "OSS PDF下载", "local_paper_search": "本地论文搜索",
    "research_lineage": "思想谱系追踪", "research_landscape": "学术地形图",
    "contrarian_finder": "反共识雷达", "journal_intelligence": "期刊情报局",
    "paper_lifecycle": "论文全周期管家", "mcp_browserbase": "MCP云端浏览器",
    "coding_assistant": "编程助手", "tool_info": "工具信息",
}


def build_schemas(tool_names: set[str]) -> list[dict]:
    schemas = []
    for tname in sorted(tool_names):
        desc = TOOL_DESCRIPTIONS.get(tname, f"Tool: {tname}")
        schemas.append({
            "type": "function",
            "function": {
                "name": f"{tname}_execute",
                "description": desc,
                "parameters": {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
            },
        })
    return schemas


def get_all_tool_names() -> set[str]:
    return set(TOOL_DESCRIPTIONS.keys())


def get_schema_tool_names(tool_schemas: list[dict]) -> set[str]:
    names: set[str] = set()
    for schema in tool_schemas:
        function_name = schema.get("function", {}).get("name", "")
        if function_name:
            names.add(_extract_tool_name(function_name))
    return names


def is_escalation_signal(selected_tool: str, exposed_tools: set[str]) -> bool:
    return (
        not selected_tool
        or selected_tool == NO_SUITABLE_TOOL
        or selected_tool not in exposed_tools
    )


def llm_select_tool(client, query, tool_schemas, temperature=0.0, max_tokens=300):
    system_prompt = (
        "You are a tool selection assistant. Given a user query and available tools, "
        "select the SINGLE most appropriate tool. Respond with ONLY the tool function name "
        "(e.g., 'stock_query_execute'). If none of the listed tools is suitable, respond exactly "
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
        # Extract tool name
        if raw.upper().strip() == "NEED_MORE_TOOLS":
            selected = NO_SUITABLE_TOOL
        else:
            selected = _extract_tool_name(raw.split("(")[0].split(".")[0].strip())
        input_tokens = resp.usage.prompt_tokens if resp.usage else 0
        return selected, input_tokens, raw
    except Exception as e:
        print(f"  LLM error: {e}")
        return "", 0, ""


# ---------- Experiment ----------

def run_experiment():
    if not API_KEY:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Mock registry
    class MockRegistry:
        def get_all_schemas(self):
            return build_schemas(get_all_tool_names())
        def get_schemas_by_names(self, tool_names):
            return build_schemas(tool_names)
        def list_all_tool_names(self):
            return get_all_tool_names()
        def get_tool_config(self, name):
            return {}
    
    registry = MockRegistry()
    engine = ToolExposureEngine(registry, enabled=True, failures_to_upgrade=2)
    
    all_tool_names = get_all_tool_names()
    all_schemas = build_schemas(all_tool_names)
    
    results = {
        "static": {"step_correct": 0, "task_complete": 0, "total_steps": 0, "tokens": []},
        "pte_fd": {
            "step_correct": 0,
            "task_complete": 0,
            "total_steps": 0,
            "tokens": [],
            "escalated": 0,
            "recovered": 0,
        },
    }
    
    task_details = []
    total_tasks = len(MULTI_STEP_TASKS)
    total_steps = sum(len(t["steps"]) for t in MULTI_STEP_TASKS)
    
    print(f"Running {total_tasks} multi-step tasks ({total_steps} total steps)")
    
    for task_idx, task in enumerate(MULTI_STEP_TASKS):
        task_id = task["task_id"]
        task_desc = task["description"]
        steps = task["steps"]
        
        print(f"\n[Task {task_idx+1}/{total_tasks}] {task_desc}")
        
        static_all_correct = True
        pte_all_correct = True
        step_details = []
        
        for step_idx, step in enumerate(steps):
            query = step["query"]
            gt = step["ground_truth"]
            
            # --- Static ---
            sel_static, tok_static, raw_static = llm_select_tool(client, query, all_schemas)
            static_ok = sel_static == gt
            results["static"]["tokens"].append(tok_static)
            results["static"]["total_steps"] += 1
            if static_ok:
                results["static"]["step_correct"] += 1
            else:
                static_all_correct = False
            
            # --- PTE-FD ---
            engine.reset()
            intent_result = detect_intent_with_confidence(query)
            pte_schemas = engine.get_schemas(intent_result)
            pte_exposed_tools = get_schema_tool_names(pte_schemas)
            sel_pte, tok_pte_first, raw_pte = llm_select_tool(client, query, pte_schemas)
            tok_pte = tok_pte_first
            pte_ok = sel_pte == gt
            escalated = False
            recovered = False
            
            if is_escalation_signal(sel_pte, pte_exposed_tools):
                # Repeat once at the same tier to model the production k=2
                # consecutive-failure threshold without inspecting ground truth.
                sel_retry, tok_retry, raw_retry = llm_select_tool(client, query, pte_schemas)
                tok_pte += tok_retry
                if is_escalation_signal(sel_retry, pte_exposed_tools):
                    engine.report_failure()
                    upgraded = engine.report_failure()
                    if upgraded:
                        pte_schemas_fd = engine.get_schemas(intent_result)
                        sel_pte_fd, tok_pte_fd, raw_pte_fd = llm_select_tool(client, query, pte_schemas_fd)
                        tok_pte += tok_pte_fd
                        escalated = True
                        sel_pte = sel_pte_fd
                        pte_ok = sel_pte == gt
                        recovered = pte_ok
                else:
                    sel_pte = sel_retry
                    pte_ok = sel_pte == gt

            if not pte_ok:
                pte_all_correct = False
            
            results["pte_fd"]["tokens"].append(tok_pte)
            results["pte_fd"]["escalated"] += int(escalated)
            results["pte_fd"]["recovered"] += int(recovered)
            results["pte_fd"]["total_steps"] += 1
            if pte_ok:
                results["pte_fd"]["step_correct"] += 1
            
            step_details.append({
                "step_index": step_idx + 1,
                "query": query,
                "ground_truth": gt,
                "static": {"selected": sel_static, "correct": static_ok, "tokens": tok_static},
                "pte_fd": {
                    "selected": sel_pte,
                    "correct": pte_ok,
                    "tokens": tok_pte,
                    "escalated": escalated,
                    "recovered": recovered,
                },
            })

            static_status = "OK" if static_ok else "FAIL"
            pte_status = "OK" if pte_ok else "FAIL"
            print(f"  Step {step_idx+1}: '{query[:40]}...' GT={gt} | Static={sel_static}({static_status}) PTE-FD={sel_pte}({pte_status})")
            
            time.sleep(0.5)
        
        if static_all_correct:
            results["static"]["task_complete"] += 1
        if pte_all_correct:
            results["pte_fd"]["task_complete"] += 1
        
        task_details.append({
            "task_id": task_id,
            "description": task_desc,
            "category": task["category"],
            "n_steps": len(steps),
            "static_all_correct": static_all_correct,
            "pte_fd_all_correct": pte_all_correct,
            "steps": step_details,
        })
    
    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print("MULTI-STEP TASK RESULTS")
    print("=" * 70)
    
    for method in ["static", "pte_fd"]:
        r = results[method]
        step_acc = r["step_correct"] / r["total_steps"] * 100 if r["total_steps"] > 0 else 0
        task_rate = r["task_complete"] / total_tasks * 100
        avg_tok = np.mean(r["tokens"]) if r["tokens"] else 0
        label = method.upper().replace("_", "-")
        print(f"  {label:10s}: Step Acc={step_acc:.1f}%, Task Complete={r['task_complete']}/{total_tasks} ({task_rate:.1f}%), Avg tokens/step={avg_tok:.0f}")
        if method == "pte_fd":
            recovery_rate = r["recovered"] / r["escalated"] * 100 if r["escalated"] else 0
            print(f"  {'':10s}  Escalated={r['escalated']}, Recovered={r['recovered']} ({recovery_rate:.1f}%)")
    
    # Save
    output = {
        "experiment": "multi_step_tasks",
        "model": MODEL,
        "n_tasks": total_tasks,
        "n_steps": total_steps,
        "summary": {
            method: {
                "step_accuracy": r["step_correct"] / r["total_steps"] * 100,
                "task_completion_rate": r["task_complete"] / total_tasks * 100,
                "avg_tokens_per_step": float(np.mean(r["tokens"])) if r["tokens"] else 0,
                "n_steps_correct": r["step_correct"],
                "n_tasks_complete": r["task_complete"],
                "escalated": r.get("escalated", 0),
                "recovered": r.get("recovered", 0),
                "recovery_rate": (
                    r.get("recovered", 0) / r.get("escalated", 1) * 100
                    if r.get("escalated", 0) else 0
                ),
            }
            for method, r in results.items()
        },
        "task_details": task_details,
    }
    
    out_path = OUTPUT_DIR / "exp6_multistep_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run_experiment()
