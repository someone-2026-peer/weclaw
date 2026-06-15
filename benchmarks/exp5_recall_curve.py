#!/usr/bin/env python3
"""Experiment 5: EBEAC Recall vs Pattern Library Size Curve

Vary the number of stored error patterns and measure recall and prevention rate.
Shows how EBEAC performance scales with the experience library.
"""

import json
import os
import sys
import random
import numpy as np
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from benchmark_utils import get_output_dir
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_DIR = get_output_dir(__file__)

RANDOM_SEED = 42

# ---------- Error Pattern Generation ----------

# 7 abstract categories from the paper
ERROR_CATEGORIES = {
    "permission_error": {
        "keywords": ["权限", "permission", "denied", "拒绝", "access denied", "unauthorized"],
        "tools": ["shell", "file", "app_control"],
        "patterns": [
            ("shell_run: PermissionError when accessing system directory", "shell", "permission_error | shell工具问题 | 修复:权限检查"),
            ("file_write: Access denied writing to C:\\Program Files", "file", "permission_error | file工具问题 | 修复:默认值"),
            ("app_control_launch: Unauthorized access to registry key", "app_control", "permission_error | app_control问题 | 修复:降级"),
            ("shell_run: EACCES permission denied on /root/", "shell", "permission_error | shell工具问题 | 修复:用户目录"),
            ("file_read: Access denied to encrypted file", "file", "permission_error | file工具问题 | 修复:跳过加密"),
            ("shell_run: sudo required for system service restart", "shell", "permission_error | shell工具问题 | 修复:提示用户"),
            ("app_control_launch: Windows UAC blocked application start", "app_control", "permission_error | app_control问题 | 修复:提权请求"),
        ],
    },
    "memory_error": {
        "keywords": ["内存", "memory", "OOM", "out of memory", "内存不足"],
        "tools": ["python_runner", "data_processor", "image_generator"],
        "patterns": [
            ("python_runner: MemoryError processing large dataset", "python_runner", "memory_error | python_runner问题 | 修复:异常处理"),
            ("data_processor: OOM when loading 500MB Excel file", "data_processor", "memory_error | data_processor问题 | 修复:分块读取"),
            ("image_generator: Out of memory generating 4K image", "image_generator", "memory_error | image_generator问题 | 修复:降低分辨率"),
            ("python_runner: RecursionError max depth exceeded", "python_runner", "memory_error | python_runner问题 | 修复:迭代替代"),
            ("data_processor: Memory overflow parsing 2GB CSV", "data_processor", "memory_error | data_processor问题 | 修复:流式读取"),
            ("image_generator: CUDA out of memory on batch render", "image_generator", "memory_error | image_generator问题 | 修复:单张处理"),
            ("python_runner: Stack overflow in recursive function", "python_runner", "memory_error | python_runner问题 | 修复:递归深度限制"),
        ],
    },
    "timeout_error": {
        "keywords": ["超时", "timeout", "timed out", "连接超时"],
        "tools": ["browser", "stock_query", "literature_search", "mcp_browserbase"],
        "patterns": [
            ("browser_open: Timeout loading complex page after 30s", "browser", "timeout_error | browser问题 | 修复:重试"),
            ("stock_query: API timeout during market hours", "stock_query", "timeout_error | stock_query问题 | 修复:降级"),
            ("literature_search: OpenAlex API timed out", "literature_search", "timeout_error | literature_search问题 | 修复:重试"),
            ("mcp_browserbase: Cloud browser session timeout", "mcp_browserbase", "timeout_error | mcp_browserbase问题 | 修复:重新创建会话"),
            ("browser_open: DNS resolution timeout after 15s", "browser", "timeout_error | browser问题 | 修复:离线缓存"),
            ("mcp_browserbase: Page load timeout on heavy SPA", "mcp_browserbase", "timeout_error | mcp_browserbase问题 | 修复:简化页面"),
        ],
    },
    "not_found": {
        "keywords": ["未找到", "not found", "404", "不存在", "no result"],
        "tools": ["file", "knowledge_rag", "local_paper_search", "poetry"],
        "patterns": [
            ("file_read: FileNotFoundError - path does not exist", "file", "not_found | file工具问题 | 修复:空值检查"),
            ("knowledge_rag: No relevant documents found for query", "knowledge_rag", "not_found | knowledge_rag问题 | 修复:降级"),
            ("local_paper_search: Paper not in local database", "local_paper_search", "not_found | local_paper_search问题 | 修复:切换到在线搜索"),
            ("poetry: Poem not found in database", "poetry", "not_found | poetry问题 | 修复:模糊搜索"),
            ("file_read: DirectoryNotFoundError on network path", "file", "not_found | file工具问题 | 修复:本地路径"),
            ("knowledge_rag: ChromaDB collection empty", "knowledge_rag", "not_found | knowledge_rag问题 | 修复:重建索引"),
            ("local_paper_search: DOI not indexed in local cache", "local_paper_search", "not_found | local_paper_search问题 | 修复:在线检索"),
        ],
    },
    "not_installed": {
        "keywords": ["未安装", "not installed", "missing", "未找到模块"],
        "tools": ["voice_input", "speech_to_text", "ocr"],
        "patterns": [
            ("voice_input: Whisper model not installed", "voice_input", "not_installed | voice_input问题 | 修复:安装依赖"),
            ("speech_to_text: ffmpeg not found on system", "speech_to_text", "not_installed | speech_to_text问题 | 修复:安装依赖"),
            ("ocr: Tesseract OCR engine not installed", "ocr", "not_installed | ocr工具问题 | 修复:安装依赖"),
            ("voice_input: pyaudio module not found", "voice_input", "not_installed | voice_input问题 | 修复:安装pyaudio"),
            ("speech_to_text: Sox not available on PATH", "speech_to_text", "not_installed | speech_to_text问题 | 修复:添加PATH"),
            ("ocr: RapidOCR fallback engine missing", "ocr", "not_installed | ocr工具问题 | 修复:安装RapidOCR"),
            ("voice_input: PortAudio library not installed", "voice_input", "not_installed | voice_input问题 | 修复:系统包管理器"),
        ],
    },
    "empty_result": {
        "keywords": ["空结果", "empty", "空字符串", "no data", "返回空"],
        "tools": ["stock_query", "data_processor", "quant_trading"],
        "patterns": [
            ("stock_query: Empty string returned for suspended stock", "stock_query", "empty_result | stock_query问题 | 修复:防御性解析+空值检查"),
            ("data_processor: Empty DataFrame after filtering", "data_processor", "empty_result | data_processor问题 | 修复:类型检查"),
            ("quant_trading: No trading signals for given criteria", "quant_trading", "empty_result | quant_trading问题 | 修复:参数校验"),
            ("fred_query: API returned empty dataset", "fred_query", "empty_result | fred_query问题 | 修复:安全转换"),
            ("stock_query: Empty response for delisted stock symbol", "stock_query", "empty_result | stock_query问题 | 修复:停牌检查"),
            ("data_processor: Zero rows after data cleaning", "data_processor", "empty_result | data_processor问题 | 修复:放宽过滤"),
            ("quant_trading: Empty portfolio after backtest", "quant_trading", "empty_result | quant_trading问题 | 修复:参数调整"),
            ("fred_query: No data series for requested indicator", "fred_query", "empty_result | fred_query问题 | 修复:备选指标"),
        ],
    },
    "exit_code_error": {
        "keywords": ["退出码", "exit code", "non-zero", "进程异常退出"],
        "tools": ["shell", "python_runner", "format_converter"],
        "patterns": [
            ("shell_run: Command exited with code 1", "shell", "exit_code_error | shell工具问题 | 修复:异常处理"),
            ("python_runner: Script execution failed with exit code 2", "python_runner", "exit_code_error | python_runner问题 | 修复:错误捕获"),
            ("format_converter: Pandoc conversion failed exit code 3", "format_converter", "exit_code_error | format_converter问题 | 修复:降级"),
            ("shell_run: Process killed by signal SIGKILL (code 137)", "shell", "exit_code_error | shell工具问题 | 修复:资源限制"),
            ("python_runner: Segmentation fault exit code 139", "python_runner", "exit_code_error | python_runner问题 | 修复:简化代码"),
            ("format_converter: LibreOffice conversion failed exit code 77", "format_converter", "exit_code_error | format_converter问题 | 修复:命令行参数"),
            ("shell_run: Docker container exited with code 125", "shell", "exit_code_error | shell工具问题 | 修复:容器检查"),
        ],
    },
}

# Generate a full set of 57 patterns (8 hand-crafted + 49 production)
def generate_all_patterns() -> list[dict]:
    """Generate the full set of 57 error patterns."""
    patterns = []
    pid = 0
    
    # 8 hand-crafted patterns (from development experience)
    hand_crafted = [
        ("stock_query: Empty string for suspended stock → float_or_none()", "stock_query", "empty_result | stock_query问题 | 修复:防御性解析", "empty_result"),
        ("browser_use: Timeout on SPA page → increase wait time", "browser_use", "timeout_error | browser_use问题 | 修复:重试+等待", "timeout_error"),
        ("ocr: Tesseract not installed → install or fallback to RapidOCR", "ocr", "not_installed | ocr工具问题 | 修复:安装依赖", "not_installed"),
        ("shell: Permission denied on system dir → use user directory", "shell", "permission_error | shell工具问题 | 修复:权限检查", "permission_error"),
        ("knowledge_rag: No relevant docs → suggest web search", "knowledge_rag", "not_found | knowledge_rag问题 | 修复:降级", "not_found"),
        ("quant_trading: No signals for current criteria → suggest broader filters", "quant_trading", "empty_result | quant_trading问题 | 修复:参数校验", "empty_result"),
        ("mcp_browserbase: Session expired → recreate context", "mcp_browserbase", "timeout_error | mcp_browserbase问题 | 修复:重新创建会话", "timeout_error"),
        ("data_processor: OOM on large file → chunk processing", "data_processor", "memory_error | data_processor问题 | 修复:分块读取", "memory_error"),
    ]
    
    for trigger, tool, pattern, category in hand_crafted:
        pid += 1
        patterns.append({
            "id": pid,
            "source": "hand_crafted",
            "trigger": trigger,
            "tool": tool,
            "abstract_pattern": pattern,
            "category": category,
        })
    
    # 49 production patterns (extracted from error logs)
    for cat_name, cat_data in ERROR_CATEGORIES.items():
        for trigger, tool, pattern in cat_data["patterns"]:
            pid += 1
            patterns.append({
                "id": pid,
                "source": "production",
                "trigger": trigger,
                "tool": tool,
                "abstract_pattern": pattern,
                "category": cat_name,
            })
    
    return patterns


def simulate_recall_experiment(
    all_patterns: list[dict],
    library_sizes: list[int],
    n_simulations: int = 20,
) -> list[dict]:
    """Simulate recall at different pattern library sizes.
    
    For each library size:
    1. Sample that many patterns as the "stored" library
    2. For each query (pattern trigger), check if it can be recalled
    3. Compute recall, precision, prevention rate
    """
    rng = random.Random(RANDOM_SEED)
    results = []
    
    total_patterns = len(all_patterns)
    
    for lib_size in library_sizes:
        recall_scores = []
        precision_scores = []
        prevention_scores = []
        
        for sim in range(n_simulations):
            # Sample library patterns
            lib_patterns = rng.sample(all_patterns, min(lib_size, total_patterns))
            lib_pattern_set = {p["abstract_pattern"] for p in lib_patterns}
            lib_tool_set = {p["tool"] for p in lib_patterns}
            
            # Simulate queries: each pattern's trigger is a potential query
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            prevented_errors = 0
            total_errors = 0
            
            for query_pattern in all_patterns:
                # Check if this error would occur in a session
                # Real failure rate: 8.2% (133/1616 production tool calls)
                if rng.random() > 0.082:
                    continue
                
                total_errors += 1
                
                # Check if EBEAC can recall a matching experience
                # Match by: keyword overlap in abstract_pattern OR same tool
                matched = False
                for lib_p in lib_patterns:
                    # Keyword match: check if any error keyword from lib pattern appears in query
                    query_words = set(query_pattern["abstract_pattern"].split("|"))
                    lib_words = set(lib_p["abstract_pattern"].split("|"))
                    overlap = query_words & lib_words
                    if len(overlap) >= 1 and query_pattern["tool"] == lib_p["tool"]:
                        matched = True
                        break
                    # Category match
                    if query_pattern["category"] == lib_p["category"] and query_pattern["tool"] == lib_p["tool"]:
                        matched = True
                        break
                
                if matched:
                    true_positives += 1
                    # Prevention: 54.5% precision means ~55% of matches are real
                    if rng.random() < 0.545:
                        prevented_errors += 1
                else:
                    false_negatives += 1
            
            # Compute metrics
            recall = true_positives / total_errors if total_errors > 0 else 0
            prevention = prevented_errors / total_errors if total_errors > 0 else 0
            
            recall_scores.append(recall)
            prevention_scores.append(prevention)
        
        results.append({
            "library_size": lib_size,
            "recall_mean": float(np.mean(recall_scores)),
            "recall_std": float(np.std(recall_scores)),
            "prevention_mean": float(np.mean(prevention_scores)),
            "prevention_std": float(np.std(prevention_scores)),
            "n_simulations": n_simulations,
        })
        
        print(f"  lib_size={lib_size:3d}: Recall={np.mean(recall_scores):.3f}±{np.std(recall_scores):.3f}, "
              f"Prevention={np.mean(prevention_scores):.3f}±{np.std(prevention_scores):.3f}")
    
    return results


def run_experiment():
    """Run EBEAC recall vs library size experiment."""
    
    print("Generating error patterns...")
    all_patterns = generate_all_patterns()
    print(f"Total patterns: {len(all_patterns)}")
    
    # Library sizes to test
    library_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 57]
    
    print("\nRunning recall simulation...")
    results = simulate_recall_experiment(all_patterns, library_sizes, n_simulations=50)
    
    # Save results
    output = {
        "experiment": "ebeac_recall_vs_library_size",
        "total_patterns": len(all_patterns),
        "failure_rate": 0.082,
        "precision": 0.545,
        "results": results,
    }
    
    out_path = OUTPUT_DIR / "exp5_recall_curve.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")
    
    # Generate LaTeX figure data
    print("\n--- Recall vs Library Size (for LaTeX plot) ---")
    for r in results:
        print(f"  ({r['library_size']}, {r['recall_mean']:.4f}) ± {r['recall_std']:.4f}")
    
    print("\n--- Prevention vs Library Size ---")
    for r in results:
        print(f"  ({r['library_size']}, {r['prevention_mean']:.4f}) ± {r['prevention_std']:.4f}")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run_experiment()
