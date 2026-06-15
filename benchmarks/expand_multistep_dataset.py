#!/usr/bin/env python3
"""Expand the WeClaw multistep benchmark dataset to paper-grade scale."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from dataset_utils import read_json, write_json


def workflow(
    task_template: str,
    domain: str,
    difficulty: str,
    steps: list[dict[str, Any]],
    success_criteria: list[str],
    slots: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "task_template": task_template,
        "domain": domain,
        "difficulty": difficulty,
        "steps": steps,
        "success_criteria": success_criteria,
        "slots": slots,
    }


WORKFLOWS: list[dict[str, Any]] = [
    workflow(
        "搜索关于{topic_zh}的论文并下载开放 PDF",
        "research",
        "medium",
        [
            {"instruction": "搜索最近关于{topic_zh}的论文", "acceptable_tools": ["literature_search", "search"], "notes": "学术搜索。"},
            {"instruction": "检查候选论文是否存在开放获取 PDF", "acceptable_tools": ["oss_pdf_search", "literature_search"], "notes": "开放 PDF 检索。"},
            {"instruction": "下载开放 PDF 到 {folder_zh}", "acceptable_tools": ["oss_pdf_download", "oss_pdf_search", "file"], "notes": "下载并保存。"},
        ],
        ["找到相关论文", "确认存在开放 PDF", "成功下载 PDF"],
        [
            {"topic_zh": "agent runtime", "folder_zh": "paper_archive"},
            {"topic_zh": "tool routing", "folder_zh": "literature_cache"},
            {"topic_zh": "context compression", "folder_zh": "research_pdfs"},
        ],
    ),
    workflow(
        "评估{topic_zh}论文的投稿方向并建立项目",
        "research",
        "hard",
        [
            {"instruction": "搜索关于{topic_zh}的代表性论文", "acceptable_tools": ["literature_search", "search"], "notes": "先找代表论文。"},
            {"instruction": "评估这类研究适合投稿哪些期刊", "acceptable_tools": ["journal_intelligence", "literature_search"], "notes": "期刊评估。"},
            {"instruction": "创建一个新的论文写作项目并设置阶段", "acceptable_tools": ["paper_lifecycle", "journal_intelligence"], "notes": "论文项目管理。"},
        ],
        ["找到代表论文", "完成期刊评估", "建立论文项目"],
        [
            {"topic_zh": "desktop agent benchmark"},
            {"topic_zh": "experience replay"},
            {"topic_zh": "hybrid tool routing"},
        ],
    ),
    workflow(
        "分析{topic_zh}研究领域并整理研究地图",
        "research",
        "hard",
        [
            {"instruction": "搜索关于{topic_zh}的代表性论文", "acceptable_tools": ["literature_search", "search"], "notes": "论文搜索。"},
            {"instruction": "生成该方向的 research landscape", "acceptable_tools": ["research_landscape", "literature_search"], "notes": "研究地图。"},
            {"instruction": "追踪关键方法的发展谱系", "acceptable_tools": ["research_lineage", "literature_search"], "notes": "研究谱系。"},
            {"instruction": "把结论整理成一份简短研究备忘", "acceptable_tools": ["doc_generator", "file"], "notes": "形成备忘。"},
        ],
        ["找到代表论文", "得到研究地图", "得到谱系分析", "形成研究备忘"],
        [
            {"topic_zh": "tool-use benchmark"},
            {"topic_zh": "agent memory"},
            {"topic_zh": "context management"},
        ],
    ),
    workflow(
        "查询{asset_zh}价格并输出简要风险判断",
        "financial_activity",
        "medium",
        [
            {"instruction": "查询{asset_zh}最新股价", "acceptable_tools": ["stock_query", "search"], "notes": "股价查询。"},
            {"instruction": "分析当前持仓风险并给出一句话建议", "acceptable_tools": ["quant_trading", "stock_query"], "notes": "风险分析。"},
        ],
        ["成功查询价格", "成功输出风险判断"],
        [
            {"asset_zh": "AAPL"},
            {"asset_zh": "腾讯控股"},
            {"asset_zh": "比亚迪"},
        ],
    ),
    workflow(
        "获取{metric_zh}并写成宏观研究备忘",
        "financial_activity",
        "hard",
        [
            {"instruction": "从 FRED 查询{metric_zh}时间序列", "acceptable_tools": ["fred_query", "search"], "notes": "宏观数据查询。"},
            {"instruction": "搜索最近关于{metric_zh}影响的研究论文", "acceptable_tools": ["literature_search", "search"], "notes": "相关研究。"},
            {"instruction": "将数据和文献信息整理成研究备忘", "acceptable_tools": ["doc_generator", "file"], "notes": "写研究备忘。"},
        ],
        ["获取宏观数据", "找到相关研究", "形成研究备忘"],
        [
            {"metric_zh": "美国 CPI"},
            {"metric_zh": "美国失业率"},
            {"metric_zh": "联邦基金利率"},
        ],
    ),
    workflow(
        "分析{asset_zh}交易信号并生成图表摘要",
        "financial_activity",
        "hard",
        [
            {"instruction": "查询{asset_zh}最近价格走势", "acceptable_tools": ["stock_query", "search"], "notes": "价格走势。"},
            {"instruction": "分析{asset_zh}当前量化交易信号", "acceptable_tools": ["quant_trading", "stock_query"], "notes": "量化信号分析。"},
            {"instruction": "绘制相关收益与波动图表", "acceptable_tools": ["data_visualization", "statistics"], "notes": "图表生成。"},
            {"instruction": "整理成一页简报摘要", "acceptable_tools": ["doc_generator", "ppt_generator"], "notes": "摘要整理。"},
        ],
        ["完成价格查询", "完成信号分析", "生成图表", "完成摘要简报"],
        [
            {"asset_zh": "半导体组合"},
            {"asset_zh": "新能源仓位"},
            {"asset_zh": "纳斯达克指数"},
        ],
    ),
    workflow(
        "查看系统状态并清理{folder_zh}",
        "system_admin",
        "easy",
        [
            {"instruction": "查看当前 CPU、内存和磁盘使用率", "acceptable_tools": ["system_monitor", "shell"], "notes": "系统资源检查。"},
            {"instruction": "清理{folder_zh}目录中的临时文件", "acceptable_tools": ["file", "shell"], "notes": "本地文件清理。"},
            {"instruction": "发送桌面通知说明清理完成", "acceptable_tools": ["notify", "cron"], "notes": "发送通知。"},
        ],
        ["完成状态检查", "完成临时文件清理", "已通知用户"],
        [
            {"folder_zh": "temp_cache"},
            {"folder_zh": "download_cache"},
            {"folder_zh": "old_logs"},
        ],
    ),
    workflow(
        "检查{app_zh}资源占用并截屏留档",
        "system_admin",
        "medium",
        [
            {"instruction": "查看{app_zh}当前的 CPU 和内存占用", "acceptable_tools": ["system_monitor", "app_control"], "notes": "资源检查。"},
            {"instruction": "截取当前异常窗口画面", "acceptable_tools": ["screen"], "notes": "截图留档。"},
            {"instruction": "发送通知提示资源检查完成", "acceptable_tools": ["notify", "cron"], "notes": "通知。"},
        ],
        ["完成资源检查", "完成截图留档", "发送完成通知"],
        [
            {"app_zh": "浏览器"},
            {"app_zh": "数据库客户端"},
            {"app_zh": "PDF 阅读器"},
        ],
    ),
    workflow(
        "启动{app_zh}、读取剪贴板并保存记录",
        "system_admin",
        "medium",
        [
            {"instruction": "启动{app_zh}并切换到前台", "acceptable_tools": ["app_control", "shell"], "notes": "打开应用。"},
            {"instruction": "读取当前剪贴板文本", "acceptable_tools": ["clipboard"], "notes": "剪贴板读取。"},
            {"instruction": "将读取结果保存到本地文件", "acceptable_tools": ["file"], "notes": "保存结果。"},
        ],
        ["启动应用", "读取剪贴板", "保存结果"],
        [
            {"app_zh": "记事本"},
            {"app_zh": "设置"},
            {"app_zh": "资源管理器"},
        ],
    ),
    workflow(
        "读取{file_zh}并生成统计图表",
        "data_analysis",
        "medium",
        [
            {"instruction": "读取{file_zh}并清洗缺失值", "acceptable_tools": ["data_processor", "file"], "notes": "数据清洗。"},
            {"instruction": "计算均值与方差", "acceptable_tools": ["statistics", "data_processor"], "notes": "统计摘要。"},
            {"instruction": "绘制相关趋势图", "acceptable_tools": ["data_visualization", "statistics"], "notes": "图表绘制。"},
        ],
        ["完成数据清洗", "完成统计计算", "生成趋势图"],
        [
            {"file_zh": "sales.csv"},
            {"file_zh": "metrics.xlsx"},
            {"file_zh": "survey.csv"},
        ],
    ),
    workflow(
        "清洗{file_zh}并导出分析报告",
        "data_analysis",
        "medium",
        [
            {"instruction": "读取{file_zh}并处理重复项", "acceptable_tools": ["data_processor", "file"], "notes": "清洗数据。"},
            {"instruction": "计算关键统计指标", "acceptable_tools": ["statistics", "data_processor"], "notes": "统计分析。"},
            {"instruction": "将分析结论整理为文档", "acceptable_tools": ["doc_generator", "file"], "notes": "形成报告。"},
        ],
        ["读取并清洗数据", "完成统计分析", "导出分析报告"],
        [
            {"file_zh": "orders.xlsx"},
            {"file_zh": "benchmark_results.csv"},
            {"file_zh": "latency_logs.csv"},
        ],
    ),
    workflow(
        "读取{file_zh}并生成演示图表简报",
        "data_analysis",
        "hard",
        [
            {"instruction": "读取{file_zh}并准备作图数据", "acceptable_tools": ["data_processor", "file"], "notes": "准备数据。"},
            {"instruction": "绘制主要对比图表", "acceptable_tools": ["data_visualization", "statistics"], "notes": "制作图表。"},
            {"instruction": "将图表整理为一页简报", "acceptable_tools": ["ppt_generator", "doc_generator"], "notes": "形成简报。"},
        ],
        ["完成数据准备", "完成图表制作", "生成简报"],
        [
            {"file_zh": "ablation_results.csv"},
            {"file_zh": "dataset_distribution.xlsx"},
            {"file_zh": "runtime_metrics.csv"},
        ],
    ),
    workflow(
        "把{audio_zh}转写并整理成文档",
        "multimedia",
        "medium",
        [
            {"instruction": "将{audio_zh}转成文字稿", "acceptable_tools": ["speech_to_text", "voice_input"], "notes": "音频转写。"},
            {"instruction": "把转写内容整理为会议纪要文档", "acceptable_tools": ["doc_generator", "format_converter"], "notes": "文档生成。"},
            {"instruction": "导出一份 PDF 版本", "acceptable_tools": ["format_converter", "pdf_tool"], "notes": "导出 PDF。"},
        ],
        ["完成语音转写", "生成文档", "导出 PDF"],
        [
            {"audio_zh": "meeting.wav"},
            {"audio_zh": "interview_recording.mp3"},
            {"audio_zh": "lecture_audio.m4a"},
        ],
    ),
    workflow(
        "识别{image_zh}并整理成可读文档",
        "multimedia",
        "medium",
        [
            {"instruction": "对{image_zh}执行 OCR 识别", "acceptable_tools": ["ocr", "file"], "notes": "执行 OCR。"},
            {"instruction": "将识别文字整理为结构化文档", "acceptable_tools": ["doc_generator", "format_converter"], "notes": "整理文本。"},
            {"instruction": "导出 PDF 版本", "acceptable_tools": ["format_converter", "pdf_tool"], "notes": "导出 PDF。"},
        ],
        ["完成 OCR", "形成文档", "导出 PDF"],
        [
            {"image_zh": "发票图片"},
            {"image_zh": "扫描合同"},
            {"image_zh": "白板截图"},
        ],
    ),
    workflow(
        "搜索{theme_zh}图片并制作演示封面",
        "multimedia",
        "hard",
        [
            {"instruction": "搜索与{theme_zh}相关的免费图片", "acceptable_tools": ["stock_photo", "search"], "notes": "图片搜索。"},
            {"instruction": "选择一张合适图片并生成封面页", "acceptable_tools": ["ppt_generator", "doc_generator", "stock_photo"], "notes": "封面制作。"},
            {"instruction": "导出一页式简报", "acceptable_tools": ["ppt_generator", "format_converter"], "notes": "导出简报。"},
        ],
        ["找到图片", "生成封面页", "导出简报"],
        [
            {"theme_zh": "桌面 AI 助手"},
            {"theme_zh": "学术演示"},
            {"theme_zh": "benchmark 报告"},
        ],
    ),
    workflow(
        "打开{site_zh}并提取标题后截图保存",
        "browser_automation",
        "medium",
        [
            {"instruction": "打开{site_zh}并等待首页加载完成", "acceptable_tools": ["browser", "browser_use", "mcp_browserbase"], "notes": "打开网页。"},
            {"instruction": "提取页面主标题文本", "acceptable_tools": ["browser", "browser_use"], "notes": "读取标题。"},
            {"instruction": "截取首屏并保存为图片", "acceptable_tools": ["screen", "browser", "browser_use"], "notes": "截图保存。"},
        ],
        ["成功打开网页", "成功提取标题", "成功保存截图"],
        [
            {"site_zh": "官网首页"},
            {"site_zh": "帮助中心页面"},
            {"site_zh": "活动落地页"},
        ],
    ),
    workflow(
        "登录{site_zh}后台并导出结果页面",
        "browser_automation",
        "hard",
        [
            {"instruction": "打开{site_zh}登录页面", "acceptable_tools": ["browser_use", "browser", "mcp_browserbase"], "notes": "打开登录页。"},
            {"instruction": "自动填写账号并完成登录", "acceptable_tools": ["browser_use", "mcp_browserbase", "browser"], "notes": "网页登录自动化。"},
            {"instruction": "导出当前报表页面内容", "acceptable_tools": ["browser_use", "browser", "mcp_browserbase"], "notes": "导出页面。"},
        ],
        ["打开登录页", "完成登录", "导出结果页面"],
        [
            {"site_zh": "实验监控面板"},
            {"site_zh": "报销系统"},
            {"site_zh": "项目管理平台"},
        ],
    ),
    workflow(
        "访问{site_zh}并读取正文后保存到文件",
        "browser_automation",
        "medium",
        [
            {"instruction": "使用浏览器打开{site_zh}", "acceptable_tools": ["browser", "browser_use", "mcp_browserbase"], "notes": "访问网页。"},
            {"instruction": "提取页面正文内容", "acceptable_tools": ["browser", "browser_use"], "notes": "读取正文。"},
            {"instruction": "把正文保存到本地笔记文件", "acceptable_tools": ["file", "doc_generator"], "notes": "保存正文。"},
        ],
        ["成功访问网页", "成功读取正文", "成功保存文件"],
        [
            {"site_zh": "新闻页面"},
            {"site_zh": "内部公告页"},
            {"site_zh": "会员专区页面"},
        ],
    ),
    workflow(
        "在知识库中查{topic_zh}并写入项目笔记",
        "knowledge",
        "medium",
        [
            {"instruction": "在知识库中搜索{topic_zh}的说明", "acceptable_tools": ["knowledge_rag", "file"], "notes": "知识库检索。"},
            {"instruction": "把找到的要点整理成三条摘要", "acceptable_tools": ["doc_generator", "file"], "notes": "摘要整理。"},
            {"instruction": "把摘要保存到项目 notes.md 中", "acceptable_tools": ["file", "doc_generator"], "notes": "写入笔记。"},
        ],
        ["找到相关知识", "形成摘要", "保存到项目笔记"],
        [
            {"topic_zh": "context compression"},
            {"topic_zh": "tool exposure"},
            {"topic_zh": "prompt cache"},
        ],
    ),
    workflow(
        "搜索{topic_zh}诗词并整理成学习卡片",
        "knowledge",
        "easy",
        [
            {"instruction": "搜索关于{topic_zh}的经典诗词", "acceptable_tools": ["poetry", "search"], "notes": "查诗词。"},
            {"instruction": "整理诗句与简短解释", "acceptable_tools": ["doc_generator", "file"], "notes": "整理解释。"},
            {"instruction": "保存到学习卡片文件", "acceptable_tools": ["file", "doc_generator"], "notes": "保存卡片。"},
        ],
        ["找到诗词", "形成解释", "保存卡片"],
        [
            {"topic_zh": "月夜"},
            {"topic_zh": "秋风"},
            {"topic_zh": "故乡"},
        ],
    ),
    workflow(
        "批量分析{topic_zh}论文并生成摘要备忘",
        "knowledge",
        "hard",
        [
            {"instruction": "搜索与{topic_zh}相关的论文", "acceptable_tools": ["literature_search", "search"], "notes": "搜索论文。"},
            {"instruction": "批量分析这些论文的方法部分", "acceptable_tools": ["batch_paper_analyzer", "literature_search"], "notes": "批量分析。"},
            {"instruction": "把分析结果整理成备忘文档", "acceptable_tools": ["doc_generator", "file"], "notes": "形成备忘。"},
        ],
        ["找到相关论文", "完成批量分析", "形成摘要备忘"],
        [
            {"topic_zh": "tool-use benchmark"},
            {"topic_zh": "experience replay"},
            {"topic_zh": "hybrid routing"},
        ],
    ),
    workflow(
        "创建提醒并记录{task_zh}到待办与日记",
        "life_management",
        "easy",
        [
            {"instruction": "创建一个提醒：{task_zh}", "acceptable_tools": ["cron", "notify", "datetime_tool"], "notes": "创建提醒。"},
            {"instruction": "新增一个与{task_zh}相关的待办事项", "acceptable_tools": ["todo"], "notes": "添加待办。"},
            {"instruction": "在日记中记录今天的准备情况", "acceptable_tools": ["diary"], "notes": "写日记。"},
        ],
        ["提醒已创建", "待办已创建", "日记已记录"],
        [
            {"task_zh": "明天下午三点提交论文"},
            {"task_zh": "周五晚上整理 benchmark"},
            {"task_zh": "下周一更新附录说明"},
        ],
    ),
    workflow(
        "记录{metric_zh}并整理健康摘要",
        "life_management",
        "medium",
        [
            {"instruction": "记录今天的{metric_zh}", "acceptable_tools": ["health", "diary"], "notes": "健康记录。"},
            {"instruction": "将记录内容整理成简短摘要", "acceptable_tools": ["doc_generator", "file"], "notes": "整理摘要。"},
            {"instruction": "把摘要保存到健康日志", "acceptable_tools": ["file", "diary"], "notes": "写入日志。"},
        ],
        ["完成健康记录", "形成摘要", "保存到日志"],
        [
            {"metric_zh": "体重和睡眠质量"},
            {"metric_zh": "血压和心率"},
            {"metric_zh": "运动时长"},
        ],
    ),
    workflow(
        "更新{person_zh}资料并记录变更说明",
        "life_management",
        "medium",
        [
            {"instruction": "查看{person_zh}当前家庭资料", "acceptable_tools": ["family_member"], "notes": "查看资料。"},
            {"instruction": "更新{person_zh}的联系方式", "acceptable_tools": ["family_member"], "notes": "更新联系方式。"},
            {"instruction": "将变更内容记录到日记", "acceptable_tools": ["diary"], "notes": "记录变更。"},
        ],
        ["查看资料", "更新联系方式", "完成变更记录"],
        [
            {"person_zh": "父亲"},
            {"person_zh": "母亲"},
            {"person_zh": "姐姐"},
        ],
    ),
    workflow(
        "查询{city_zh}天气并生成出行建议",
        "daily_assistant",
        "easy",
        [
            {"instruction": "查询{city_zh}明天的天气和温度", "acceptable_tools": ["weather", "search"], "notes": "天气查询。"},
            {"instruction": "补充{city_zh}当前时间", "acceptable_tools": ["datetime_tool"], "notes": "时间查询。"},
            {"instruction": "给出一句话出行建议", "acceptable_tools": ["doc_generator", "file"], "notes": "生成建议。"},
        ],
        ["获得天气信息", "获得时间信息", "完成出行建议"],
        [
            {"city_zh": "上海"},
            {"city_zh": "深圳"},
            {"city_zh": "杭州"},
        ],
    ),
    workflow(
        "安排{task_zh}并发送完成提醒",
        "daily_assistant",
        "easy",
        [
            {"instruction": "为{task_zh}设置定时提醒", "acceptable_tools": ["cron", "notify", "datetime_tool"], "notes": "定时提醒。"},
            {"instruction": "创建一个相关待办事项", "acceptable_tools": ["todo", "cron"], "notes": "待办创建。"},
            {"instruction": "发送一条桌面通知确认设置完成", "acceptable_tools": ["notify", "cron"], "notes": "发送通知。"},
        ],
        ["完成提醒设置", "完成待办创建", "发送确认通知"],
        [
            {"task_zh": "周五下午提交稿件"},
            {"task_zh": "今晚检查日志结果"},
            {"task_zh": "明早同步 benchmark 表格"},
        ],
    ),
    workflow(
        "查询{city_zh}当前时间并完成快速计算",
        "daily_assistant",
        "easy",
        [
            {"instruction": "查询{city_zh}现在几点", "acceptable_tools": ["datetime_tool"], "notes": "时间查询。"},
            {"instruction": "帮我算一下 {expr_zh}", "acceptable_tools": ["calculator"], "notes": "简单计算。"},
            {"instruction": "把结果整理成一句提醒文本", "acceptable_tools": ["doc_generator", "file"], "notes": "文字整理。"},
        ],
        ["获得时间", "完成计算", "整理提醒文本"],
        [
            {"city_zh": "纽约", "expr_zh": "235*48"},
            {"city_zh": "伦敦", "expr_zh": "128+76+19"},
            {"city_zh": "新加坡", "expr_zh": "97*88"},
        ],
    ),
    workflow(
        "在历史对话和日志中定位{topic_zh}",
        "self_reflection",
        "hard",
        [
            {"instruction": "搜索之前关于{topic_zh}的聊天记录", "acceptable_tools": ["chat_history"], "notes": "历史聊天回溯。"},
            {"instruction": "查看最近日志中是否再次出现{topic_zh}", "acceptable_tools": ["log_viewer", "tool_audit"], "notes": "日志检查。"},
            {"instruction": "回忆是否有相似修复经验", "acceptable_tools": ["experience_recall", "tool_audit"], "notes": "经验回忆。"},
            {"instruction": "定位相关代码实现位置", "acceptable_tools": ["codebase_search"], "notes": "代码定位。"},
        ],
        ["找到相关聊天", "确认日志情况", "找到相似经验", "定位代码位置"],
        [
            {"topic_zh": "empty_result"},
            {"topic_zh": "structured_output_unsupported"},
            {"topic_zh": "file_not_found"},
        ],
    ),
    workflow(
        "生成{topic_zh}的审计报告并回溯日志",
        "self_reflection",
        "hard",
        [
            {"instruction": "生成最近与{topic_zh}相关的审计报告", "acceptable_tools": ["tool_audit", "log_viewer"], "notes": "生成审计报告。"},
            {"instruction": "查看对应的失败日志", "acceptable_tools": ["log_viewer", "tool_audit"], "notes": "日志回溯。"},
            {"instruction": "查找是否有相似经验可供参考", "acceptable_tools": ["experience_recall", "log_viewer"], "notes": "查经验。"},
        ],
        ["生成审计报告", "完成日志回溯", "找到相似经验"],
        [
            {"topic_zh": "browser 自动化失败"},
            {"topic_zh": "OCR 调用错误"},
            {"topic_zh": "benchmark 复跑异常"},
        ],
    ),
    workflow(
        "搜索{topic_zh}实现并整理定位说明",
        "self_reflection",
        "medium",
        [
            {"instruction": "搜索代码库中 {topic_zh} 的实现位置", "acceptable_tools": ["codebase_search"], "notes": "代码搜索。"},
            {"instruction": "查看最近相关日志或审计信息", "acceptable_tools": ["log_viewer", "tool_audit"], "notes": "日志与审计。"},
            {"instruction": "把定位结果整理成说明文档", "acceptable_tools": ["doc_generator", "file"], "notes": "形成说明。"},
        ],
        ["找到实现位置", "完成相关信息检查", "形成定位说明"],
        [
            {"topic_zh": "ToolExposureEngine"},
            {"topic_zh": "experience recall"},
            {"topic_zh": "context compressor"},
        ],
    ),
    workflow(
        "整理{topic_zh}文档并生成演示材料",
        "document_processing",
        "medium",
        [
            {"instruction": "把相关文档转换为 PDF", "acceptable_tools": ["format_converter", "pdf_tool"], "notes": "转换文档。"},
            {"instruction": "生成一份 Word 摘要文档", "acceptable_tools": ["doc_generator", "format_converter"], "notes": "生成摘要。"},
            {"instruction": "基于文档内容生成一页式 PPT", "acceptable_tools": ["ppt_generator", "doc_generator"], "notes": "生成 PPT。"},
        ],
        ["完成格式转换", "生成摘要文档", "生成 PPT"],
        [
            {"topic_zh": "benchmark 结果"},
            {"topic_zh": "投稿计划"},
            {"topic_zh": "系统架构说明"},
        ],
    ),
    workflow(
        "合并{topic_zh}文件并输出最终报告",
        "document_processing",
        "medium",
        [
            {"instruction": "把相关 PDF 文件合并成一个文档", "acceptable_tools": ["pdf_tool"], "notes": "合并 PDF。"},
            {"instruction": "将结论整理为 docx 报告", "acceptable_tools": ["doc_generator", "format_converter"], "notes": "形成报告。"},
            {"instruction": "导出最终 PDF 版本", "acceptable_tools": ["format_converter", "pdf_tool"], "notes": "导出最终 PDF。"},
        ],
        ["完成 PDF 合并", "生成 docx 报告", "导出最终 PDF"],
        [
            {"topic_zh": "附录说明"},
            {"topic_zh": "实验结果页"},
            {"topic_zh": "投稿材料"},
        ],
    ),
    workflow(
        "把{topic_zh}内容整理成汇报简报",
        "document_processing",
        "hard",
        [
            {"instruction": "读取并整理与{topic_zh}相关的文档要点", "acceptable_tools": ["file", "doc_generator"], "notes": "整理要点。"},
            {"instruction": "生成一份演示用 PPT 草稿", "acceptable_tools": ["ppt_generator", "doc_generator"], "notes": "生成 PPT 草稿。"},
            {"instruction": "导出为可分享 PDF", "acceptable_tools": ["format_converter", "pdf_tool"], "notes": "导出 PDF。"},
        ],
        ["整理文档要点", "生成演示草稿", "导出分享版本"],
        [
            {"topic_zh": "ablation study"},
            {"topic_zh": "benchmark artifact"},
            {"topic_zh": "submission roadmap"},
        ],
    ),
]


def render_workflow_item(flow: dict[str, Any], variant_idx: int, task_number: int) -> dict[str, Any]:
    slot = flow["slots"][variant_idx % len(flow["slots"])]
    steps = []
    for step_idx, step in enumerate(flow["steps"], start=1):
        steps.append(
            {
                "step_id": f"s{step_idx}",
                "instruction": step["instruction"].format(**slot),
                "acceptable_tools": step["acceptable_tools"],
                "notes": step["notes"],
            }
        )
    return {
        "task_id": f"ms{task_number:03d}",
        "task": flow["task_template"].format(**slot),
        "domain": flow["domain"],
        "difficulty": flow["difficulty"],
        "steps": steps,
        "success_criteria": flow["success_criteria"],
    }


def build_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    task_number = 1
    for flow in WORKFLOWS:
        for variant_idx in range(len(flow["slots"])):
            tasks.append(render_workflow_item(flow, variant_idx, task_number))
            task_number += 1
    return tasks


def validate(tasks: list[dict[str, Any]]) -> None:
    if not (50 <= len(tasks) <= 100):
        raise ValueError(f"Expected 50-100 tasks, got {len(tasks)}")
    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task_id detected.")

    step_count = 0
    domain_counter = Counter()
    difficulty_counter = Counter()
    for task in tasks:
        domain_counter[task["domain"]] += 1
        difficulty_counter[task["difficulty"]] += 1
        steps = task["steps"]
        if not (2 <= len(steps) <= 5):
            raise ValueError(f"Task {task['task_id']} has invalid step count {len(steps)}")
        step_ids = [step["step_id"] for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError(f"Duplicate step_id in {task['task_id']}")
        step_count += len(steps)

    if len(domain_counter) < 9:
        raise ValueError(f"Expected broad domain coverage, got {dict(domain_counter)}")
    if "hard" not in difficulty_counter or "easy" not in difficulty_counter:
        raise ValueError(f"Difficulty distribution too narrow: {dict(difficulty_counter)}")
    if step_count < 150:
        raise ValueError(f"Expected at least 150 total steps, got {step_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand WeClaw multistep benchmark dataset.")
    parser.add_argument("--input", required=True, help="Path to current weclaw_multistep_100.json")
    parser.add_argument("--output", required=True, help="Path to output expanded JSON")
    args = parser.parse_args()

    payload = read_json(args.input)
    if not isinstance(payload, dict):
        raise TypeError("Dataset payload must be a JSON object.")

    tasks = build_tasks()
    validate(tasks)

    payload["version"] = "0.2.0-release-candidate99"
    payload["notes"] = (
        "Paper-grade multistep release candidate for exp8 execution-level evaluation. "
        "This version expands the formal sample to 99 tasks with broad domain coverage, "
        "2-4 steps per task, and diversified execution-level tool-routing patterns."
    )
    payload["items"] = tasks
    write_json(args.output, payload)
    print("Expanded multistep dataset generated.")


if __name__ == "__main__":
    main()
