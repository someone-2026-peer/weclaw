#!/usr/bin/env python3
"""Expand the WeClaw tool-selection dataset from 200 to 500 items."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_utils import read_json, write_json


def scenario(
    sub_intent: str,
    primary_tool: str,
    acceptable_tools: list[str],
    negative_tools: list[str],
    notes: str,
    zh_templates: list[str],
    en_templates: list[str],
    mix_templates: list[str],
    slots: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "sub_intent": sub_intent,
        "primary_tool": primary_tool,
        "acceptable_tools": acceptable_tools,
        "negative_tools": negative_tools,
        "notes": notes,
        "templates": {
            "zh": zh_templates,
            "en": en_templates,
            "zh-en": mix_templates,
        },
        "slots": slots,
    }


SCENARIOS: dict[str, list[dict[str, Any]]] = {
    "browser_automation": [
        scenario(
            "open_and_extract",
            "browser",
            ["browser", "browser_use"],
            ["search", "file"],
            "轻量网页读取任务。",
            ["打开{site_zh}并提取{target_zh}", "访问{site_zh}后读取{target_zh}"],
            ["Open {site_en} and extract the {target_en}.", "Visit {site_en} and read the {target_en}."],
            ["Open {site_en} 并提取{target_zh}。", "Visit {site_en} 后读取 {target_en}。"],
            [
                {"site_zh": "产品发布页", "site_en": "the product launch page", "target_zh": "主标题", "target_en": "main headline"},
                {"site_zh": "学校官网新闻页", "site_en": "the university news page", "target_zh": "顶部标题", "target_en": "top title"},
                {"site_zh": "活动公告页", "site_en": "the event announcement page", "target_zh": "摘要文本", "target_en": "summary text"},
                {"site_zh": "帮助中心首页", "site_en": "the help-center homepage", "target_zh": "首屏标题", "target_en": "first visible heading"},
            ],
        ),
        scenario(
            "form_automation",
            "browser_use",
            ["browser_use", "browser", "mcp_browserbase"],
            ["search", "ocr"],
            "多步网页交互自动化。",
            ["自动填写{site_zh}上的{action_zh}", "帮我在{site_zh}里完成{action_zh}"],
            ["Automatically complete {action_en} on {site_en}.", "Use the browser to finish {action_en} on {site_en}."],
            ["在 {site_en} 上自动完成{action_zh}。", "Use browser automation 在 {site_en} 里处理 {action_en}。"],
            [
                {"site_zh": "报销系统", "site_en": "the reimbursement portal", "action_zh": "费用报销表单", "action_en": "the expense reimbursement form"},
                {"site_zh": "招聘后台", "site_en": "the recruiting dashboard", "action_zh": "候选人筛选表单", "action_en": "the candidate filtering form"},
                {"site_zh": "会议预约站点", "site_en": "the meeting booking site", "action_zh": "预约登记", "action_en": "the booking workflow"},
                {"site_zh": "项目管理平台", "site_en": "the project management portal", "action_zh": "状态更新流程", "action_en": "the status update flow"},
            ],
        ),
        scenario(
            "cloud_browser_access",
            "mcp_browserbase",
            ["mcp_browserbase", "browser_use", "browser"],
            ["search", "screen"],
            "云端浏览器访问。",
            ["用云端浏览器打开{site_zh}并抓取{target_zh}", "通过 browserbase 访问{site_zh}并读取{target_zh}"],
            ["Open {site_en} with a cloud browser and capture the {target_en}.", "Use Browserbase to access {site_en} and read the {target_en}."],
            ["Use browserbase 打开 {site_en} 并抓取{target_zh}。", "通过 cloud browser 访问 {site_en} 并读取 {target_en}。"],
            [
                {"site_zh": "内部仪表盘", "site_en": "the internal dashboard", "target_zh": "关键指标文本", "target_en": "key metrics"},
                {"site_zh": "会员专区页面", "site_en": "the member-only page", "target_zh": "正文内容", "target_en": "main body text"},
                {"site_zh": "登录后报表页面", "site_en": "the post-login report page", "target_zh": "导出区域", "target_en": "export section"},
                {"site_zh": "实验监控面板", "site_en": "the experiment monitor panel", "target_zh": "状态摘要", "target_en": "status summary"},
            ],
        ),
        scenario(
            "open_and_capture",
            "browser",
            ["browser", "browser_use", "screen"],
            ["search"],
            "网页打开与截图。",
            ["打开{site_zh}后截取{target_zh}", "访问{site_zh}并截图{target_zh}"],
            ["Open {site_en} and capture the {target_en}.", "Visit {site_en} and take a screenshot of the {target_en}."],
            ["Open {site_en} 后截取{target_zh}。", "访问 {site_en} 并 screenshot the {target_en}。"],
            [
                {"site_zh": "官网首页", "site_en": "the company homepage", "target_zh": "首屏横幅", "target_en": "hero banner"},
                {"site_zh": "价格页", "site_en": "the pricing page", "target_zh": "套餐区域", "target_en": "pricing section"},
                {"site_zh": "活动落地页", "site_en": "the campaign landing page", "target_zh": "报名按钮区域", "target_en": "signup button area"},
                {"site_zh": "控制台首页", "site_en": "the console homepage", "target_zh": "顶部统计面板", "target_en": "top stats panel"},
            ],
        ),
    ],
    "file_operation": [
        scenario(
            "file_read",
            "file",
            ["file"],
            ["shell", "search"],
            "本地文件读取。",
            ["读取{path_zh}里的内容", "打开{path_zh}并返回文本"],
            ["Read the contents of {path_en}.", "Open {path_en} and return the text."],
            ["Read {path_en} 并返回内容。", "打开 {path_en} and show the text."],
            [
                {"path_zh": "reports/summary.txt", "path_en": "reports/summary.txt"},
                {"path_zh": "logs/error.log", "path_en": "logs/error.log"},
                {"path_zh": "notes/idea.md", "path_en": "notes/idea.md"},
                {"path_zh": "drafts/outline.md", "path_en": "drafts/outline.md"},
            ],
        ),
        scenario(
            "file_move",
            "file",
            ["file", "shell"],
            ["search", "browser"],
            "文件移动归档。",
            ["把{source_zh}移动到{target_zh}", "将{source_zh}里的文件转移到{target_zh}"],
            ["Move files from {source_en} into {target_en}.", "Relocate the contents of {source_en} to {target_en}."],
            ["Move {source_en} 到 {target_zh}。", "把 {source_zh} files 移到 {target_en}。"],
            [
                {"source_zh": "downloaded_pdfs", "source_en": "downloaded_pdfs", "target_zh": "paper_archive", "target_en": "paper_archive"},
                {"source_zh": "raw_exports", "source_en": "raw_exports", "target_zh": "processed_exports", "target_en": "processed_exports"},
                {"source_zh": "todo_backup", "source_en": "todo_backup", "target_zh": "archive_2026", "target_en": "archive_2026"},
                {"source_zh": "old_logs", "source_en": "old_logs", "target_zh": "log_archive", "target_en": "log_archive"},
            ],
        ),
        scenario(
            "batch_rename",
            "file",
            ["file", "shell"],
            ["browser", "ocr"],
            "批量重命名。",
            ["给{folder_zh}中的文件批量加上{prefix_zh}前缀", "把{folder_zh}里的文件统一改名为带{prefix_zh}前缀"],
            ["Batch rename files in {folder_en} with the prefix {prefix_en}.", "Add the prefix {prefix_en} to every file in {folder_en}."],
            ["Batch rename {folder_en} 里的文件，前缀用 {prefix_zh}。", "给 {folder_zh} files 加上 {prefix_en} prefix。"],
            [
                {"folder_zh": "results", "folder_en": "results", "prefix_zh": "20260604", "prefix_en": "20260604"},
                {"folder_zh": "exports", "folder_en": "exports", "prefix_zh": "final", "prefix_en": "final"},
                {"folder_zh": "draft_figures", "folder_en": "draft_figures", "prefix_zh": "v2", "prefix_en": "v2"},
                {"folder_zh": "meeting_notes", "folder_en": "meeting_notes", "prefix_zh": "backup", "prefix_en": "backup"},
            ],
        ),
        scenario(
            "archive_extract",
            "shell",
            ["shell", "file"],
            ["browser", "ocr"],
            "解压归档文件。",
            ["解压{archive_zh}并保留目录结构", "把{archive_zh}解压到当前工作目录"],
            ["Extract {archive_en} while keeping the directory structure.", "Unpack {archive_en} into the current workspace."],
            ["Extract {archive_en} 并保留原目录结构。", "把 {archive_zh} unpack 到当前目录。"],
            [
                {"archive_zh": "benchmarks.zip", "archive_en": "benchmarks.zip"},
                {"archive_zh": "paper_assets.zip", "archive_en": "paper_assets.zip"},
                {"archive_zh": "backup_20260604.zip", "archive_en": "backup_20260604.zip"},
                {"archive_zh": "dataset_bundle.zip", "archive_en": "dataset_bundle.zip"},
            ],
        ),
    ],
    "system_admin": [
        scenario(
            "launch_app",
            "app_control",
            ["app_control", "shell"],
            ["calculator", "browser"],
            "启动应用。",
            ["打开{app_zh}应用", "启动{app_zh}并切到前台"],
            ["Launch {app_en} and bring it to the front.", "Open {app_en} in the desktop session."],
            ["Launch {app_en} 并切到前台。", "打开 {app_zh} app。"],
            [
                {"app_zh": "记事本", "app_en": "Notepad"},
                {"app_zh": "设置", "app_en": "Settings"},
                {"app_zh": "资源管理器", "app_en": "File Explorer"},
                {"app_zh": "便笺", "app_en": "Sticky Notes"},
            ],
        ),
        scenario(
            "screen_capture",
            "screen",
            ["screen"],
            ["browser", "ocr"],
            "屏幕截图。",
            ["截取{target_zh}并保存图片", "为{target_zh}拍一张截图"],
            ["Take a screenshot of the {target_en}.", "Capture the {target_en} on screen."],
            ["Take a screenshot of {target_zh}。", "截取 the {target_en} 并保存。"],
            [
                {"target_zh": "当前桌面", "target_en": "current desktop"},
                {"target_zh": "活动窗口", "target_en": "active window"},
                {"target_zh": "设置面板", "target_en": "settings panel"},
                {"target_zh": "报告预览页", "target_en": "report preview"},
            ],
        ),
        scenario(
            "desktop_notification",
            "notify",
            ["notify", "cron"],
            ["todo", "browser"],
            "桌面通知。",
            ["发送一个通知提醒我{task_zh}", "弹出桌面通知告诉我{task_zh}"],
            ["Send a desktop notification to remind me to {task_en}.", "Pop up a notification saying {task_en}."],
            ["Send 一个 notification 提醒我{task_zh}。", "弹出 desktop notification: {task_en}."],
            [
                {"task_zh": "保存实验结果", "task_en": "save the experiment results"},
                {"task_zh": "检查日志输出", "task_en": "check the log output"},
                {"task_zh": "提交论文草稿", "task_en": "submit the paper draft"},
                {"task_zh": "备份当前目录", "task_en": "back up the current folder"},
            ],
        ),
        scenario(
            "clipboard_read",
            "clipboard",
            ["clipboard"],
            ["file", "screen"],
            "剪贴板读取。",
            ["读取剪贴板中的{item_zh}", "返回当前剪贴板里的{item_zh}"],
            ["Read the {item_en} from the clipboard.", "Return the current clipboard {item_en}."],
            ["Read clipboard 里的{item_zh}。", "返回当前 clipboard {item_en}。"],
            [
                {"item_zh": "文本", "item_en": "text"},
                {"item_zh": "链接", "item_en": "URL"},
                {"item_zh": "命令片段", "item_en": "command snippet"},
                {"item_zh": "临时笔记", "item_en": "temporary note"},
            ],
        ),
    ],
    "system_monitoring": [
        scenario(
            "resource_monitoring",
            "system_monitor",
            ["system_monitor", "shell"],
            ["browser", "file"],
            "资源监控。",
            ["查看当前{metric_zh}使用情况", "监控系统的{metric_zh}占用"],
            ["Inspect current {metric_en} usage.", "Monitor the system's {metric_en} consumption."],
            ["Check 当前{metric_zh} usage。", "Monitor the system {metric_en} load。"],
            [
                {"metric_zh": "CPU 和内存", "metric_en": "CPU and memory"},
                {"metric_zh": "网络与磁盘", "metric_en": "network and disk"},
                {"metric_zh": "GPU 和内存", "metric_en": "GPU and memory"},
                {"metric_zh": "系统资源", "metric_en": "system resources"},
            ],
        ),
        scenario(
            "disk_monitoring",
            "system_monitor",
            ["system_monitor", "shell"],
            ["file", "browser"],
            "磁盘监控。",
            ["检查{drive_zh}的剩余空间和占用率", "查看{drive_zh}最近的磁盘使用情况"],
            ["Check free space and usage on {drive_en}.", "Inspect recent disk usage for {drive_en}."],
            ["Check {drive_en} 的 disk usage。", "查看 {drive_zh} current free space。"],
            [
                {"drive_zh": "C 盘", "drive_en": "drive C"},
                {"drive_zh": "D 盘", "drive_en": "drive D"},
                {"drive_zh": "实验数据盘", "drive_en": "the experiment data drive"},
                {"drive_zh": "备份分区", "drive_en": "the backup partition"},
            ],
        ),
        scenario(
            "network_monitoring",
            "system_monitor",
            ["system_monitor", "shell"],
            ["browser", "search"],
            "网络监控。",
            ["查看{target_zh}的网络流量", "监控{target_zh}的带宽占用"],
            ["Inspect network traffic for {target_en}.", "Monitor bandwidth consumption of {target_en}."],
            ["Inspect {target_en} 的 network traffic。", "监控 {target_zh} bandwidth usage。"],
            [
                {"target_zh": "当前浏览器进程", "target_en": "the current browser process"},
                {"target_zh": "下载任务", "target_en": "the download task"},
                {"target_zh": "视频会议应用", "target_en": "the video meeting app"},
                {"target_zh": "同步服务", "target_en": "the sync service"},
            ],
        ),
        scenario(
            "app_resource_check",
            "system_monitor",
            ["system_monitor", "app_control"],
            ["browser", "search"],
            "应用资源检查。",
            ["查看{app_zh}是否占用异常内存", "检查{app_zh}的资源使用是否过高"],
            ["Check whether {app_en} is consuming too much memory.", "Inspect whether {app_en} shows abnormal resource usage."],
            ["Check {app_en} 是否占用异常 memory。", "Inspect {app_zh} resource usage。"],
            [
                {"app_zh": "浏览器", "app_en": "the browser"},
                {"app_zh": "数据库客户端", "app_en": "the database client"},
                {"app_zh": "实验脚本进程", "app_en": "the experiment process"},
                {"app_zh": "PDF 阅读器", "app_en": "the PDF reader"},
            ],
        ),
    ],
    "daily_assistant": [
        scenario(
            "weather_lookup",
            "weather",
            ["weather", "search"],
            ["datetime_tool", "browser"],
            "天气查询。",
            ["查询{city_zh}明天的天气和温度", "帮我看一下{city_zh}今天的天气预报"],
            ["Check tomorrow's weather in {city_en}.", "Look up today's weather forecast for {city_en}."],
            ["Check {city_en} 明天的 weather。", "查询 {city_zh} today's temperature and weather。"],
            [
                {"city_zh": "深圳", "city_en": "Shenzhen"},
                {"city_zh": "杭州", "city_en": "Hangzhou"},
                {"city_zh": "北京", "city_en": "Beijing"},
                {"city_zh": "东京", "city_en": "Tokyo"},
            ],
        ),
        scenario(
            "scheduled_reminder",
            "cron",
            ["cron", "notify", "datetime_tool", "todo"],
            ["search"],
            "定时提醒。",
            ["帮我在{time_zh}提醒{task_zh}", "设置一个{time_zh}执行的提醒：{task_zh}"],
            ["Set a reminder for {time_en} to {task_en}.", "Schedule a reminder at {time_en} for {task_en}."],
            ["Set a reminder 在 {time_zh} 提醒我{task_zh}。", "Schedule {task_en} at {time_en}。"],
            [
                {"time_zh": "今晚八点", "time_en": "8 PM tonight", "task_zh": "提交实验结果", "task_en": "submit the experiment results"},
                {"time_zh": "明早九点", "time_en": "9 AM tomorrow", "task_zh": "给导师发邮件", "task_en": "email the advisor"},
                {"time_zh": "周五下午三点", "time_en": "3 PM on Friday", "task_zh": "检查日志回放结果", "task_en": "check the replay logs"},
                {"time_zh": "下周一上午十点", "time_en": "10 AM next Monday", "task_zh": "整理 benchmark 附录", "task_en": "整理 benchmark appendix"},
            ],
        ),
        scenario(
            "time_lookup",
            "datetime_tool",
            ["datetime_tool", "weather"],
            ["calculator"],
            "时间查询。",
            ["查询{city_zh}现在几点", "帮我看一下{city_zh}当前时间"],
            ["Tell me the current time in {city_en}.", "What time is it now in {city_en}?"],
            ["Tell me {city_en} 现在几点。", "查询 {city_zh} current time。"],
            [
                {"city_zh": "纽约", "city_en": "New York"},
                {"city_zh": "伦敦", "city_en": "London"},
                {"city_zh": "悉尼", "city_en": "Sydney"},
                {"city_zh": "新加坡", "city_en": "Singapore"},
            ],
        ),
        scenario(
            "simple_calculation",
            "calculator",
            ["calculator"],
            ["datetime_tool", "search"],
            "简单计算。",
            ["帮我算一下{a}*{b}", "计算 {a}+{b}+{c} 的结果"],
            ["Calculate {a} multiplied by {b}.", "Compute {a} + {b} + {c}."],
            ["Calculate {a}*{b} 给我。", "帮我 compute {a}+{b}+{c}。"],
            [
                {"a": "128", "b": "76", "c": "19"},
                {"a": "235", "b": "48", "c": "64"},
                {"a": "97", "b": "88", "c": "42"},
                {"a": "156", "b": "23", "c": "87"},
            ],
        ),
    ],
    "knowledge": [
        scenario(
            "knowledge_base_search",
            "knowledge_rag",
            ["knowledge_rag", "file"],
            ["search", "browser"],
            "知识库检索。",
            ["在知识库中查找{topic_zh}相关说明", "检索本地知识库里的{topic_zh}笔记"],
            ["Search the local knowledge base for notes about {topic_en}.", "Find knowledge-base entries related to {topic_en}."],
            ["Search 知识库 for {topic_en}.", "检索本地 knowledge base 里的{topic_zh}说明。"],
            [
                {"topic_zh": "context compression", "topic_en": "context compression"},
                {"topic_zh": "tool exposure", "topic_en": "tool exposure"},
                {"topic_zh": "prompt cache", "topic_en": "prompt cache"},
                {"topic_zh": "orphan tool call", "topic_en": "orphan tool calls"},
            ],
        ),
        scenario(
            "poetry_lookup",
            "poetry",
            ["poetry", "search"],
            ["knowledge_rag", "literature_search"],
            "诗词检索。",
            ["帮我找一首关于{theme_zh}的古诗", "查询描写{theme_zh}的经典诗词"],
            ["Find a classical poem about {theme_en}.", "Look up a poem describing {theme_en}."],
            ["Find a poem about {theme_en}。", "帮我查一首写{theme_zh}的 classical poem。"],
            [
                {"theme_zh": "月夜", "theme_en": "moonlight"},
                {"theme_zh": "秋风", "theme_en": "autumn wind"},
                {"theme_zh": "春雨", "theme_en": "spring rain"},
                {"theme_zh": "故乡", "theme_en": "homesickness"},
            ],
        ),
        scenario(
            "batch_paper_analysis",
            "batch_paper_analyzer",
            ["batch_paper_analyzer", "literature_search"],
            ["search", "ocr"],
            "批量论文分析。",
            ["批量分析{topic_zh}相关论文的方法部分", "对这组{topic_zh}论文做批量摘要分析"],
            ["Batch analyze the methods sections of papers about {topic_en}.", "Run batch paper analysis on this {topic_en} paper set."],
            ["Batch analyze {topic_en} papers 的方法部分。", "对这批 {topic_zh} 论文做 summary analysis。"],
            [
                {"topic_zh": "agent routing", "topic_en": "agent routing"},
                {"topic_zh": "context compression", "topic_en": "context compression"},
                {"topic_zh": "tool-use benchmark", "topic_en": "tool-use benchmarks"},
                {"topic_zh": "retrieval augmentation", "topic_en": "retrieval augmentation"},
            ],
        ),
        scenario(
            "knowledge_note_comparison",
            "knowledge_rag",
            ["knowledge_rag", "file"],
            ["search", "browser"],
            "知识笔记对比。",
            ["比较两份关于{topic_zh}的本地笔记", "帮我对照知识库中两条{topic_zh}记录"],
            ["Compare two local notes about {topic_en}.", "Cross-check two knowledge-base entries on {topic_en}."],
            ["Compare two notes about {topic_en}。", "帮我对照两条 {topic_zh} 本地知识记录。"],
            [
                {"topic_zh": "runtime architecture", "topic_en": "runtime architecture"},
                {"topic_zh": "tool recall", "topic_en": "tool recall"},
                {"topic_zh": "event bus", "topic_en": "event buses"},
                {"topic_zh": "error prevention", "topic_en": "error prevention"},
            ],
        ),
    ],
    "chat_history_retrieval": [
        scenario(
            "conversation_lookup",
            "chat_history",
            ["chat_history"],
            ["search", "knowledge_rag"],
            "历史对话检索。",
            ["查一下我们之前关于{topic_zh}的聊天记录", "搜索以前讨论{topic_zh}的对话"],
            ["Find the earlier chat where we discussed {topic_en}.", "Retrieve my previous conversation about {topic_en}."],
            ["Find 我们之前关于 {topic_en} 的 chat。", "搜索 earlier chat 里讨论{topic_zh}的内容。"],
            [
                {"topic_zh": "ITR 复现", "topic_en": "the ITR reproduction"},
                {"topic_zh": "Applied Sciences 投稿包", "topic_en": "the Applied Sciences submission package"},
                {"topic_zh": "EBEAC replay 指标", "topic_en": "EBEAC replay metrics"},
                {"topic_zh": "tool exposure 路线", "topic_en": "tool exposure"},
            ],
        ),
        scenario(
            "conversation_lookup",
            "chat_history",
            ["chat_history"],
            ["search", "log_viewer"],
            "历史问题定位。",
            ["帮我找之前提过{topic_zh}问题的聊天", "查看以前关于{topic_zh} bug 的对话"],
            ["Locate the earlier chat about the {topic_en} bug.", "Find my old discussion of the {topic_en} issue."],
            ["Locate the earlier chat about {topic_en}。", "帮我找之前讨论{topic_zh} bug 的 chat。"],
            [
                {"topic_zh": "OCR 空路径", "topic_en": "OCR empty-path"},
                {"topic_zh": "browser_use 结构化输出", "topic_en": "browser_use structured output"},
                {"topic_zh": "tool recall 漏召回", "topic_en": "tool recall miss"},
                {"topic_zh": "context 压缩抖动", "topic_en": "context jitter"},
            ],
        ),
        scenario(
            "conversation_lookup",
            "chat_history",
            ["chat_history"],
            ["search", "system_monitor"],
            "历史方案检索。",
            ["检索之前关于{topic_zh}方案的聊天记录", "查一下我们以前怎么讨论{topic_zh}"],
            ["Retrieve the previous discussion about the {topic_en} plan.", "Find our earlier chat on {topic_en}."],
            ["Retrieve the previous discussion on {topic_en}。", "查一下 earlier chat 里关于{topic_zh}的方案。"],
            [
                {"topic_zh": "500 条数据集扩容", "topic_en": "the 500-item dataset expansion"},
                {"topic_zh": "benchmark artifact 发布", "topic_en": "benchmark artifact release"},
                {"topic_zh": "摘要修订", "topic_en": "the abstract revision"},
                {"topic_zh": "多篇论文路线", "topic_en": "the multi-paper roadmap"},
            ],
        ),
        scenario(
            "conversation_lookup",
            "chat_history",
            ["chat_history"],
            ["paper_lifecycle", "search"],
            "投稿路线聊天记录。",
            ["查找之前关于{topic_zh}投稿建议的对话", "搜索提到{topic_zh}的旧聊天"],
            ["Find the old chat mentioning {topic_en}.", "Locate my previous discussion about {topic_en}."],
            ["Find the old chat mentioning {topic_en}。", "搜索以前提到{topic_zh}的 conversation。"],
            [
                {"topic_zh": "IEEE Access", "topic_en": "IEEE Access"},
                {"topic_zh": "Applied Sciences", "topic_en": "Applied Sciences"},
                {"topic_zh": "Q3 OA 期刊", "topic_en": "Q3 open-access venues"},
                {"topic_zh": "EI 投稿路线", "topic_en": "the EI submission route"},
            ],
        ),
    ],
    "life_management": [
        scenario(
            "todo_management",
            "todo",
            ["todo", "cron"],
            ["notify", "search"],
            "待办管理。",
            ["添加一个待办：{task_zh}", "帮我创建{task_zh}这条待办"],
            ["Create a todo item to {task_en}.", "Add a task reminding me to {task_en}."],
            ["Create a todo item：{task_zh}。", "帮我 add a task to {task_en}。"],
            [
                {"task_zh": "整理 benchmark 表格", "task_en": "整理 the benchmark tables"},
                {"task_zh": "周五提交摘要", "task_en": "submit the abstract on Friday"},
                {"task_zh": "今晚检查日志", "task_en": "check the logs tonight"},
                {"task_zh": "更新论文附录", "task_en": "update the paper appendix"},
            ],
        ),
        scenario(
            "health_record",
            "health",
            ["health", "diary"],
            ["todo", "search"],
            "健康记录。",
            ["记录今天的{metric_zh}", "帮我保存今天的{metric_zh}数据"],
            ["Record today's {metric_en}.", "Save today's {metric_en} in my health log."],
            ["Record today's {metric_en}。", "帮我保存今天的{metric_zh}。"],
            [
                {"metric_zh": "体重和睡眠质量", "metric_en": "weight and sleep quality"},
                {"metric_zh": "血压和心率", "metric_en": "blood pressure and heart rate"},
                {"metric_zh": "运动时长", "metric_en": "exercise duration"},
                {"metric_zh": "步数统计", "metric_en": "step count"},
            ],
        ),
        scenario(
            "diary_entry",
            "diary",
            ["diary"],
            ["todo", "paper_lifecycle"],
            "日记记录。",
            ["写一条日记，内容是{topic_zh}", "把{topic_zh}记录到今天的日记里"],
            ["Write a diary entry about {topic_en}.", "Add a diary note describing {topic_en}."],
            ["Write a diary entry 关于 {topic_en}。", "把{topic_zh}写到 today's diary。"],
            [
                {"topic_zh": "今天完成了 exp10 复跑", "topic_en": "finishing the exp10 rerun"},
                {"topic_zh": "今天整理了投稿计划", "topic_en": "organizing the submission plan"},
                {"topic_zh": "今天扩展了 benchmark 数据", "topic_en": "expanding the benchmark dataset"},
                {"topic_zh": "今天修订了主稿附录", "topic_en": "revising the paper appendix"},
            ],
        ),
        scenario(
            "family_info_update",
            "family_member",
            ["family_member"],
            ["diary", "todo"],
            "家庭成员信息更新。",
            ["更新{person_zh}的联系方式", "查看并修改{person_zh}的家庭资料"],
            ["Update the contact details for {person_en}.", "Review and edit the family profile of {person_en}."],
            ["Update {person_en} 的联系方式。", "查看并修改 {person_zh} family profile。"],
            [
                {"person_zh": "父亲", "person_en": "my father"},
                {"person_zh": "母亲", "person_en": "my mother"},
                {"person_zh": "姐姐", "person_en": "my sister"},
                {"person_zh": "弟弟", "person_en": "my younger brother"},
            ],
        ),
    ],
    "financial_activity": [
        scenario(
            "stock_price_lookup",
            "stock_query",
            ["stock_query", "search"],
            ["fred_query", "browser"],
            "股票价格查询。",
            ["查询{asset_zh}的最新股价", "帮我看一下{asset_zh}现在的市场价格"],
            ["Check the latest price of {asset_en}.", "Look up the current market price for {asset_en}."],
            ["Check {asset_en} 最新股价。", "帮我看一下 {asset_zh} current price。"],
            [
                {"asset_zh": "英伟达", "asset_en": "NVIDIA"},
                {"asset_zh": "微软", "asset_en": "Microsoft"},
                {"asset_zh": "贵州茅台", "asset_en": "Kweichow Moutai"},
                {"asset_zh": "特斯拉", "asset_en": "Tesla"},
            ],
        ),
        scenario(
            "trading_signal_analysis",
            "quant_trading",
            ["quant_trading", "stock_query", "fred_query"],
            ["search"],
            "量化交易分析。",
            ["分析{asset_zh}最近的交易信号", "帮我评估{asset_zh}当前的量化买卖机会"],
            ["Analyze the latest trading signal for {asset_en}.", "Evaluate the current quant signal for {asset_en}."],
            ["Analyze {asset_en} 的 trading signal。", "帮我评估 {asset_zh} current quant opportunity。"],
            [
                {"asset_zh": "AAPL", "asset_en": "AAPL"},
                {"asset_zh": "腾讯控股", "asset_en": "Tencent"},
                {"asset_zh": "比亚迪", "asset_en": "BYD"},
                {"asset_zh": "台积电", "asset_en": "TSMC"},
            ],
        ),
        scenario(
            "macro_data_lookup",
            "fred_query",
            ["fred_query", "search"],
            ["stock_query", "browser"],
            "宏观数据查询。",
            ["获取{metric_zh}的时间序列数据", "查询{metric_zh}最近几年的走势"],
            ["Retrieve the time series for {metric_en}.", "Look up recent history for {metric_en}."],
            ["Retrieve {metric_en} time series。", "查询 {metric_zh} recent trend。"],
            [
                {"metric_zh": "美国 CPI", "metric_en": "US CPI"},
                {"metric_zh": "美国失业率", "metric_en": "US unemployment rate"},
                {"metric_zh": "联邦基金利率", "metric_en": "the federal funds rate"},
                {"metric_zh": "美国 GDP", "metric_en": "US GDP"},
            ],
        ),
        scenario(
            "risk_analysis",
            "quant_trading",
            ["quant_trading", "stock_query"],
            ["fred_query", "search"],
            "持仓风险分析。",
            ["评估{asset_zh}持仓的风险敞口", "帮我分析{asset_zh}当前仓位风险"],
            ["Analyze the position risk of {asset_en}.", "Evaluate the current portfolio risk for {asset_en}."],
            ["Analyze {asset_en} 持仓 risk。", "帮我评估 {asset_zh} current portfolio exposure。"],
            [
                {"asset_zh": "半导体组合", "asset_en": "the semiconductor portfolio"},
                {"asset_zh": "新能源仓位", "asset_en": "the EV position"},
                {"asset_zh": "银行股组合", "asset_en": "the bank-stock basket"},
                {"asset_zh": "AI 股票仓位", "asset_en": "the AI stock position"},
            ],
        ),
    ],
    "multimedia": [
        scenario(
            "ocr",
            "ocr",
            ["ocr", "file"],
            ["speech_to_text", "image_generator"],
            "OCR 文本识别。",
            ["识别{item_zh}里的文字", "帮我提取{item_zh}中的文本"],
            ["Extract text from {item_en}.", "Run OCR on {item_en} and return the text."],
            ["Extract text from {item_en}。", "帮我识别{item_zh}里的文字。"],
            [
                {"item_zh": "发票照片", "item_en": "a receipt image"},
                {"item_zh": "扫描合同", "item_en": "a scanned contract"},
                {"item_zh": "白板截图", "item_en": "a whiteboard screenshot"},
                {"item_zh": "快递单图片", "item_en": "a shipping-label photo"},
            ],
        ),
        scenario(
            "audio_transcription",
            "speech_to_text",
            ["speech_to_text", "voice_input"],
            ["ocr", "stock_photo"],
            "音频转写。",
            ["把{audio_zh}转成文字稿", "转写{audio_zh}里的语音内容"],
            ["Transcribe {audio_en} into text.", "Convert the speech in {audio_en} into notes."],
            ["Transcribe {audio_en} 成文字。", "把{audio_zh}里的语音转成 notes。"],
            [
                {"audio_zh": "会议录音", "audio_en": "the meeting recording"},
                {"audio_zh": "采访音频", "audio_en": "the interview audio"},
                {"audio_zh": "课程录音", "audio_en": "the lecture recording"},
                {"audio_zh": "语音备忘", "audio_en": "the voice memo"},
            ],
        ),
        scenario(
            "live_voice_transcription",
            "voice_input",
            ["voice_input", "speech_to_text"],
            ["ocr", "image_generator"],
            "实时语音输入。",
            ["实时记录我的{topic_zh}说明", "开启语音输入并转写{topic_zh}内容"],
            ["Start live voice input for my {topic_en} explanation.", "Use live transcription while I describe {topic_en}."],
            ["Use live voice input 记录我的{topic_zh}说明。", "Start voice transcription for my {topic_en} explanation."],
            [
                {"topic_zh": "实验设计", "topic_en": "experiment design"},
                {"topic_zh": "会议纪要", "topic_en": "meeting summary"},
                {"topic_zh": "代码讲解", "topic_en": "code walkthrough"},
                {"topic_zh": "投稿计划", "topic_en": "submission plan"},
            ],
        ),
        scenario(
            "stock_photo_search",
            "stock_photo",
            ["stock_photo", "search"],
            ["image_generator", "ocr"],
            "图库图片搜索。",
            ["搜索一张关于{theme_zh}的免费图片", "帮我找适合{theme_zh}主题的图库图"],
            ["Find a free stock image about {theme_en}.", "Search for a stock photo suitable for {theme_en}."],
            ["Find a free image about {theme_en}。", "帮我找适合{theme_zh}的 stock photo。"],
            [
                {"theme_zh": "论文封面", "theme_en": "a paper cover"},
                {"theme_zh": "桌面 AI 助手", "theme_en": "a desktop AI assistant"},
                {"theme_zh": "数据分析报告", "theme_en": "a data-analysis report"},
                {"theme_zh": "学术演示封面", "theme_en": "an academic presentation cover"},
            ],
        ),
    ],
    "document_processing": [
        scenario(
            "format_conversion",
            "format_converter",
            ["format_converter", "pdf_tool", "doc_generator"],
            ["oss_pdf_download", "ocr"],
            "格式转换。",
            ["把{src_zh}转换成{dst_zh}", "将{src_zh}导出为{dst_zh}"],
            ["Convert {src_en} into {dst_en}.", "Export {src_en} as {dst_en}."],
            ["Convert {src_en} 成 {dst_zh}。", "将{src_zh} export as {dst_en}。"],
            [
                {"src_zh": "markdown 报告", "src_en": "a markdown report", "dst_zh": "PDF", "dst_en": "PDF"},
                {"src_zh": "docx 草稿", "src_en": "a docx draft", "dst_zh": "PDF", "dst_en": "PDF"},
                {"src_zh": "研究笔记", "src_en": "research notes", "dst_zh": "docx", "dst_en": "DOCX"},
                {"src_zh": "会议纪要", "src_en": "meeting notes", "dst_zh": "PDF", "dst_en": "PDF"},
            ],
        ),
        scenario(
            "pdf_merge",
            "pdf_tool",
            ["pdf_tool"],
            ["format_converter", "doc_generator"],
            "PDF 合并。",
            ["把这批{doc_zh}合并成一个 PDF", "将多个{doc_zh}整合为单个 PDF 文件"],
            ["Merge these {doc_en} into one PDF.", "Combine multiple {doc_en} as a single PDF."],
            ["Merge these {doc_en} 成一个 PDF。", "把多份{doc_zh}合并为 single PDF。"],
            [
                {"doc_zh": "会议纪要", "doc_en": "meeting notes"},
                {"doc_zh": "论文附录", "doc_en": "appendix drafts"},
                {"doc_zh": "实验结果页", "doc_en": "result sheets"},
                {"doc_zh": "扫描件", "doc_en": "scanned pages"},
            ],
        ),
        scenario(
            "ppt_generation",
            "ppt_generator",
            ["ppt_generator", "doc_generator", "stock_photo"],
            ["pdf_tool", "format_converter"],
            "PPT 生成。",
            ["根据{topic_zh}生成一页式 PPT", "把{topic_zh}整理成演示文稿摘要"],
            ["Create a short slide deck about {topic_en}.", "Generate a one-page presentation for {topic_en}."],
            ["Create a slide deck about {topic_en}。", "把{topic_zh}整理成 PPT 摘要。"],
            [
                {"topic_zh": "benchmark 结果", "topic_en": "benchmark results"},
                {"topic_zh": "投稿计划", "topic_en": "the submission plan"},
                {"topic_zh": "系统架构", "topic_en": "the system architecture"},
                {"topic_zh": "数据集扩展", "topic_en": "dataset expansion"},
            ],
        ),
        scenario(
            "doc_generation",
            "doc_generator",
            ["doc_generator", "format_converter", "ppt_generator"],
            ["pdf_tool"],
            "文档生成。",
            ["根据{topic_zh}生成一份 Word 文档", "把{topic_zh}整理成 docx 报告"],
            ["Generate a DOCX document about {topic_en}.", "Create a Word report for {topic_en}."],
            ["Generate a DOCX about {topic_en}。", "把{topic_zh}整理成 Word 报告。"],
            [
                {"topic_zh": "实验汇总", "topic_en": "the experiment summary"},
                {"topic_zh": "附录说明", "topic_en": "the appendix notes"},
                {"topic_zh": "会议纪要", "topic_en": "the meeting notes"},
                {"topic_zh": "研究备忘", "topic_en": "the research memo"},
            ],
        ),
    ],
    "data_analysis": [
        scenario(
            "tabular_processing",
            "data_processor",
            ["data_processor", "statistics", "file"],
            ["search", "browser"],
            "表格处理。",
            ["读取{file_zh}并计算总和", "处理{file_zh}中的表格数据"],
            ["Read {file_en} and compute the totals.", "Process the spreadsheet data in {file_en}."],
            ["Read {file_en} 并计算总量。", "处理 {file_zh} 中的 tabular data。"],
            [
                {"file_zh": "sales.csv", "file_en": "sales.csv"},
                {"file_zh": "budget.xlsx", "file_en": "budget.xlsx"},
                {"file_zh": "metrics.csv", "file_en": "metrics.csv"},
                {"file_zh": "survey.xlsx", "file_en": "survey.xlsx"},
            ],
        ),
        scenario(
            "chart_generation",
            "data_visualization",
            ["data_visualization", "statistics"],
            ["data_processor", "pdf_tool"],
            "图表生成。",
            ["为{topic_zh}绘制图表", "把{topic_zh}做成可视化图"],
            ["Create a chart for {topic_en}.", "Visualize {topic_en} as a figure."],
            ["Create a chart for {topic_en}。", "把{topic_zh}画成 visualization。"],
            [
                {"topic_zh": "月度收入趋势", "topic_en": "monthly revenue trends"},
                {"topic_zh": "日活用户变化", "topic_en": "daily active users"},
                {"topic_zh": "实验耗时分布", "topic_en": "experiment runtime distribution"},
                {"topic_zh": "模型准确率对比", "topic_en": "model accuracy comparison"},
            ],
        ),
        scenario(
            "statistical_summary",
            "statistics",
            ["statistics", "data_processor"],
            ["data_visualization", "search"],
            "统计摘要。",
            ["统计{topic_zh}的均值和方差", "给出{topic_zh}的统计摘要"],
            ["Compute summary statistics for {topic_en}.", "Calculate the mean and variance of {topic_en}."],
            ["Compute stats for {topic_en}。", "统计{topic_zh}的 mean and variance。"],
            [
                {"topic_zh": "样本表格", "topic_en": "the sample table"},
                {"topic_zh": "实验重复结果", "topic_en": "the repeated experiment results"},
                {"topic_zh": "温度测量数据", "topic_en": "the temperature measurements"},
                {"topic_zh": "A/B 测试数据", "topic_en": "the A/B test dataset"},
            ],
        ),
        scenario(
            "data_cleaning",
            "data_processor",
            ["data_processor", "statistics"],
            ["data_visualization", "search"],
            "数据清洗。",
            ["清洗{topic_zh}中的缺失值和重复项", "处理{topic_zh}里的脏数据"],
            ["Clean missing values and duplicates in {topic_en}.", "Prepare {topic_en} by removing dirty records."],
            ["Clean {topic_en} 里的缺失值。", "处理 {topic_zh} 中的 dirty data。"],
            [
                {"topic_zh": "实验记录表", "topic_en": "the experiment table"},
                {"topic_zh": "订单数据", "topic_en": "the order dataset"},
                {"topic_zh": "问卷结果", "topic_en": "the survey results"},
                {"topic_zh": "训练日志表", "topic_en": "the training log table"},
            ],
        ),
    ],
    "research": [
        scenario(
            "literature_search",
            "literature_search",
            ["literature_search", "search"],
            ["knowledge_rag", "ocr"],
            "学术搜索。",
            ["搜索最近关于{topic_zh}的论文", "帮我查找{topic_zh}方向的学术文献"],
            ["Search recent papers about {topic_en}.", "Look for academic papers on {topic_en}."],
            ["Search recent papers about {topic_en}。", "帮我查找{topic_zh}方向 literature。"],
            [
                {"topic_zh": "tool routing", "topic_en": "tool routing"},
                {"topic_zh": "agent memory replay", "topic_en": "agent memory replay"},
                {"topic_zh": "context compression", "topic_en": "context compression"},
                {"topic_zh": "desktop agent benchmark", "topic_en": "desktop agent benchmarks"},
            ],
        ),
        scenario(
            "open_pdf_lookup",
            "oss_pdf_search",
            ["oss_pdf_search", "literature_search"],
            ["oss_pdf_download", "search"],
            "开放 PDF 查询。",
            ["检查{topic_zh}论文是否有开放 PDF", "查询{topic_zh}这篇文章能否下载开放全文"],
            ["Check whether an open-access PDF exists for {topic_en}.", "See if {topic_en} has a free PDF."],
            ["Check whether {topic_en} 有 open-access PDF。", "查询{topic_zh}是否可下载开放全文。"],
            [
                {"topic_zh": "这篇 agent 系统论文", "topic_en": "this agent systems paper"},
                {"topic_zh": "这个 DOI 对应论文", "topic_en": "this DOI"},
                {"topic_zh": "这篇 benchmark 文章", "topic_en": "this benchmark paper"},
                {"topic_zh": "这篇 memory study", "topic_en": "this memory study"},
            ],
        ),
        scenario(
            "journal_selection",
            "journal_intelligence",
            ["journal_intelligence", "literature_search"],
            ["paper_lifecycle", "search"],
            "期刊选择。",
            ["评估{topic_zh}论文适合投哪些期刊", "帮我判断{topic_zh}稿件的投稿方向"],
            ["Assess which journals fit a paper on {topic_en}.", "Evaluate suitable venues for a manuscript about {topic_en}."],
            ["Assess suitable journals for {topic_en}。", "帮我判断{topic_zh}稿件投稿方向。"],
            [
                {"topic_zh": "桌面 agent runtime", "topic_en": "desktop agent runtime"},
                {"topic_zh": "tool routing benchmark", "topic_en": "tool routing benchmarks"},
                {"topic_zh": "experience replay", "topic_en": "experience replay"},
                {"topic_zh": "context management", "topic_en": "context management"},
            ],
        ),
        scenario(
            "paper_project_management",
            "paper_lifecycle",
            ["paper_lifecycle", "journal_intelligence"],
            ["doc_generator", "search"],
            "论文项目管理。",
            ["创建一个关于{topic_zh}的新论文项目", "为{topic_zh}研究建立投稿项目"],
            ["Create a new paper project about {topic_en}.", "Set up a manuscript workflow for {topic_en}."],
            ["Create a new paper project about {topic_en}。", "为{topic_zh}建立 manuscript workflow。"],
            [
                {"topic_zh": "adaptive runtime", "topic_en": "adaptive runtime"},
                {"topic_zh": "tool-selection benchmark", "topic_en": "tool-selection benchmarks"},
                {"topic_zh": "EBEAC replay", "topic_en": "EBEAC replay"},
                {"topic_zh": "RCR pipeline", "topic_en": "the RCR pipeline"},
            ],
        ),
    ],
    "self_reflection": [
        scenario(
            "log_inspection",
            "log_viewer",
            ["log_viewer", "tool_audit", "experience_recall"],
            ["search", "browser"],
            "日志检查。",
            ["查看最近与{topic_zh}相关的错误日志", "检查日志中是否再次出现{topic_zh}"],
            ["Inspect recent logs related to {topic_en}.", "Check whether {topic_en} appears again in the logs."],
            ["Inspect recent logs for {topic_en}。", "检查日志中是否再次出现{topic_zh}。"],
            [
                {"topic_zh": "empty_result", "topic_en": "empty_result"},
                {"topic_zh": "structured_output_unsupported", "topic_en": "structured_output_unsupported"},
                {"topic_zh": "file_not_found", "topic_en": "file_not_found"},
                {"topic_zh": "rotated_log_missing", "topic_en": "rotated_log_missing"},
            ],
        ),
        scenario(
            "tool_audit",
            "tool_audit",
            ["tool_audit", "log_viewer"],
            ["search", "browser"],
            "工具审计。",
            ["生成一份关于{topic_zh}的工具审计报告", "帮我汇总最近{topic_zh}相关的调用链"],
            ["Generate an audit report for {topic_en}.", "Summarize the recent tool chain related to {topic_en}."],
            ["Generate an audit report for {topic_en}。", "帮我汇总最近{topic_zh}相关调用链。"],
            [
                {"topic_zh": "浏览器自动化失败", "topic_en": "browser automation failures"},
                {"topic_zh": "OCR 调用错误", "topic_en": "OCR failures"},
                {"topic_zh": "exp10 benchmark 运行", "topic_en": "the exp10 benchmark run"},
                {"topic_zh": "context 压缩异常", "topic_en": "context compression issues"},
            ],
        ),
        scenario(
            "code_lookup",
            "codebase_search",
            ["codebase_search"],
            ["tool_audit", "search"],
            "代码定位。",
            ["搜索代码库中{topic_zh}的实现位置", "帮我找{topic_zh}在哪个模块里实现"],
            ["Search the codebase for where {topic_en} is implemented.", "Locate the module implementing {topic_en}."],
            ["Search the codebase for {topic_en}。", "帮我找{topic_zh}在哪个 module。"],
            [
                {"topic_zh": "ToolExposureEngine", "topic_en": "ToolExposureEngine"},
                {"topic_zh": "experience recall", "topic_en": "experience recall"},
                {"topic_zh": "context compressor", "topic_en": "the context compressor"},
                {"topic_zh": "event bus", "topic_en": "the event bus"},
            ],
        ),
        scenario(
            "experience_lookup",
            "experience_recall",
            ["experience_recall", "log_viewer", "codebase_search"],
            ["search"],
            "历史经验回忆。",
            ["回忆一下之前关于{topic_zh}的修复经验", "查找以前出现过的{topic_zh}案例"],
            ["Recall previous fixes for {topic_en}.", "Look up earlier cases related to {topic_en}."],
            ["Recall previous fixes for {topic_en}。", "查找以前出现过的{topic_zh}案例。"],
            [
                {"topic_zh": "empty_result bug", "topic_en": "the empty_result bug"},
                {"topic_zh": "browser_use 失败", "topic_en": "browser_use failures"},
                {"topic_zh": "ocr 文件缺失", "topic_en": "OCR file-missing cases"},
                {"topic_zh": "格式转换异常", "topic_en": "format-conversion failures"},
            ],
        ),
    ],
}


BASE_LANGUAGE_PLAN = {"zh": 11, "en": 6, "zh-en": 4}
EXTRA_LANGUAGE_PLAN = {
    "browser_automation": "en",
    "file_operation": "en",
    "system_admin": "en",
    "knowledge": "en",
    "life_management": "en",
    "document_processing": "zh-en",
}


def render_query(scn: dict[str, Any], language: str, serial: int) -> str:
    templates = scn["templates"][language]
    template = templates[serial % len(templates)]
    slot = scn["slots"][serial % len(scn["slots"])]
    return template.format(**slot)


def next_query_id(items: list[dict[str, Any]]) -> int:
    ids = []
    for item in items:
        query_id = str(item.get("query_id", ""))
        if query_id.startswith("ts"):
            try:
                ids.append(int(query_id[2:]))
            except ValueError:
                pass
    return max(ids, default=0) + 1


def generate_items(existing_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    additions: list[dict[str, Any]] = []
    current_id = next_query_id(existing_items)
    per_intent_counters: dict[str, int] = {intent: 0 for intent in SCENARIOS}

    for intent, scenarios in SCENARIOS.items():
        language_plan = dict(BASE_LANGUAGE_PLAN)
        extra_language = EXTRA_LANGUAGE_PLAN.get(intent)
        if extra_language:
            language_plan[extra_language] += 1

        for language, count in language_plan.items():
            for offset in range(count):
                scenario_idx = offset % len(scenarios)
                scenario_round = offset // len(scenarios)
                scn = scenarios[scenario_idx]
                serial = per_intent_counters[intent] + scenario_round + scenario_idx
                additions.append(
                    {
                        "query_id": f"ts{current_id:03d}",
                        "query": render_query(scn, language, serial),
                        "language": language,
                        "primary_intent": intent,
                        "sub_intent": scn["sub_intent"],
                        "acceptable_tools": scn["acceptable_tools"],
                        "primary_tool": scn["primary_tool"],
                        "negative_tools": scn["negative_tools"],
                        "notes": scn["notes"],
                    }
                )
                current_id += 1
            per_intent_counters[intent] += count

    return additions


def validate(payload: dict[str, Any]) -> None:
    items = payload["items"]
    if len(items) != payload["target_size"]:
        raise ValueError(f"Expected {payload['target_size']} items, got {len(items)}")

    query_ids = [item["query_id"] for item in items]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query_id detected.")

    intents = Counter(item["primary_intent"] for item in items)
    languages = Counter(item["language"] for item in items)
    positive = Counter()
    for item in items:
        for tool in item["acceptable_tools"]:
            positive[tool] += 1

    min_positive = payload["coverage_constraints"]["high_frequency_tools_min_positive_examples"]
    missing = {
        tool: positive[tool]
        for tool in payload["coverage_constraints"]["high_frequency_tools"]
        if positive[tool] < min_positive
    }
    if missing:
        raise ValueError(f"Coverage constraint failed: {missing}")

    if min(intents.values()) < 35 or max(intents.values()) > 37:
        raise ValueError(f"Intent distribution out of expected range: {dict(intents)}")

    if min(languages.values()) < 90:
        raise ValueError(f"Language distribution too skewed: {dict(languages)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand WeClaw tool-selection dataset to 500 items.")
    parser.add_argument("--input", required=True, help="Path to current weclaw_tool_selection_500.json")
    parser.add_argument("--output", required=True, help="Path to output expanded JSON")
    args = parser.parse_args()

    payload = read_json(args.input)
    if not isinstance(payload, dict):
        raise TypeError("Dataset payload must be a JSON object.")
    existing_items = payload["items"]
    if len(existing_items) != 200:
        raise ValueError(f"Expected 200 existing items, got {len(existing_items)}")

    additions = generate_items(existing_items)
    if len(additions) != 300:
        raise ValueError(f"Expected 300 additions, got {len(additions)}")

    payload["version"] = "0.4.0-sample500"
    payload["annotation_status"] = "balanced_first_500_with_coverage_constraints"
    payload["notes"] = (
        "This is the first balanced 500-query release candidate for tool-selection benchmarking. "
        "It covers 14 primary intents with Chinese, English, and mixed-language queries, and "
        "enforces minimum positive coverage for high-frequency core tools."
    )
    payload["items"] = existing_items + additions

    validate(payload)
    write_json(args.output, payload)
    print("Expanded tool-selection dataset generated.")


if __name__ == "__main__":
    main()
