#!/usr/bin/env python3
"""E11 public benchmark query pool (ToolBench-lite style, 100 entries).

Extends ``exp7_toolbench_lite.BUILTIN_TOOLBENCH_LITE`` (12 seed entries) to a
100-query, multi-domain, download-free public set for the E11 real-execution
benchmark. Queries are original API-style instructions (no external dataset is
copied), and every ``acceptable_tools`` entry is a valid name in the exp1
75-tool universe so the harness can expose and select it.

Each entry is a tuple ``(id, domain, query_en, acceptable_tools)``, plus a
parallel Chinese rendering in ``_PUBLIC_ZH`` (id -> zh query). The E11 primary
run uses the Chinese text so the whole 200q set is language-consistent with the
production intent detector (``detect_intent_with_confidence``), whose keyword
categories are Chinese; the English originals are retained for the documented
cross-lingual boundary run. Only the surface query text varies -- the ground
truth (``acceptable_tools``/``domain``/split) is held fixed across languages.
The ``in_sandbox`` flag (whether the success path lands in one of the 15 sandbox
families) is computed by ``e11_query_set.py``, not stored here.

Domain mix (~57 in-sandbox / ~43 out-of-sandbox) mirrors ToolBench's real-API
character: generic web/finance/multimedia tasks fall outside the sandbox and
become capability errors, while weather/data/document/file/math/knowledge tasks
have a genuine in-sandbox success path.
"""
from __future__ import annotations

# (id, domain, query, acceptable_tools) -- acceptable_tools ⊆ exp1 75-tool universe
_PUBLIC: list[tuple[str, str, str, list[str]]] = [
    # ---- weather (in-sandbox: weather) ----
    ("pb_weather_01", "weather", "What is the weather like in Tokyo today?", ["weather", "search"]),
    ("pb_weather_02", "weather", "Check the weekend forecast for London.", ["weather"]),
    ("pb_weather_03", "weather", "Is it going to rain in Singapore tomorrow?", ["weather", "search"]),
    ("pb_weather_04", "weather", "Get the current temperature and humidity in Paris.", ["weather"]),
    ("pb_weather_05", "weather", "Compare the weather between Sydney and Seoul.", ["weather"]),
    ("pb_weather_06", "weather", "What is the wind speed in New York right now?", ["weather"]),
    # ---- data & analytics (in-sandbox: data_processor/statistics/file) ----
    ("pb_data_01", "data", "Read sales.csv and compute the total revenue.", ["data_processor", "statistics", "file"]),
    ("pb_data_02", "data", "Calculate the average and standard deviation of this column.", ["statistics", "data_processor"]),
    ("pb_data_03", "data", "Load the grades spreadsheet and summarize the scores.", ["data_processor", "statistics"]),
    ("pb_data_04", "data", "Clean the missing values from the survey dataset.", ["data_processor"]),
    ("pb_data_05", "data", "Compute summary statistics for the inventory prices.", ["statistics", "data_processor"]),
    ("pb_data_06", "data", "Read the CSV file and list all of its columns.", ["data_processor", "file"]),
    ("pb_data_07", "data", "Find the median of the monthly units sold.", ["statistics"]),
    ("pb_data_08", "data", "Create a bar chart of quarterly revenue.", ["data_visualization", "statistics"]),
    ("pb_data_09", "data", "Group the sales data by region and aggregate the totals.", ["data_processor", "statistics"]),
    ("pb_data_10", "data", "Detect outliers in the experiment measurements.", ["statistics", "data_processor"]),
    # ---- document processing (mixed: format_converter/pdf_tool in; doc_generator/ppt out) ----
    ("pb_doc_01", "document", "Convert the markdown report into a PDF.", ["format_converter", "pdf_tool"]),
    ("pb_doc_02", "document", "Merge these three PDF files into one document.", ["pdf_tool"]),
    ("pb_doc_03", "document", "Split a PDF into separate single-page files.", ["pdf_tool"]),
    ("pb_doc_04", "document", "Convert a Word document into plain text.", ["format_converter"]),
    ("pb_doc_05", "document", "Generate a Word report from these notes.", ["doc_generator", "format_converter"]),
    ("pb_doc_06", "document", "Create a PowerPoint deck from the summary.", ["ppt_generator", "doc_generator"]),
    ("pb_doc_07", "document", "Encrypt a confidential PDF document with a password.", ["pdf_tool"]),
    ("pb_doc_08", "document", "Extract the page count and metadata from a PDF.", ["pdf_tool"]),
    ("pb_doc_09", "document", "Convert a CSV table into an Excel spreadsheet.", ["format_converter", "data_processor"]),
    ("pb_doc_10", "document", "Turn these slides into a printable handout document.", ["doc_generator", "ppt_generator"]),
    # ---- file & system (mixed: file/system_monitor in; shell/screen out) ----
    ("pb_file_01", "file", "Read the contents of notes.txt.", ["file"]),
    ("pb_file_02", "file", "Write this text into a new file called draft.txt.", ["file"]),
    ("pb_file_03", "file", "Summarize the content of the report file.", ["file"]),
    ("pb_file_04", "file", "List all files in the project directory.", ["shell", "file"]),
    ("pb_file_05", "file", "Take a screenshot of the current desktop.", ["screen"]),
    ("pb_file_06", "file", "Run a shell command to count the lines in a log.", ["shell"]),
    ("pb_file_07", "file", "Archive these files into a single zip.", ["shell", "file"]),
    ("pb_file_08", "file", "Check the available disk space and system load.", ["system_monitor", "shell"]),
    # ---- finance & macro (mixed: finance/calculator in; stock/fred/quant out) ----
    ("pb_fin_01", "finance", "Get the latest stock price for Apple.", ["stock_query", "search"]),
    ("pb_fin_02", "finance", "Analyze the trading signal for TSLA.", ["quant_trading", "stock_query"]),
    ("pb_fin_03", "finance", "Retrieve US GDP growth data from FRED.", ["fred_query", "search"]),
    ("pb_fin_04", "finance", "What is the current CPI inflation rate?", ["fred_query", "search"]),
    ("pb_fin_05", "finance", "Query the NVIDIA stock quote and daily change.", ["stock_query"]),
    ("pb_fin_06", "finance", "Backtest a moving-average trading strategy.", ["quant_trading"]),
    ("pb_fin_07", "finance", "Get the federal funds rate history.", ["fred_query"]),
    ("pb_fin_08", "finance", "Record a personal expense of 42 dollars for lunch.", ["finance"]),
    ("pb_fin_09", "finance", "Summarize my monthly income and spending.", ["finance"]),
    ("pb_fin_10", "finance", "Convert 250 USD to EUR at a rate of 0.92.", ["calculator", "search"]),
    # ---- browser & web (out-of-sandbox) ----
    ("pb_web_01", "web", "Open example.com and extract the headline text.", ["browser", "browser_use"]),
    ("pb_web_02", "web", "Automate filling out an online registration form.", ["browser_use", "browser"]),
    ("pb_web_03", "web", "Crawl a list of product pages and collect prices.", ["crawlee_tool", "browser_use"]),
    ("pb_web_04", "web", "Log into a website and download the monthly report.", ["browser_use", "browser"]),
    ("pb_web_05", "web", "Search the web for the latest AI news.", ["search", "duckduckgo_search"]),
    ("pb_web_06", "web", "Take a screenshot of a rendered webpage.", ["browser", "screen"]),
    ("pb_web_07", "web", "Scrape the price table from an online store.", ["crawlee_tool", "browser_use"]),
    ("pb_web_08", "web", "Navigate to a URL and read the main article body.", ["browser"]),
    # ---- research & academic (mixed: knowledge_rag/python_runner in; literature out) ----
    ("pb_res_01", "research", "Search recent papers about LLM tool use.", ["literature_search", "search"]),
    ("pb_res_02", "research", "Find open-access PDFs for a list of DOIs.", ["literature_search", "batch_paper_analyzer"]),
    ("pb_res_03", "research", "Summarize a batch of paper abstracts.", ["batch_paper_analyzer", "literature_review"]),
    ("pb_res_04", "research", "Search my local knowledge base for RAG notes.", ["knowledge_rag", "file"]),
    ("pb_res_05", "research", "Generate a literature review outline for a topic.", ["literature_review", "ai_writer"]),
    ("pb_res_06", "research", "Run a small Python snippet to parse a results file.", ["python_runner", "file"]),
    ("pb_res_07", "research", "Index these documents into the knowledge base.", ["knowledge_rag"]),
    ("pb_res_08", "research", "Look up citation metadata for a paper title.", ["literature_search", "search"]),
    # ---- multimedia (mostly out-of-sandbox) ----
    ("pb_media_01", "multimedia", "Extract the text from a scanned receipt image.", ["ocr", "file"]),
    ("pb_media_02", "multimedia", "Transcribe an audio recording into text.", ["speech_to_text", "voice_input"]),
    ("pb_media_03", "multimedia", "Generate an image of a sunset over mountains.", ["image_generator"]),
    ("pb_media_04", "multimedia", "Search for a free stock photo of a laptop.", ["stock_photo", "search"]),
    ("pb_media_05", "multimedia", "Capture a photo from the webcam.", ["media_capture"]),
    ("pb_media_06", "multimedia", "Convert my speech into a text note.", ["voice_input", "speech_to_text"]),
    ("pb_media_07", "multimedia", "Scan a paper document with the camera.", ["document_scanner", "ocr"]),
    ("pb_media_08", "multimedia", "Create a concept diagram of the system architecture.", ["concept_diagrams", "mind_map"]),
    # ---- productivity (in-sandbox: todo/datetime_tool/diary) ----
    ("pb_prod_01", "productivity", "Add a to-do item to review the pull request.", ["todo"]),
    ("pb_prod_02", "productivity", "List all of my pending tasks.", ["todo", "daily_task"]),
    ("pb_prod_03", "productivity", "What is the current date and time?", ["datetime_tool"]),
    ("pb_prod_04", "productivity", "Set a reminder to submit the report at 8pm.", ["cron", "notify", "todo"]),
    ("pb_prod_05", "productivity", "Write a diary entry about finishing the experiment.", ["diary"]),
    ("pb_prod_06", "productivity", "Send an email with the weekly summary.", ["email"]),
    ("pb_prod_07", "productivity", "Show my schedule for today.", ["datetime_tool", "course_schedule"]),
    ("pb_prod_08", "productivity", "Record today's workout in my journal.", ["diary", "fitness_nutrition"]),
    # ---- math & compute (in-sandbox: calculator/python_runner) ----
    ("pb_math_01", "math", "Calculate 128 times 76.", ["calculator"]),
    ("pb_math_02", "math", "Evaluate the expression (12 + 8) * 3.", ["calculator", "python_runner"]),
    ("pb_math_03", "math", "Compute the square root of 2025.", ["calculator", "python_runner"]),
    ("pb_math_04", "math", "Run a Python expression to sum a range of numbers.", ["python_runner"]),
    ("pb_math_05", "math", "What is 15 percent of 2400?", ["calculator"]),
    ("pb_math_06", "math", "Convert 3.5 hours into minutes.", ["calculator", "datetime_tool"]),
    # ---- knowledge & QA (in-sandbox: knowledge_rag/poetry) ----
    ("pb_know_01", "knowledge", "Search the knowledge base for context compression.", ["knowledge_rag", "file"]),
    ("pb_know_02", "knowledge", "Find a classical poem about the moon.", ["poetry"]),
    ("pb_know_03", "knowledge", "Look up poems by Li Bai.", ["poetry", "search"]),
    ("pb_know_04", "knowledge", "Query the vector index for tool-selection docs.", ["knowledge_rag"]),
    ("pb_know_05", "knowledge", "Recommend some classic Tang dynasty poems.", ["poetry"]),
    ("pb_know_06", "knowledge", "Search my notes for the RAG grounding explanation.", ["knowledge_rag", "file"]),
    # ---- health & lifestyle (mixed: diary/finance in; health/medication out) ----
    ("pb_health_01", "health", "Record my weight and sleep quality for today.", ["health", "diary"]),
    ("pb_health_02", "health", "Log a blood pressure reading.", ["health"]),
    ("pb_health_03", "health", "Set a medication reminder for 9am.", ["medication", "cron"]),
    ("pb_health_04", "health", "Plan a weekly workout routine.", ["fitness_nutrition"]),
    ("pb_health_05", "health", "Track my daily expenses this week.", ["finance"]),
    ("pb_health_06", "health", "Write a journal entry about my mood today.", ["diary"]),
    # ---- travel & maps (mostly out-of-sandbox) ----
    ("pb_travel_01", "travel", "Find the weather for my trip to Bangkok.", ["weather", "search"]),
    ("pb_travel_02", "travel", "Search for flights from Beijing to Shanghai.", ["search", "browser"]),
    ("pb_travel_03", "travel", "Look up restaurant reviews near the hotel.", ["search", "browser_use"]),
    ("pb_travel_04", "travel", "Book a hotel room through the website.", ["browser_use"]),
    ("pb_travel_05", "travel", "Get directions between two addresses.", ["search", "browser"]),
    ("pb_travel_06", "travel", "Check the exchange rate for my destination.", ["calculator", "search"]),
]


# Chinese rendering of the same 100 public tasks (id -> zh query). The E11
# primary run uses these so the whole 200q set is language-consistent with the
# production intent detector (``detect_intent_with_confidence``), whose
# INTENT_CATEGORIES are Chinese keywords; the English originals in ``_PUBLIC``
# are retained for the cross-lingual boundary run. Only the surface text varies
# -- id/domain/acceptable_tools (the ground truth) are held fixed.
_PUBLIC_ZH: dict[str, str] = {
    # ---- weather ----
    "pb_weather_01": "今天东京的天气怎么样？",
    "pb_weather_02": "查一下伦敦周末的天气预报。",
    "pb_weather_03": "新加坡明天会下雨吗？",
    "pb_weather_04": "获取巴黎当前的气温和湿度。",
    "pb_weather_05": "比较一下悉尼和首尔的天气。",
    "pb_weather_06": "现在纽约的风速是多少？",
    # ---- data & analytics ----
    "pb_data_01": "读取 sales.csv 并计算总营收。",
    "pb_data_02": "计算这一列数据的平均值和标准差。",
    "pb_data_03": "加载成绩表格并汇总分数。",
    "pb_data_04": "清理调查数据集中的缺失值。",
    "pb_data_05": "计算库存价格的描述性统计量。",
    "pb_data_06": "读取这个 CSV 文件并列出所有列。",
    "pb_data_07": "求每月销量的中位数。",
    "pb_data_08": "根据季度营收生成一张柱状图。",
    "pb_data_09": "按地区对销售数据分组并汇总合计。",
    "pb_data_10": "检测实验测量数据中的异常值。",
    # ---- document processing ----
    "pb_doc_01": "把这份 markdown 报告转换成 PDF。",
    "pb_doc_02": "把这三个 PDF 文件合并成一个文档。",
    "pb_doc_03": "把一个 PDF 拆分成单页文件。",
    "pb_doc_04": "把 Word 文档转换成纯文本。",
    "pb_doc_05": "根据这些笔记生成一份 Word 报告。",
    "pb_doc_06": "根据这份摘要制作一个 PowerPoint 演示文稿。",
    "pb_doc_07": "给一份机密 PDF 文档加密码。",
    "pb_doc_08": "提取一个 PDF 的页数和元数据。",
    "pb_doc_09": "把 CSV 表格转换成 Excel 电子表格。",
    "pb_doc_10": "把这些幻灯片转成可打印的讲义文档。",
    # ---- file & system ----
    "pb_file_01": "读取 notes.txt 的内容。",
    "pb_file_02": "把这段文字写入一个名为 draft.txt 的新文件。",
    "pb_file_03": "总结这份报告文件的内容。",
    "pb_file_04": "列出项目目录下的所有文件。",
    "pb_file_05": "给当前桌面截一张图。",
    "pb_file_06": "运行一条 shell 命令统计日志的行数。",
    "pb_file_07": "把这些文件打包成一个 zip 压缩包。",
    "pb_file_08": "检查可用磁盘空间和系统负载。",
    # ---- finance & macro ----
    "pb_fin_01": "获取苹果公司的最新股价。",
    "pb_fin_02": "分析特斯拉(TSLA)的交易信号。",
    "pb_fin_03": "从 FRED 获取美国 GDP 增长数据。",
    "pb_fin_04": "当前的 CPI 通胀率是多少？",
    "pb_fin_05": "查询英伟达的股票报价和当日涨跌。",
    "pb_fin_06": "回测一个均线交易策略。",
    "pb_fin_07": "获取联邦基金利率的历史数据。",
    "pb_fin_08": "记录一笔 42 美元的午餐个人开支。",
    "pb_fin_09": "汇总我每月的收入和支出。",
    "pb_fin_10": "按 0.92 的汇率把 250 美元换算成欧元。",
    # ---- browser & web ----
    "pb_web_01": "打开 example.com 并提取标题文本。",
    "pb_web_02": "自动填写一份在线注册表单。",
    "pb_web_03": "爬取一批商品页面并收集价格。",
    "pb_web_04": "登录一个网站并下载月度报告。",
    "pb_web_05": "在网上搜索最新的 AI 新闻。",
    "pb_web_06": "给渲染后的网页截一张图。",
    "pb_web_07": "从一家网店抓取价格表。",
    "pb_web_08": "打开一个网址并读取正文主体。",
    # ---- research & academic ----
    "pb_res_01": "搜索关于大模型工具调用的最新论文。",
    "pb_res_02": "为一批 DOI 查找开放获取的 PDF。",
    "pb_res_03": "总结一批论文摘要。",
    "pb_res_04": "在我的本地知识库里搜索 RAG 笔记。",
    "pb_res_05": "为一个主题生成文献综述提纲。",
    "pb_res_06": "运行一小段 Python 代码解析结果文件。",
    "pb_res_07": "把这些文档索引到知识库中。",
    "pb_res_08": "根据论文标题查询引用元数据。",
    # ---- multimedia ----
    "pb_media_01": "从一张扫描的收据图片中提取文字。",
    "pb_media_02": "把一段音频录音转写成文字。",
    "pb_media_03": "生成一张山间日落的图片。",
    "pb_media_04": "搜索一张笔记本电脑的免费图库照片。",
    "pb_media_05": "用摄像头拍一张照片。",
    "pb_media_06": "把我说的话转成文字笔记。",
    "pb_media_07": "用相机扫描一份纸质文档。",
    "pb_media_08": "画一张系统架构的概念图。",
    # ---- productivity ----
    "pb_prod_01": "添加一条待办事项：审查这个 pull request。",
    "pb_prod_02": "列出我所有待处理的任务。",
    "pb_prod_03": "现在的日期和时间是多少？",
    "pb_prod_04": "设置一个晚上 8 点提交报告的提醒。",
    "pb_prod_05": "写一篇关于完成实验的日记。",
    "pb_prod_06": "发送一封带周报总结的邮件。",
    "pb_prod_07": "显示我今天的日程安排。",
    "pb_prod_08": "在我的日志里记录今天的锻炼。",
    # ---- math & compute ----
    "pb_math_01": "计算 128 乘以 76。",
    "pb_math_02": "计算表达式 (12 + 8) * 3 的值。",
    "pb_math_03": "计算 2025 的平方根。",
    "pb_math_04": "运行一个 Python 表达式对一段数字求和。",
    "pb_math_05": "2400 的 15% 是多少？",
    "pb_math_06": "把 3.5 小时换算成分钟。",
    # ---- knowledge & QA ----
    "pb_know_01": "在知识库里搜索上下文压缩的相关内容。",
    "pb_know_02": "找一首关于月亮的古诗。",
    "pb_know_03": "查询李白的诗。",
    "pb_know_04": "在向量索引里查询工具选择相关的文档。",
    "pb_know_05": "推荐一些经典的唐诗。",
    "pb_know_06": "在我的笔记里搜索 RAG 依据(grounding)的解释。",
    # ---- health & lifestyle ----
    "pb_health_01": "记录我今天的体重和睡眠质量。",
    "pb_health_02": "记录一次血压读数。",
    "pb_health_03": "设置一个早上 9 点的服药提醒。",
    "pb_health_04": "规划一周的锻炼计划。",
    "pb_health_05": "记录我这一周的每日开支。",
    "pb_health_06": "写一篇关于我今天心情的日记。",
    # ---- travel & maps ----
    "pb_travel_01": "查一下我去曼谷旅行的天气。",
    "pb_travel_02": "搜索从北京到上海的航班。",
    "pb_travel_03": "查询酒店附近的餐厅评价。",
    "pb_travel_04": "通过网站预订一间酒店客房。",
    "pb_travel_05": "获取两个地址之间的路线指引。",
    "pb_travel_06": "查一下我目的地的汇率。",
}


def build_public_entries(language: str = "zh") -> list[dict]:
    """Return the 100 public entries as dicts (id/domain/query/acceptable_tools/language).

    ``language='zh'`` (default) selects the Chinese rendering used by the E11
    primary run; ``language='en'`` selects the original English text used by the
    cross-lingual boundary run. ``acceptable_tools`` and ``domain`` are identical
    across both, so the ground truth is unchanged -- only the query text varies.
    """
    if language not in ("zh", "en"):
        raise ValueError(f"unsupported language {language!r} (expected 'zh' or 'en')")
    return [
        {"id": i, "domain": d,
         "query": (_PUBLIC_ZH[i] if language == "zh" else q),
         "acceptable_tools": list(a), "language": language}
        for (i, d, q, a) in _PUBLIC
    ]


# E11 primary set: language-consistent with the production zh intent detector.
PUBLIC_TOOLBENCH_LITE_E11: list[dict] = build_public_entries("zh")
# Cross-lingual boundary set: original English rendering, retained for the
# documented zh-vs-en intent-detection boundary finding.
PUBLIC_TOOLBENCH_LITE_E11_EN: list[dict] = build_public_entries("en")

if __name__ == "__main__":
    for lang in ("zh", "en"):
        entries = build_public_entries(lang)
        ids = [e["id"] for e in entries]
        qs = [e["query"] for e in entries]
        print(f"[{lang}] count:", len(entries),
              "unique ids:", len(set(ids)) == len(ids),
              "unique queries:", len(set(qs)) == len(qs),
              "domains:", len({e["domain"] for e in entries}))
