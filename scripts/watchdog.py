#!/usr/bin/env python3
"""WeClaw Watchdog — 进程守护脚本。

职责：
1. 监控 WeClaw 进程，异常退出时自动重启（最多 max_restarts 次）
2. 配合 self_control.restart 工具，检测 .restart_flag 实现受控重启
3. 调试模式直通：检测 pydevd/debugpy 时跳过守护，直接运行

修正项：
- C4: 调试模式检测（pydevd + debugpy）
- N1: 成功运行后重置 restart_count
- N3: 封装到 main() 函数，避免模块级 return
- N7: Windows 文件锁安全删除（_safe_unlink）
- N8: 确保 logs 目录存在
- N9: 崩溃退出码也递增 restart_count
- N13: 增加 debugpy 检测
- P4: 调试模式入口使用 src.__main__.main()

用法：
    python scripts/watchdog.py          # 守护模式启动
    python scripts/watchdog.py --v2     # 传递 --v2 给 WeClaw
    python scripts/watchdog.py --cli    # 传递 --cli 给 WeClaw
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# 项目根目录（scripts/ 的上一级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 配置
MAX_RESTARTS = 3                   # 最大连续重启次数
UPTIME_THRESHOLD = 60              # 运行超过此秒视为"成功运行"，重置计数
FLAG_FILE = _PROJECT_ROOT / ".restart_flag"
LOG_FILE = _PROJECT_ROOT / "logs" / "watchdog.log"


def _is_debug_mode() -> bool:
    """C4 + N13 修正：检测是否处于调试模式。"""
    # Python -d 标志
    if getattr(sys, "flags", None) and sys.flags.debug:
        return True
    # pydevd（PyCharm / VS Code Python 调试器）
    if "pydevd" in sys.modules:
        return True
    # debugpy（VS Code 新版调试器）
    if "debugpy" in sys.modules:
        return True
    return False


def _safe_unlink(path: Path) -> None:
    """N7 修正：Windows 文件锁安全删除。

    Windows 下进程退出后文件锁可能尚未释放，
    需要重试机制避免 PermissionError。
    """
    if not path.exists():
        return
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        time.sleep(0.5)
        try:
            path.unlink(missing_ok=True)
        except (PermissionError, FileNotFoundError):
            pass


def _log(message: str) -> None:
    """写入 watchdog 日志（追加模式）。"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass  # 日志写入失败不应阻止 watchdog 运行
    print(f"[watchdog] {message}")


def _find_python() -> str:
    """查找 Python 解释器路径（优先使用 .venv）。"""
    venv_python = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    # Linux/macOS
    venv_python_unix = _PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python_unix.exists():
        return str(venv_python_unix)
    return sys.executable


def _get_weclaw_args() -> list[str]:
    """获取传递给 WeClaw 的命令行参数（透传 watchdog 自身参数）。"""
    return sys.argv[1:]


def main() -> None:
    """N3 修正：封装到 main() 函数内，避免模块级 return SyntaxError。"""

    # C4 + N13 修正：调试模式直通
    if _is_debug_mode():
        _log("检测到调试模式，跳过 Watchdog 直接运行 WeClaw")
        # P4 修正：使用 src.__main__.main() 作为入口
        from src.__main__ import main as src_main
        return src_main()

    # N8 修正：确保 logs 目录存在
    (_PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    _log(f"Watchdog 启动 (max_restarts={MAX_RESTARTS}, uptime_threshold={UPTIME_THRESHOLD}s)")

    python_path = _find_python()
    weclaw_args = _get_weclaw_args()
    restart_count = 0
    start_time = time.monotonic()

    while restart_count < MAX_RESTARTS:
        # 清理上一轮的 restart flag
        _safe_unlink(FLAG_FILE)

        # 构建启动命令
        cmd = [python_path, "-m", "src"] + weclaw_args
        _log(f"启动 WeClaw: {' '.join(cmd)}")

        # 启动子进程（捕获 stderr 以获取 Qt/原生层崩溃信息）
        proc_start = time.monotonic()
        stderr_log = _PROJECT_ROOT / "logs" / "watchdog_stderr.log"
        try:
            with open(stderr_log, "a", encoding="utf-8", errors="replace") as stderr_f:
                stderr_f.write(f"\n=== WeClaw session start: {datetime.now().isoformat()} ===\n")
                stderr_f.flush()
                proc = subprocess.Popen(cmd, cwd=str(_PROJECT_ROOT), stderr=stderr_f)
                exit_code = proc.wait()
        except KeyboardInterrupt:
            _log("收到 Ctrl+C，Watchdog 退出")
            break
        except Exception as e:
            _log(f"启动 WeClaw 失败: {e}")
            restart_count += 1
            continue

        proc_duration = time.monotonic() - proc_start

        # 记录退出信息
        _log(f"WeClaw 退出: exit_code={exit_code}, 运行时长={proc_duration:.1f}s")

        # 判断退出原因
        if FLAG_FILE.exists():
            # self_control.restart 写入的 flag → 受控重启（用户主动行为，非失败）
            _safe_unlink(FLAG_FILE)
            restart_count = 0  # 受控重启重置计数（不计入失败次数）
            _log(f"检测到 restart_flag，执行受控重启（计数已重置）")
            continue
        elif exit_code != 0:
            # 非零退出码 → 崩溃或异常退出
            # 但如果进程运行了足够久，说明之前是稳定的，重置计数
            if proc_duration >= UPTIME_THRESHOLD:
                restart_count = 0
                _log(f"运行 {proc_duration:.1f}s 后异常退出 (exit_code={exit_code})，视为稳定运行后崩溃，重置计数")
            restart_count += 1
            _log(f"异常退出 (exit_code={exit_code})，尝试重启 (第 {restart_count} 次)")
            # 崩溃后短暂等待，避免快速循环
            time.sleep(2)
            continue
        else:
            # exit_code == 0 且无 restart_flag → 正常退出
            # N1 修正：运行时间足够长则重置计数器
            if proc_duration >= UPTIME_THRESHOLD:
                restart_count = 0
                _log(f"正常运行 {proc_duration:.1f}s 后退出，重置重启计数")
            break

    # 超限检查
    if restart_count >= MAX_RESTARTS:
        _log(f"连续重启 {restart_count} 次均失败，Watchdog 放弃")
        print(f"\n[watchdog] 连续重启 {MAX_RESTARTS} 次均失败，请检查 logs/watchdog.log")
        sys.exit(1)

    total_uptime = time.monotonic() - start_time
    _log(f"Watchdog 退出 (总运行时长={total_uptime:.1f}s)")


if __name__ == "__main__":
    main()
