"""Run exp10b with system Python 3.14."""
import os, sys, subprocess

env = os.environ.copy()
env["PYTHONPATH"] = r"<PROJECT>"
env["WECLAW_BENCHMARK_RUN_ID"] = sys.argv[1] if len(sys.argv) > 1 else "20260608_n500_static"
# Remove problematic env vars
for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
    env.pop(k, None)

with open(r"<PROJECT>\.env", encoding="utf-8") as f:
    for line in f:
        if line.startswith("DEEPSEEK_API_KEY="):
            env["DEEPSEEK_API_KEY"] = line.split("=", 1)[1].strip()
            break

method = sys.argv[2] if len(sys.argv) > 2 else "static"
cwd = r"<PROJECT>\docs\8 计划发布的论文papers\weclaw_adaptive_runtime\benchmarks"
script = os.path.join(cwd, "exp10b_tool_selection_500_llm.py")

print(f"Method: {method}, RUN_ID: {env['WECLAW_BENCHMARK_RUN_ID']}", flush=True)
result = subprocess.run(
    [r"C:\Python314\python.exe", script, "--method", method],
    cwd=cwd, env=env,
)
print(f"Exit code: {result.returncode}", flush=True)
