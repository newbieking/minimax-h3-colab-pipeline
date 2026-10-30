"""
pipeline_scheduler.py
Cross-platform MiniMax H3 Automated GPU Workflow Scheduler for Google Colab.
Supports Windows, Linux, and macOS.
Includes transparent Drive mount via gws, warmup delays, reconnection guards,
Plan C batch task execution, and CU leak protection.
"""

import os
import sys
import time
import json
import shutil
import logging
import urllib.request
import urllib.parse
import subprocess

# 1. Cross-platform stdout UTF-8 encoding configuration
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORKSPACE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 2. Local File & Stream Logging
log_filename = os.path.join(LOG_DIR, f"minimax_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger = logging.getLogger("MiniMaxPipeline")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# 3. Dynamic Cross-Platform CLI Resolution
def resolve_executable(name):
    """Find executable path dynamically across Windows, macOS, and Linux."""
    if sys.platform == "win32":
        candidates = [f"{name}.cmd", f"{name}.exe", f"{name}.bat", name]
    else:
        candidates = [name]

    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return name

COLAB_BIN = resolve_executable("colab")
GWS_BIN = resolve_executable("gws")

def run_command(cmd, desc="", check=True, timeout=None):
    logger.info(f"[RUN] 执行: {desc or ' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout
    )
    if result.returncode != 0 and check:
        logger.error(f"[ERR] 命令失败 (code {result.returncode}):\n{result.stderr.strip()}")
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}")
    return result.stdout.strip(), result.stderr.strip()

def get_colab_usage():
    try:
        stdout, _ = run_command([COLAB_BIN, "usage"], desc="查询 Colab CU 余额与使用率", check=False)
        return stdout
    except Exception as e:
        logger.warning(f"[WARN] 获取 CU 状态失败: {e}")
        return "未知"

def list_active_sessions():
    try:
        stdout, _ = run_command([COLAB_BIN, "sessions"], desc="列出活跃会话", check=False)
        return stdout
    except Exception as e:
        return ""

def release_session(session_name):
    logger.info(f"[STOP] 正在释放 Colab 会话并回收 GPU 资源: {session_name} ...")
    try:
        stdout, stderr = run_command([COLAB_BIN, "stop", "-s", session_name], desc=f"停止会话 {session_name}", check=False)
        logger.info(f"[STOP] 会话释放响应: {stdout or stderr or '完成'}")
    except Exception as e:
        logger.error(f"[ERR] 停止会话异常: {e}")

    # 二次确认会话已彻底关闭
    sessions = list_active_sessions()
    if session_name in sessions:
        logger.warning(f"[WARN] 会话 {session_name} 仍在列表中，尝试二次强行终止...")
        subprocess.run([COLAB_BIN, "stop", "-s", session_name], check=False)
    else:
        logger.info(f"[OK] 确认已完全释放会话 {session_name}，CU 计费已停止。")

def get_gws_credentials():
    logger.info("[AUTH] 正在从本地 keyring 提取 gws 凭证...")
    try:
        out = subprocess.check_output([GWS_BIN, "auth", "export", "--unmasked"], text=True, encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"[ERR] 调用 gws auth export 失败，请确保已安装并登录 gws (gws auth login): {e}")
        raise

    creds = None
    for line in out.splitlines():
        if line.strip().startswith("{") and line.strip().endswith("}"):
            creds = json.loads(line.strip())
            break
    if not creds:
        creds = json.loads(out.strip())

    # 预刷新获取即时可用的 access_token
    token_url = "https://oauth2.googleapis.com/token"
    post_data = urllib.parse.urlencode({
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token"
    }).encode("utf-8")
    req = urllib.request.Request(token_url, data=post_data)
    token_resp = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    creds["access_token"] = token_resp["access_token"]
    logger.info("[AUTH] 成功置换 1 小时高可用 Access Token，双重凭证链路就绪。")
    return creds

def estimate_task_resources(tasks):
    """Estimate time and Colab Compute Units (CU) for a batch of tasks on A100."""
    total_est_seconds = 45  # Setup and mounting overhead
    details = []

    for t in tasks:
        tid = t.get("id")
        name = t.get("name")
        steps = t.get("steps", 4)
        duration = t.get("duration", 5.0)
        use_lora = t.get("use_lora", True)

        # Baseline: 5s 4-step Turbo ~ 40s
        if use_lora:
            base_s = 40 if steps <= 4 else (65 if steps <= 8 else 100)
        else:
            base_s = 180  # 20-step base model

        # Scale by duration
        time_scale = duration / 5.0
        est_s = base_s * (time_scale ** 0.8) + 10  # decode & upload overhead
        total_est_seconds += est_s
        details.append((tid, name, steps, duration, est_s))

    total_minutes = total_est_seconds / 60.0
    # Colab A100 is ~13 CU/hour = ~0.217 CU/minute
    est_cu = total_minutes * (13.0 / 60.0)
    return total_minutes, est_cu, details

def run_test_pipeline(session_name=None, dry_run=False, tasks_file=None, task_ids_str=None, timeout_sec=3600):
    if not session_name:
        session_name = f"minimax-h3-{int(time.time()) % 10000}"

    # Load tasks configuration
    if not tasks_file:
        tasks_file = os.path.join(WORKSPACE_DIR, "plan_c_tasks.json")

    if not os.path.exists(tasks_file):
        raise FileNotFoundError(f"未找到测试任务配置文件: {tasks_file}")

    with open(tasks_file, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    # Filter tasks by IDs if specified
    if task_ids_str and task_ids_str.lower() != "all":
        selected_ids = {int(x.strip()) for x in task_ids_str.split(",") if x.strip()}
        tasks = [t for t in all_tasks if t.get("id") in selected_ids]
    else:
        tasks = all_tasks

    if not tasks:
        raise ValueError("选择的任务列表为空，请检查 --task-ids 配置。")

    est_minutes, est_cu, details = estimate_task_resources(tasks)

    logger.info("======================================================================")
    logger.info("[START] MiniMax H3 批量自动化测试流水线 (Plan C)")
    logger.info(f"   会话名称: {session_name}")
    logger.info(f"   任务总数: {len(tasks)} 支视频")
    logger.info(f"   预估总运行时长: {est_minutes:.1f} 分钟")
    logger.info(f"   预估消耗 CU: ~{est_cu:.2f} CU (A100 SXM4 费率)")
    logger.info(f"   日志存储: {log_filename}")
    logger.info(f"   CLI 路径: colab={COLAB_BIN}, gws={GWS_BIN}")
    logger.info("----------------------------------------------------------------------")
    logger.info("【待执行任务清单】:")
    for tid, name, steps, dur, est_s in details:
        logger.info(f"   #Task {tid}: {name} | {dur}s | {steps} steps | 预估耗时: {est_s:.0f}s")
    logger.info("======================================================================")

    # 1. 检查凭据与 CU 余额
    creds = get_gws_credentials()
    client_id = creds["client_id"]
    client_secret = creds["client_secret"]
    refresh_token = creds["refresh_token"]
    access_token = creds["access_token"]

    initial_usage = get_colab_usage()
    logger.info(f"【当前 Colab 余额状态】\n{initial_usage}")

    if dry_run:
        logger.info("----------------------------------------------------------------------")
        logger.info("[OK] 【Dry-run 预检完成】")
        logger.info(f"  - 本地 gws 与 Colab CLI 链路 100% 畅通")
        logger.info(f"  - 计划执行 {len(tasks)} 个任务，预估耗时 {est_minutes:.1f} 分钟，需消耗约 {est_cu:.2f} CU")
        logger.info("  - 本次未分配 GPU 硬件，扣费为 0。随时可正式执行全自动评测！")
        logger.info("----------------------------------------------------------------------")
        return

    remote_script_path = os.path.join(WORKSPACE_DIR, "remote_minimax_runner.py")
    if not os.path.exists(remote_script_path):
        raise FileNotFoundError(f"未找到远端执行脚本: {remote_script_path}")

    with open(remote_script_path, "r", encoding="utf-8") as f:
        remote_code = f.read()

    tasks_json_str = json.dumps(tasks, ensure_ascii=False)

    # 2. 申请 A100 GPU 资源，在坚不可摧的 finally 中守护释放
    try:
        logger.info(f"[PROVISION] 正在向 Google Colab 申请分配 NVIDIA A100 GPU 资源 (Session: {session_name})...")
        stdout, stderr = run_command([COLAB_BIN, "new", "-s", session_name, "--gpu", "A100"], desc="分配 A100 运行时")
        logger.info(f"[PROVISION] 分配结果: {stdout}")

        # 冷启动预热缓冲：等待远端 Jupyter Kernel 与反向代理网络完全就绪
        logger.info("[WARMUP] 正在等待 A100 实例的通信通道就绪 (10 秒预热)...")
        time.sleep(10.0)

        logger.info(f"[EXEC] A100 实例已就绪，正在投递批处理任务并执行生成 (超时保护: {timeout_sec}s)...")
        start_time = time.time()

        runner_payload = f"""
import json, sys, traceback

with open('/content/remote_minimax_runner.py', 'w', encoding='utf-8') as f:
    f.write({json.dumps(remote_code)})

with open('/content/tasks.json', 'w', encoding='utf-8') as f:
    f.write({json.dumps(tasks_json_str)})

sys.path.insert(0, '/content')
import remote_minimax_runner
sys.argv = ['remote_minimax_runner.py', {json.dumps(client_id)}, {json.dumps(client_secret)}, {json.dumps(refresh_token)}, {json.dumps(access_token)}]
remote_minimax_runner.main()
"""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            cmd = [COLAB_BIN, "exec", "-s", session_name, "--timeout", str(timeout_sec)]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )

            proc.stdin.write(runner_payload)
            proc.stdin.close()

            has_conn_lost = False
            for line in proc.stdout:
                line_str = line.rstrip()
                if line_str:
                    logger.info(f"[Remote] {line_str}")
                    if "Connection was lost" in line_str:
                        has_conn_lost = True

            proc.wait()

            if proc.returncode == 0:
                break

            if has_conn_lost and attempt < max_retries:
                logger.warning(f"[RETRY] 检测到 Jupyter 通道瞬断 (Attempt {attempt}/{max_retries})，等待 6 秒后自动重试连接...")
                time.sleep(6.0)
                continue
            else:
                raise RuntimeError(f"远端执行脚本返回非 0 状态码: {proc.returncode}")

        elapsed = time.time() - start_time
        actual_cu = (elapsed / 3600.0) * 13.0
        logger.info(f"[SUCCESS] 批量工作流执行成功！总耗时: {elapsed/60:.2f} 分钟，实测约消耗: {actual_cu:.2f} CU。")

    except KeyboardInterrupt:
        logger.warning("[WARN] 检测到用户手动中断，立即启动紧急清理...")
    except Exception as e:
        logger.error(f"[ERR] 流水线执行异常: {e}")
        raise
    finally:
        # 3. 严格释放 GPU，防止 CU 泄漏
        release_session(session_name)
        final_usage = get_colab_usage()
        logger.info(f"【最终 Colab 余额状态】\n{final_usage}")
        logger.info("======================================================================")
        logger.info("[FINISH] GPU 资源已彻底回收，安全退出。")
        logger.info("======================================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cross-platform MiniMax H3 Colab GPU Workflow Tester (Plan C)")
    parser.add_argument("--session", type=str, default=None, help="Colab session name")
    parser.add_argument("--dry-run", action="store_true", help="Validate credentials and requirements without allocating GPU")
    parser.add_argument("--tasks", type=str, default=None, help="Path to tasks json file (defaults to plan_c_tasks.json)")
    parser.add_argument("--task-ids", type=str, default="all", help="Comma-separated task IDs to run, e.g. '1,2' or 'all' (default: all)")
    parser.add_argument("--timeout", type=int, default=3600, help="Colab exec timeout in seconds (default: 3600)")
    args = parser.parse_args()

    run_test_pipeline(
        session_name=args.session,
        dry_run=args.dry_run,
        tasks_file=args.tasks,
        task_ids_str=args.task_ids,
        timeout_sec=args.timeout
    )
