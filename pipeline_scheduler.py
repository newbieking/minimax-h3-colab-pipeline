"""
pipeline_scheduler.py
Cross-platform MiniMax H3 Automated GPU Workflow Scheduler for Google Colab.
Supports Windows, Linux, and macOS.
Includes transparent Drive mount via gws, warmup delays, reconnection guards, and CU leak protection.
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

def run_test_pipeline(session_name=None, dry_run=False):
    if not session_name:
        session_name = f"minimax-h3-{int(time.time()) % 10000}"

    logger.info("======================================================================")
    logger.info("[START] 开始执行 MiniMax H3 在线 GPU 工作流自动化流水线")
    logger.info(f"   会话名称: {session_name}")
    logger.info(f"   日志存储: {log_filename}")
    logger.info(f"   CLI 路径: colab={COLAB_BIN}, gws={GWS_BIN}")
    logger.info("======================================================================")

    # 1. 检查凭据与 CU 余额
    creds = get_gws_credentials()
    client_id = creds["client_id"]
    client_secret = creds["client_secret"]
    refresh_token = creds["refresh_token"]
    access_token = creds["access_token"]

    initial_usage = get_colab_usage()
    logger.info(f"【初始状态】\n{initial_usage}")

    if dry_run:
        logger.info("[OK] 【Dry-run 检查完成】本地 CLI、gws 凭证及 CU 配额均正常，随时可启动在线 GPU 任务。")
        return

    remote_script_path = os.path.join(WORKSPACE_DIR, "remote_minimax_runner.py")
    if not os.path.exists(remote_script_path):
        raise FileNotFoundError(f"未找到远端执行脚本: {remote_script_path}")

    with open(remote_script_path, "r", encoding="utf-8") as f:
        remote_code = f.read()

    # 2. 申请 A100 GPU 资源，在坚不可摧的 finally 中守护释放
    try:
        logger.info(f"[PROVISION] 正在向 Google Colab 申请分配 NVIDIA A100 GPU 资源 (Session: {session_name})...")
        stdout, stderr = run_command([COLAB_BIN, "new", "-s", session_name, "--gpu", "A100"], desc="分配 A100 运行时")
        logger.info(f"[PROVISION] 分配结果: {stdout}")

        # 冷启动预热缓冲：等待远端 Jupyter Kernel 与反向代理网络完全就绪
        logger.info("[WARMUP] 正在等待 A100 实例的通信通道就绪 (10 秒预热)...")
        time.sleep(10.0)

        logger.info("[EXEC] A100 实例已就绪，正在投递并执行远端自动化工作流脚本...")
        start_time = time.time()

        runner_payload = f"""
import json, sys, traceback
with open('/content/remote_minimax_runner.py', 'w', encoding='utf-8') as f:
    f.write({json.dumps(remote_code)})

sys.path.insert(0, '/content')
import remote_minimax_runner
sys.argv = ['remote_minimax_runner.py', {json.dumps(client_id)}, {json.dumps(client_secret)}, {json.dumps(refresh_token)}, {json.dumps(access_token)}]
remote_minimax_runner.main()
"""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            cmd = [COLAB_BIN, "exec", "-s", session_name, "--timeout", "1200"]
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
        logger.info(f"[SUCCESS] 远端工作流执行成功！总运行时长: {elapsed/60:.2f} 分钟。")

    except KeyboardInterrupt:
        logger.warning("[WARN] 检测到用户手动中断，立即启动紧急清理...")
    except Exception as e:
        logger.error(f"[ERR] 流水线执行异常: {e}")
        raise
    finally:
        # 3. 严格释放 GPU，防止 CU 泄漏
        release_session(session_name)
        final_usage = get_colab_usage()
        logger.info(f"【最终状态】\n{final_usage}")
        logger.info("======================================================================")
        logger.info("[FINISH] GPU 资源已彻底回收，安全退出。")
        logger.info("======================================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cross-platform MiniMax H3 Colab GPU Workflow Tester")
    parser.add_argument("--session", type=str, default=None, help="Colab session name")
    parser.add_argument("--dry-run", action="store_true", help="Validate credentials and requirements without allocating GPU")
    args = parser.parse_args()

    run_test_pipeline(session_name=args.session, dry_run=args.dry_run)
