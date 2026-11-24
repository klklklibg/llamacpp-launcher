"""DFlash vs MTP vs 无投机：同一 target、同一 prompt 的对照实验。

target : DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q8_0 (qwen35, 33 层, 其中 blk.32 是 nextn)
draft  : Qwen3.5-9B-DFlash.Q8_0 (dflash, 1.3B, 6 层)
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

BIN = "<FILL_PATH>"
TARGET = "<FILL_PATH>"
DFLASH = "<FILL_PATH>"

CTX = 8192
PORT = 8097
PROMPT = ("用大约 200 字说明为什么 GPU 的显存带宽比算力更常成为推理瓶颈，"
          "并举例说明在什么样的负载下这个结论会反转。")
NPRED = 192
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mtp_logs")
os.makedirs(LOGDIR, exist_ok=True)

RE_MEAN = re.compile(r"mean acc len\s*[:=]\s*([\d.]+)")
RE_ACC = re.compile(r"acc rate/pos\s*[:=]\s*([\d.,\s]+)")


def build(spec=None, n_max=2, extra=None, p_min=0.0, n_min=0):
    a = [os.path.join(BIN, "llama-server.exe"),
         "-m", TARGET,
         "-ngl", "99", "-c", str(CTX), "-b", "2048", "-ub", "1024",
         "-t", "6", "-fa", "on", "-lm", "none",
         "--jinja", "--metrics", "-np", "1",
         "--no-mmproj",
         "--host", "127.0.0.1", "--port", str(PORT)]
    if spec == "mtp":
        a += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max)]
    elif spec == "dflash":
        a += ["-md", DFLASH, "--spec-type", "draft-dflash",
              "--spec-draft-n-max", str(n_max)]
    if p_min:
        a += ["--spec-draft-p-min", str(p_min)]
    if n_min:
        a += ["--spec-draft-n-min", str(n_min)]
    if extra:
        a += extra
    return a


def run(label, args, rounds=2):
    logpath = os.path.join(LOGDIR, "df_%s.log" % label)
    log = open(logpath, "w", encoding="utf-8", errors="replace")
    print("\n=== %s ===" % label)
    print(" ".join(args[1:]))
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        ok = False
        while time.time() - t0 < 300:
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2).read()
                ok = True
                break
            except Exception:
                if proc.poll() is not None:
                    break
                time.sleep(0.4)
        load = time.time() - t0
        if not ok:
            with open(logpath, encoding="utf-8", errors="replace") as f:
                tailtxt = f.read()[-1200:]
            print("启动失败 (%.1fs)\n%s" % (load, tailtxt))
            return None
        print("ready %.1fs" % load)

        vals, drafts = [], []
        for i in range(rounds):
            body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                               "max_tokens": NPRED, "temperature": 0.7}).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:%d/v1/chat/completions" % PORT, data=body,
                headers={"Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=600).read())
            tm = d.get("timings", {})
            tps = tm.get("predicted_per_second", 0)
            dn, da = tm.get("draft_n", 0), tm.get("draft_n_accepted", 0)
            vals.append(tps)
            drafts.append((da, dn))
            print("  run%d %.2f t/s  tokens=%d  draft %d/%d"
                  % (i + 1, tps, tm.get("predicted_n", 0), da, dn))

        with open(logpath, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        mean = RE_MEAN.findall(txt)
        mean = float(mean[-1]) if mean else 0.0
        s = sorted(vals)
        med = s[len(s) // 2] if s else 0
        print("  --> median %.2f  mean_acc_len %.2f" % (med, mean))
        return {"label": label, "median": med, "vals": vals,
                "drafts": drafts, "mean_acc": mean}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        time.sleep(3)


def gpu_mem():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def main():
    if not os.path.isfile(DFLASH):
        print("缺少 DFlash draft:", DFLASH)
        return 1
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cases = [("baseline", build(None), rounds)]
    cases.append(("mtp_n2", build("mtp", 2), rounds))
    for n in (2, 3, 4, 5, 7, 11, 15):
        cases.append(("dflash_n%d" % n, build("dflash", n), rounds))

    out = []
    for label, args, r in cases:
        print("显存(启动前):", gpu_mem())
        res = run(label, args, r)
        if res:
            out.append(res)
    print("\n===== 汇总 =====")
    for r in out:
        print("%-12s %7.2f t/s   acc_len %.2f" % (r["label"], r["median"], r["mean_acc"]))
    with open(os.path.join(LOGDIR, "dflash_vs_mtp.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
