"""测 ctx 大小对 decode 速度的影响，并同时采样满载 GPU 时钟。"""
import json
import os
import re
import subprocess
import threading
import time
import urllib.request

BIN = "<FILL_PATH>"
MODEL = "<FILL_PATH>"
PROMPT = "用大约 200 字说明为什么 GPU 的显存带宽比算力更常成为推理瓶颈。"
PORT = 8096
CTXS = [4096, 8192, 16384, 32768, 65536, 126000]

clock_stop = False
clocks = []


def sample_clock():
    while not clock_stop:
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu,temperature.gpu,power.draw",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
            if o:
                parts = [x.strip() for x in o.split(",")]
                clocks.append((int(parts[0]), int(parts[1])))
        except Exception:
            pass
        time.sleep(0.4)


def probe(n=2):
    best = None
    for _ in range(n):
        body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                           "max_tokens": 128, "temperature": 0.7}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
        tm = d.get("timings", {})
        r = {"tps": round(tm.get("predicted_per_second", 0), 2),
             "draft_n": tm.get("draft_n", 0), "acc": tm.get("draft_n_accepted", 0)}
        if best is None or r["tps"] > best["tps"]:
            best = r
    return best


def run(ctx):
    global clock_stop, clocks
    logp = os.path.join("<FILL_PATH>", f"srv_ctx{ctx}.log")
    with open(logp, "w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.Popen([BIN, "-m", MODEL, "-ngl", "99", "-c", str(ctx), "-b", "2048",
                              "-ub", "1024", "-t", "6", "-fa", "on", "-lm", "none", "--jinja",
                              "-np", "1", "--host", "127.0.0.1", "--port", str(PORT),
                              "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
                             stdout=lf, stderr=subprocess.STDOUT)
        t0 = time.time()
        ready = False
        while time.time() - t0 < 300:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
                ready = True
                break
            except Exception:
                if p.poll() is not None:
                    break
                time.sleep(0.4)
        res = {"ctx": ctx, "ready": ready, "load_s": round(time.time() - t0, 1)}
        if ready:
            clocks = []
            clock_stop = False
            th = threading.Thread(target=sample_clock, daemon=True)
            th.start()
            time.sleep(0.8)
            res.update(probe())
            clock_stop = True
            time.sleep(0.5)
            busy = [c for c, u in clocks if u > 50]
            res["sm_mhz_busy"] = round(sum(busy) / len(busy)) if busy else None
        p.terminate()
        try:
            p.wait(timeout=25)
        except subprocess.TimeoutExpired:
            p.kill()
        txt = open(logp, encoding="utf-8", errors="replace").read()
        m = re.search(r"KV self size\s*=\s*([\d.]+)\s*MiB", txt)
        res["kv_mib"] = float(m.group(1)) if m else None
        print(json.dumps(res, ensure_ascii=False), flush=True)
        time.sleep(2)
        return res


if __name__ == "__main__":
    out = [run(c) for c in CTXS]
    with open("<FILL_PATH>", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("DONE")
