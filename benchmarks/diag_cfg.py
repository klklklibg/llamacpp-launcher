"""隔离诊断：为什么实际 server 跑不到 70 t/s。

逐项关闭可疑因素，量化每一项的代价：
  A 纯 baseline + 126K ctx
  B MTP n2 + 126K ctx
  C MTP n2 + 8K ctx
  D MTP n2 + 8K ctx + 视觉
"""
import json
import os
import re
import subprocess
import time
import urllib.request

BIN = "<FILL_PATH>"
MODEL = "<FILL_PATH>"
PROMPT = "用大约 200 字说明为什么 GPU 的显存带宽比算力更常成为推理瓶颈。"
PORT = 8096


def build(name, ctx, mtp, vision):
    a = [BIN, "-m", MODEL, "-ngl", "99", "-c", str(ctx), "-b", "2048", "-ub", "1024",
         "-t", "6", "-fa", "on", "-lm", "none", "--jinja", "--metrics", "-np", "1",
         "--host", "127.0.0.1", "--port", str(PORT)]
    if mtp:
        a += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    if vision:
        a += ["--mmproj-auto"]
    return a


def probe():
    body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 128, "temperature": 0.7}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=300).read())
    wall = time.time() - t
    tm = d.get("timings", {})
    return {"wall_tps": round(d["usage"]["completion_tokens"] / wall, 2),
            "server_tps": round(tm.get("predicted_per_second", 0), 2),
            "ms_per_tok": round(tm.get("predicted_per_token_ms", 0), 2),
            "draft_n": tm.get("draft_n", 0),
            "draft_acc": tm.get("draft_n_accepted", 0),
            "prompt_tps": round(tm.get("prompt_per_second", 0), 1)}


def run(tag, ctx, mtp, vision):
    logp = os.path.join("<FILL_PATH>", f"srv_{tag}.log")
    with open(logp, "w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.Popen(build(tag, ctx, mtp, vision), stdout=lf, stderr=subprocess.STDOUT)
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
                time.sleep(0.5)
        load = time.time() - t0
        res = {"tag": tag, "ctx": ctx, "mtp": mtp, "vision": vision, "load_s": round(load, 1)}
        if not ready:
            res["error"] = "server failed to start"
        else:
            try:
                probe()
                time.sleep(0.3)
                res.update(probe())
            except Exception as e:
                res["error"] = repr(e)[:200]
        p.terminate()
        try:
            p.wait(timeout=25)
        except subprocess.TimeoutExpired:
            p.kill()
        time.sleep(2.0)
    txt = open(logp, encoding="utf-8", errors="replace").read()
    kv = re.findall(r"KV self size\s*=\s*([\d.]+)\s*MiB", txt)
    buf = re.findall(r"(?:CUDA0 )?model buffer size\s*=\s*([\d.]+)\s*MiB", txt)
    off = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", txt)
    res["kv_mib"] = float(kv[0]) if kv else None
    res["buf_mib"] = float(buf[-1]) if buf else None
    res["offload"] = off[0] if off else None
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


if __name__ == "__main__":
    out = []
    out.append(run("A_baseline_126k", 126000, False, False))
    out.append(run("B_mtp_126k", 126000, True, False))
    out.append(run("C_mtp_8k", 8192, True, False))
    out.append(run("D_mtp_8k_vision", 8192, True, True))
    with open("<FILL_PATH>", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("DONE")
