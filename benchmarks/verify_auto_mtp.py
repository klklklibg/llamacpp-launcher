"""最终状态验证：用启动器的真实参数起 server，测 MTP 吞吐。"""
import json
import os
import subprocess
import time
import urllib.request

import llmcore as core

cfg = core.load_config()
m = core.Model(path=cfg["last_model"], name="t", size=1, quant="Q8_0")
args = core.build_args(cfg, m)
print("ARGS:", " ".join(args))
print("NOTE:", core.SPEC_NOTE)
print()

log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "mtp_logs", "srv_auto.log"), "w", encoding="utf-8", errors="replace")
proc = subprocess.Popen([os.path.join(cfg["backend_path"], "llama-server.exe")] + args,
                        stdout=log, stderr=subprocess.STDOUT)

t0 = time.time()
ok = False
while time.time() - t0 < 240:
    try:
        urllib.request.urlopen("http://127.0.0.1:8095/health", timeout=2).read()
        ok = True
        break
    except Exception:
        if proc.poll() is not None:
            break
        time.sleep(0.4)
print("ready=%s load=%.1fs" % (ok, time.time() - t0))

if ok:
    P = "用大约 200 字说明为什么 GPU 的显存带宽比算力更常成为推理瓶颈。"
    vals = []
    for i in range(4):
        body = json.dumps({"messages": [{"role": "user", "content": P}],
                           "max_tokens": 128, "temperature": 0.7}).encode()
        req = urllib.request.Request("http://127.0.0.1:8095/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
        tm = d["timings"]
        dn, da = tm.get("draft_n", 0), tm.get("draft_n_accepted", 0)
        vals.append(tm["predicted_per_second"])
        print("run%d: %6.2f t/s   draft %d/%d (%.0f%%)"
              % (i + 1, tm["predicted_per_second"], da, dn, 100.0 * da / dn if dn else 0))
    s = sorted(vals)
    print("--- median %.2f   min %.2f   max %.2f ---" % (s[len(s) // 2], s[0], s[-1]))

print("显存:", end=" ")
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print(out)
except Exception as e:
    print(e)

proc.terminate()
try:
    proc.wait(timeout=20)
except subprocess.TimeoutExpired:
    proc.kill()
log.close()
