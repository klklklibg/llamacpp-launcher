"""测 ngram 投机（无需 draft 模型）是否对 decode 有帮助。"""
import json, time, urllib.request
import llmcore as core

TARGET = "<FILL_PATH>"
PROMPT = "用大约 120 字说明为什么 GPU 的显存带宽比算力更常成为推理瓶颈。"

def run(tag, spec_type=None, port=8093):
    m = core.Model(path=TARGET, name="bonsai27b", size=7_206_168_928, quant="PTQ1_0")
    cfg = core.load_config()
    cfg["backend_path"] = "<FILL_PATH>"
    p = {"port": port, "ctx": 4096, "ubatch": 512, "fa": "on", "parallel": 1}
    if spec_type:
        p.update({"extra": f"--spec-type {spec_type}"})
    else:
        p.update({"extra": ""})
    cfg["params"].update(p)
    sp = core.ServerProc(lambda l: None, lambda s, msg: None)
    t0 = time.time()
    sp.start(cfg, m)
    while time.time() - t0 < 200 and not sp.listen_url and sp.is_running():
        time.sleep(0.4)
    if not sp.listen_url:
        print(tag, "FAILED"); sp.stop(); return None
    body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 128, "temperature": 0.7}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t1 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    wall = time.time() - t1
    print(f"{tag}: wall={wall:.2f}s tokens={d['usage']['completion_tokens']} "
          f"tg={sp.perf.tg_tps:.2f} eff={d['usage']['completion_tokens']/wall:.2f}")
    sp.stop(); time.sleep(1.2)
    return wall

run("ngram-simple", "ngram-simple")
run("ngram-mod", "ngram-mod")
