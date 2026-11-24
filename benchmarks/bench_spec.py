"""对比投机解码在 P100 上的真实收益：同模型、同提示、同输出长度。"""
import json, time, urllib.request
import llmcore as core

TARGET = "<FILL_PATH>"
DRAFT = "<FILL_PATH>"
PROMPT = "用大约 120 字说明为什么 GPU 的显存带宽比算力更常成为推理瓶颈。"

def run(tag, spec=False, port=8091):
    m = core.Model(path=TARGET, name="bonsai27b", size=7_206_168_928, quant="PTQ1_0")
    cfg = core.load_config()
    cfg["backend_path"] = "<FILL_PATH>"
    cfg["params"].update({"port": port, "ctx": 4096, "ubatch": 512, "fa": "on",
                          "parallel": 1, "spec_enabled": spec,
                          "spec_model": DRAFT if spec else "", "spec_type": "draft-simple",
                          "spec_n_max": 3, "spec_ngl": 99})
    sp = core.ServerProc(lambda l: None, lambda s, msg: None)
    t0 = time.time()
    sp.start(cfg, m)
    while time.time() - t0 < 240 and not sp.listen_url and sp.is_running():
        time.sleep(0.4)
    if not sp.listen_url:
        print(tag, "FAILED to start"); sp.stop(); return None
    load = time.time() - t0
    body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 128, "temperature": 0.7}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t1 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    wall = time.time() - t1
    u = d["usage"]
    tg = sp.perf.tg_tps
    sp.stop(); time.sleep(1.5)
    print(f"{tag}: load={load:.1f}s wall={wall:.2f}s tokens={u['completion_tokens']} "
          f"tg={tg:.2f} t/s eff={u['completion_tokens']/wall:.2f} t/s")
    return {"tag": tag, "tg": tg, "eff": u["completion_tokens"] / wall,
            "tokens": u["completion_tokens"]}

res = []
res.append(run("baseline"))
res.append(run("spec_draft", spec=True))
print(json.dumps(res, ensure_ascii=False))
