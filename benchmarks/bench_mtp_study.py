"""MTP (Multi-Token Prediction) 在 P100 上的机理实验。

三组：
  A. n_max 扫描（p_min=0，无自适应截断）-> 逐位置接受率衰减曲线
  B. p_min 扫描（n_max 固定较大）-> 自适应提前停止能否救回大 n_max
  C. 无投机 baseline

同一模型、同一 prompt、同一输出长度，串行执行。
"""
import json
import os
import re
import subprocess
import sys
import time

BIN = "<FILL_PATH>"
MODEL = "<FILL_PATH>"
PROMPT = ("用大约 200 字说明为什么 GPU 的显存带宽比算力更常成为推理瓶颈，"
          "并举例说明在什么样的负载下这个结论会反转。")
N_PREDICT = 256
LOGDIR = "<FILL_PATH>"
os.makedirs(LOGDIR, exist_ok=True)

# 注意：必须排除 "prompt eval time"，否则会匹配到 prefill 那一行
RE_EVAL = re.compile(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)")
RE_PROMPT = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
RE_STATS = re.compile(    r"statistics\s+(\S+?):\s*#calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+),\s*"
    r"#gen drafts =\s*(\d+),\s*#acc drafts =\s*(\d+),\s*"
    r"#gen tokens =\s*(\d+),\s*#acc tokens =\s*(\d+)"
    r"(?:,\s*#mean acc len =\s*([\d.]+))?"
    r"(?:,\s*#acc rate/pos =\s*\(([^)]*)\))?"
    r"(?:,\s*dur\(b,g,a\) =\s*([\d.]+),\s*([\d.]+),\s*([\d.]+) ms)?"
)


def run(tag, n_max=None, p_min=None, verbose=True):
    args = [BIN, "-m", MODEL, "-p", PROMPT, "-n", str(N_PREDICT), "-st",
            "--ignore-eos", "--no-mmproj", "-c", "2048", "-np", "1", "-fa", "on",
            "-lm", "none", "-ngl", "99", "--temp", "0.7", "-lv", "4"]
    if n_max is not None:
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max)]
        if p_min is not None:
            args += ["--spec-draft-p-min", str(p_min)]

    t0 = time.time()
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=600)
    wall = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    with open(os.path.join(LOGDIR, f"{tag}.log"), "w", encoding="utf-8", errors="replace") as f:
        f.write(out)

    res = {"tag": tag, "n_max": n_max, "p_min": p_min, "wall_s": round(wall, 1)}
    m = RE_EVAL.search(out)
    if m:
        res.update({"tg_ms_per_tok": float(m.group(3)), "tg_tps": float(m.group(4)),
                    "n_tokens": int(m.group(2))})
    m = RE_PROMPT.search(out)
    if m:
        res["pp_tps"] = round(int(m.group(2)) / (float(m.group(1)) / 1000.0), 1)
    m = RE_STATS.search(out)
    if m:
        res.update({
            "spec_type": m.group(1),
            "n_calls": int(m.group(4)),
            "gen_drafts": int(m.group(5)),
            "acc_drafts": int(m.group(6)),
            "gen_tokens": int(m.group(7)),
            "acc_tokens": int(m.group(8)),
            "mean_acc_len": float(m.group(9)) if m.group(9) else None,
            "acc_rate_pos": [float(x) for x in m.group(10).split(",")] if m.group(10) else [],
            "t_draft_ms": float(m.group(12)) if m.group(12) else None,
        })
    if "tg_tps" not in res:
        res["error"] = out[-600:]
    return res


def main():
    configs = [("baseline", None, None)]
    for nm in (1, 2, 3, 4, 6):
        configs.append((f"mtp_n{nm}", nm, 0.0))
    for pm in (0.2, 0.4, 0.6):
        configs.append((f"mtp_n6_p{pm}", 6, pm))
    for pm in (0.3, 0.5):
        configs.append((f"mtp_n8_p{pm}", 8, pm))

    results = []
    base_tps = None
    for tag, nm, pm in configs:
        r = run(tag, nm, pm)
        if tag == "baseline" and "tg_tps" in r:
            base_tps = r["tg_tps"]
        if base_tps and "tg_tps" in r:
            r["speedup"] = round(r["tg_tps"] / base_tps, 3)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)

    with open("<FILL_PATH>", "w", encoding="utf-8") as f:
        json.dump({"baseline_tps": base_tps, "results": results}, f, ensure_ascii=False, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
