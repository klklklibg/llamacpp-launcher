"""两模型受控对比：同题、同参数、同后端、同判定规则。

题库自建，覆盖数学/物理/化学，每题唯一确定答案，便于自动判分。
关键指标：正确率、每题 reasoning token 消耗、每正确答案 token 成本。
"""
import json
import re
import sys
import time
import urllib.request

# 30 题：数学 10 / 物理 10 / 化学 10
QUESTIONS = [
    # --- 数学 ---
    ("math", "计算 17 × 23 的值。", "391"),
    ("math", "一个等差数列首项为 3，公差为 5，求第 20 项。", "98"),
    ("math", "求方程 x² - 7x + 12 = 0 的两个根之和。", "7"),
    ("math", "若 f(x) = 2x³ - 3x, 求 f'(1)。", "3"),
    ("math", "求 1 到 100 所有整数之和。", "5050"),
    ("math", "一个圆的半径是 7 cm，求其面积（保留 π）。", "49π"),
    ("math", "从 5 个人中选 3 人组成一组，有多少种选法？", "10"),
    ("math", "求 log₂(64) 的值。", "6"),
    ("math", "若 a + b = 10 且 ab = 21，求 a² + b²。", "58"),
    ("math", "求 ∫₀¹ 3x² dx 的值。", "1"),
    # --- 物理 ---
    ("physics", "质量 2 kg 的物体受 10 N 恒力，从静止开始 3 秒后速度为多少 m/s？", "15"),
    ("physics", "光在真空中的速度约为多少 m/s（取两位有效数字的科学计数法）？", "3.0×10^8"),
    ("physics", "一个 100 W 的灯泡连续工作 5 小时，消耗多少 kWh 电能？", "0.5"),
    ("physics", "电阻 4 Ω 两端加 12 V 电压，通过电流为多少 A？", "3"),
    ("physics", "自由落体 2 秒（g 取 10 m/s²），下落距离为多少 m？", "20"),
    ("physics", "波长 2 m、频率 50 Hz 的波，传播速度为多少 m/s？", "100"),
    ("physics", "把 1 kg 水从 20°C 加热到 80°C，需多少 kJ 热量（比热容 4.2 kJ/(kg·K)）？", "252"),
    ("physics", "动量 6 kg·m/s、质量 3 kg 的物体动能为多少 J？", "6"),
    ("physics", "平行板电容器电压加倍，电荷量变为原来的几倍？", "2"),
    ("physics", "理想气体等温压缩到原体积一半，压强变为原来的几倍？", "2"),
    # --- 化学 ---
    ("chemistry", "1 mol 水分子含有多少个水分子（用阿伏伽德罗常数表示）？", "6.02×10^23"),
    ("chemistry", "H₂SO₄ 中硫元素的化合价是多少？", "+6"),
    ("chemistry", "常温常压下，1 mol 理想气体的体积约为多少 L？", "22.4"),
    ("chemistry", "NaOH 的摩尔质量是多少 g/mol？", "40"),
    ("chemistry", "pH = 3 的溶液中氢离子浓度为多少 mol/L？", "10^-3"),
    ("chemistry", "甲烷 CH₄ 中碳原子采取什么杂化方式？", "sp3"),
    ("chemistry", "2H₂ + O₂ → 2H₂O 中，4 mol H₂ 完全反应生成多少 mol 水？", "4"),
    ("chemistry", "元素周期表中原子序数为 26 的元素符号是什么？", "Fe"),
    ("chemistry", "0.5 L 溶液含 1 mol NaCl，其物质的量浓度为多少 mol/L？", "2"),
    ("chemistry", "碳酸钙 CaCO₃ 的摩尔质量是多少 g/mol？", "100"),
]

SYS = "You are a precise science assistant. Solve the problem, then end your reply with a single final line in the exact format: ANSWER: <answer>"


def strip_think(s: str) -> str:
    s = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", s, flags=re.S | re.I)
    s = re.sub(r"^\s*</?think\b[^>]*>", "", s, flags=re.I)
    return s.strip()


def ask(port, q, max_tokens=3072):
    body = json.dumps({
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": q}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    wall = time.time() - t0
    u = d.get("usage", {})
    msg = d["choices"][0]["message"]
    raw = (msg.get("content") or "")
    # 部分后端把 reasoning 放在单独字段
    rt = msg.get("reasoning_content") or msg.get("reasoning") or ""
    text = strip_think(raw)
    m = re.findall(r"ANSWER\s*:\s*(.+)", text, flags=re.I)
    if m:
        pred = m[-1].strip().strip("。.")
    else:
        # 很多模型直接给答案、不加前缀：退回最后一行非空内容
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        pred = lines[-1].strip("。.") if lines else ""
    return {
        "pred": pred,
        "content": text,
        "reasoning_chars": len(rt) + max(0, len(raw) - len(text)),
        "completion_tokens": u.get("completion_tokens", 0),
        "prompt_tokens": u.get("prompt_tokens", 0),
        "wall": wall,
    }


def norm(s: str) -> str:
    s = s.lower().replace(" ", "").replace("，", ",").replace("−", "-")
    s = s.replace("×10^", "e").replace("*10^", "e").replace("x10^", "e")
    s = s.replace("^", "").replace("$", "")
    return s.strip("。.")


def judge(pred, gold) -> bool:
    p, g = norm(pred), norm(gold)
    if not p:
        return False
    if p == g:
        return True
    # 纯数值比较：允许相对误差 1%（应对 100 vs 100.09 这类近似答案）
    try:
        pf, gf = float(p.replace(",", "")), float(g.replace(",", ""))
        if abs(pf - gf) < 1e-6:
            return True
        if gf != 0 and abs(pf - gf) / abs(gf) < 0.01:
            return True
    except Exception:
        pass
    # 从预测里抽出第一个数值，与标准答案做同样比较
    m = re.search(r"-?\d+\.?\d*", p)
    mg = re.search(r"-?\d+\.?\d*", g)
    if m and mg:
        try:
            pf, gf = float(m.group()), float(mg.group())
            if abs(pf - gf) < 1e-6:
                return True
            if gf != 0 and abs(pf - gf) / abs(gf) < 0.01:
                return True
        except Exception:
            pass
    return g in p


def run(tag, port):
    rows = []
    t0 = time.time()
    for i, (cat, q, gold) in enumerate(QUESTIONS, 1):
        try:
            r = ask(port, q)
        except Exception as e:
            print(f"  [{i}] ERROR {type(e).__name__}: {e}", flush=True)
            rows.append({"cat": cat, "q": q, "gold": gold, "pred": "", "ok": False,
                         "tokens": 0, "wall": 0})
            continue
        ok = judge(r["pred"], gold)
        rows.append({"cat": cat, "q": q, "gold": gold, "pred": r["pred"], "ok": ok,
                     "tokens": r["completion_tokens"], "wall": r["wall"],
                     "reasoning_chars": r["reasoning_chars"]})
        print(f"  [{i:2}/{len(QUESTIONS)}] {'OK ' if ok else 'X  '} "
              f"{r['completion_tokens']:5} tok {r['wall']:6.1f}s  pred={r['pred'][:24]!r} "
              f"gold={gold!r}", flush=True)
    total_wall = time.time() - t0
    n = len(rows)
    correct = sum(1 for x in rows if x["ok"])
    toks = sum(x["tokens"] for x in rows)
    summary = {
        "tag": tag,
        "accuracy": round(correct / n * 100, 2),
        "correct": correct,
        "n": n,
        "total_tokens": toks,
        "tokens_per_question": round(toks / n, 1),
        "tokens_per_correct": round(toks / correct, 1) if correct else None,
        "correct_per_1m_tokens": round(correct / toks * 1e6, 1) if toks else None,
        "wall": round(total_wall, 1),
        "by_cat": {},
    }
    for cat in ("math", "physics", "chemistry"):
        sub = [x for x in rows if x["cat"] == cat]
        c = sum(1 for x in sub if x["ok"])
        summary["by_cat"][cat] = {
            "acc": round(c / len(sub) * 100, 1),
            "correct": c,
            "tokens": sum(x["tokens"] for x in sub),
        }
    return summary, rows


if __name__ == "__main__":
    tag, port = sys.argv[1], int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else f"eval_{tag}.json"
    print(f"=== {tag} (port {port}) ===", flush=True)
    summary, rows = run(tag, port)
    json.dump({"summary": summary, "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
