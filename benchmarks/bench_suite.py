"""统一基准：投机技术 / KV 量化 / 内容类型 / 长 prompt 四个维度。

用法:
    python bench_suite.py kv          KV cache 量化
    python bench_suite.py content     内容类型
    python bench_suite.py family      投机技术横评
    python bench_suite.py prefill     长 prompt prefill
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
SMALL = "<FILL_PATH>"

PORT = 8097
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mtp_logs")
os.makedirs(LOGDIR, exist_ok=True)

RE_MEAN = re.compile(r"mean acc len\s*[:=]\s*([\d.]+)")
RE_ACCPOS = re.compile(r"acc rate/pos\s*[:=]\s*([\d.,\s]+)")

PROMPTS = {
    "math": ("一个水池有甲乙两个进水管，甲单独注满需要 6 小时，乙单独需要 4 小时，"
             "丙排水管单独排空需要 8 小时。三管齐开，多久注满？请分步推导。"),
    "code": ("写一个 Python 实现的 LRU 缓存类，支持 TTL 过期和线程安全，"
             "给出完整可运行代码。"),
    "essay": ("用大约 200 字说明为什么 GPU 的显存带宽比算力更常成为推理瓶颈，"
              "并举例说明在什么样的负载下这个结论会反转。"),
    "chat": "今天心情有点低落，随便聊聊吧。",
    # ngram 的典型甜点：保持原文结构，只做局部插入
    "rewrite": ("把下面这段代码逐行加上中文注释，代码本身一个字都不要改：\n\n"
                "def quicksort(arr):\n"
                "    if len(arr) <= 1:\n"
                "        return arr\n"
                "    pivot = arr[len(arr) // 2]\n"
                "    left = [x for x in arr if x < pivot]\n"
                "    middle = [x for x in arr if x == pivot]\n"
                "    right = [x for x in arr if x > pivot]\n"
                "    return quicksort(left) + middle + quicksort(right)\n\n"
                "def binary_search(arr, target):\n"
                "    low, high = 0, len(arr) - 1\n"
                "    while low <= high:\n"
                "        mid = (low + high) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        if arr[mid] < target:\n"
                "            low = mid + 1\n"
                "        else:\n"
                "            high = mid - 1\n"
                "    return -1\n"),
}


def build(spec=None, n_max=2, ctx=8192, ctk="f16", ctv="f16", p_min=0.0,
          extra=None, ub=1024):
    a = [os.path.join(BIN, "llama-server.exe"),
         "-m", TARGET,
         "-ngl", "99", "-c", str(ctx), "-b", "2048", "-ub", str(ub),
         "-t", "6", "-fa", "on", "-lm", "none",
         "--jinja", "--metrics", "-np", "1",
         "--no-mmproj",
         "--host", "127.0.0.1", "--port", str(PORT)]
    if ctk != "f16":
        a += ["-ctk", ctk]
    if ctv != "f16":
        a += ["-ctv", ctv]
    if spec == "mtp":
        a += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max)]
    elif spec == "dflash":
        a += ["-md", DFLASH, "--spec-type", "draft-dflash",
              "--spec-draft-n-max", str(n_max)]
    elif spec == "simple":
        a += ["-md", SMALL, "--spec-type", "draft-simple",
              "--spec-draft-n-max", str(n_max)]
    elif spec and spec.startswith("ngram"):
        a += ["--spec-type", spec]
    if p_min:
        a += ["--spec-draft-p-min", str(p_min)]
    if extra:
        a += extra
    return a


def post(path, body, timeout=900):
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (PORT, path),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def run(label, args, prompt="essay", rounds=2, n_predict=192, env=None):
    logpath = os.path.join(LOGDIR, "suite_%s.log" % label)
    log = open(logpath, "w", encoding="utf-8", errors="replace")
    print("\n=== %s ===" % label, flush=True)
    print(" ".join(args[1:]), flush=True)
    if env:
        print("ENV: %s" % env, flush=True)
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=e)
    try:
        t0 = time.time()
        ok = False
        while time.time() - t0 < 420:
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
                print("启动失败 (%.1fs)\n%s" % (load, f.read()[-900:]), flush=True)
            return None
        mem = gpu_mem()
        print("ready %.1fs  vram %s" % (load, mem), flush=True)

        vals, drafts, prefills = [], [], []
        prompt_n = 0
        for i in range(rounds):
            # 固定 seed：保证每轮生成内容一致，接受率与速度才可复现
            d = post("/v1/chat/completions",
                     {"messages": [{"role": "user",
                                    "content": PROMPTS.get(prompt, prompt)}],
                      "max_tokens": n_predict, "temperature": 0.7,
                      "seed": 1234,
                      "cache_prompt": False})
            tm = d.get("timings", {})
            prompt_n = tm.get("prompt_n", prompt_n)
            vals.append(tm.get("predicted_per_second", 0))
            prefills.append(tm.get("prompt_per_second", 0))
            drafts.append((tm.get("draft_n_accepted", 0), tm.get("draft_n", 0)))
            print("  run%d %.2f t/s  prefill %.1f t/s  out %d  draft %d/%d"
                  % (i + 1, vals[-1], prefills[-1], tm.get("predicted_n", 0),
                     drafts[-1][0], drafts[-1][1]), flush=True)

        with open(logpath, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        m = RE_MEAN.findall(txt)
        mean = float(m[-1]) if m else 0.0
        pos = RE_ACCPOS.findall(txt)
        s = sorted(vals)
        med = s[len(s) // 2] if s else 0
        dn = sum(x[1] for x in drafts)
        da = sum(x[0] for x in drafts)
        print("  --> median %.2f  acc_len %.2f  acc_rate %s"
              % (med, mean, ("%.1f%%" % (100.0 * da / dn)) if dn else "-"), flush=True)
        return {"label": label, "median": med, "vals": vals, "drafts": drafts,
                "mean_acc": mean, "acc_rate": (da / dn) if dn else None,
                "prefill": prefills, "vram": mem, "load": load,
                "acc_pos": pos[-1].strip() if pos else ""}
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
            capture_output=True, text=True).stdout.strip().split("\n")[0]
    except Exception:
        return "?"


def save(name, out):
    with open(os.path.join(LOGDIR, "%s.json" % name), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n===== 汇总 %s =====" % name)
    for r in out:
        print("%-22s %7.2f t/s  acc_len %.2f  acc %s"
              % (r["label"], r["median"], r["mean_acc"],
                 ("%.1f%%" % (100 * r["acc_rate"])) if r["acc_rate"] else "-"))


# ---------------- 各实验 ----------------

def exp_kv(rounds=2):
    cases = [
        ("kv_f16", build("mtp", 2)),
        ("kv_q8_0", build("mtp", 2, ctk="q8_0", ctv="q8_0")),
        ("kv_q8k_f16v", build("mtp", 2, ctk="q8_0")),
        ("kv_q4_0", build("mtp", 2, ctk="q4_0", ctv="q4_0")),
        ("kv_q8_0_nospec", build(None, ctk="q8_0", ctv="q8_0")),
    ]
    out = []
    for label, a in cases:
        r = run(label, a, "essay", rounds)
        if r:
            out.append(r)
    save("suite_kv", out)


def exp_content(rounds=2):
    out = []
    for p in ("math", "code", "essay", "chat", "rewrite"):
        r = run("base_%s" % p, build(None), p, rounds)
        if r:
            out.append(r)
        r = run("mtp_%s" % p, build("mtp", 2), p, rounds)
        if r:
            out.append(r)
    save("suite_content", out)


def exp_family(rounds=2):
    cases = [
        ("fam_baseline", build(None)),
        ("fam_mtp_n2", build("mtp", 2)),
        ("fam_dflash_n2", build("dflash", 2)),
        ("fam_ngram_simple", build("ngram-simple")),
        ("fam_ngram_mapk", build("ngram-map-k")),
        ("fam_ngram_mapk4v", build("ngram-map-k4v")),
        ("fam_ngram_mod", build("ngram-mod")),
    ]
    out = []
    for label, a in cases:
        for p in ("essay", "rewrite"):
            lab = "%s_%s" % (label, p)
            r = run(lab, a, p, rounds)
            if r:
                out.append(r)
    save("suite_family", out)


def exp_prefill(rounds=1):
    # 用重复填充构造长 prompt；rounds=1，避免 prompt cache 干扰 prefill 计时
    filler = ("这是一段用于填充上下文长度的说明文字，内容本身没有特殊含义，"
              "只是为了让输入序列变长，从而测量预填充阶段的速度。")
    out = []
    for name, mult in (("short", 0), ("mid", 100), ("long", 450)):
        prompt = PROMPTS["essay"] + ("\n" + filler * mult if mult else "")
        for tag, a in (("nospec", build(None, ctx=32768)),
                       ("mtp", build("mtp", 2, ctx=32768))):
            lab = "pf_%s_%s" % (name, tag)
            r = run(lab, a, prompt, rounds, n_predict=64)
            if r:
                out.append(r)
    save("suite_prefill", out)


def exp_ctx(rounds=2):
    """ctx 上限：f16 与 q8_0 KV 各自能撑多大，速度代价多少。"""
    out = []
    for tag, ctk, ctv in (("f16", "f16", "f16"), ("q8", "q8_0", "q8_0")):
        for ctx in (8192, 32768, 65536, 131072, 196608):
            lab = "ctx_%s_%d" % (tag, ctx)
            r = run(lab, build("mtp", 2, ctx=ctx, ctk=ctk, ctv=ctv),
                    "essay", rounds)
            if r is None:
                out.append({"label": lab, "median": 0, "mean_acc": 0,
                            "acc_rate": None, "vram": "OOM", "prefill": [],
                            "failed": True})
            else:
                out.append(r)
    save("suite_ctx", out)


def exp_combo(rounds=2):
    """多投机类型级联：ngram 在前（纯 CPU 查表，命中即省），MTP 兜底。"""
    cases = [
        ("com_mtp", build("mtp", 2)),
        ("com_ngram_mod", build("ngram-mod")),
        ("com_ngrammod_mtp", build(None, extra=["--spec-type", "ngram-mod,draft-mtp",
                                                "--spec-draft-n-max", "2"])),
        ("com_ngrammapk_mtp", build(None, extra=["--spec-type", "ngram-map-k,draft-mtp",
                                                 "--spec-draft-n-max", "2"])),
        ("com_mtp_ngrammod", build(None, extra=["--spec-type", "draft-mtp,ngram-mod",
                                                "--spec-draft-n-max", "2"])),
    ]
    out = []
    for label, a in cases:
        for p in ("rewrite", "code"):
            lab = "%s_%s" % (label, p)
            r = run(lab, a, p, rounds)
            if r:
                out.append(r)
    save("suite_combo", out)


def exp_env(rounds=2):
    """这个 fork 的运行时开关 A/B：验证哪些定制路径真的在起作用。"""
    filler = ("这是一段用于填充上下文长度的说明文字，内容本身没有特殊含义，"
              "只是为了让输入序列变长，从而测量预填充阶段的速度。")
    long_prompt = PROMPTS["essay"] + "\n" + filler * 450
    out = []

    def add(lab, args, prompt, env=None, npred=192):
        r = run(lab, args, prompt, rounds, npred, env=env)
        if r:
            out.append(r)

    # decode 侧：MMVQ / 融合 / 激活缓存
    for lab, env in (("env_base", None),
                     ("env_no_mmvq", {"GGML_CUDA_GP100_MMVQ_DISABLE": "1"}),
                     ("env_no_fusion", {"GGML_CUDA_DISABLE_FUSION": "1"}),
                     ("env_no_actcache", {"GGML_CUDA_GP100_NO_ACT_CACHE": "1"})):
        add("mtp_" + lab, build("mtp", 2, ctx=32768), "essay", env)
    # prefill 侧：FA-GEMM 是 fork 的核心优化
    for lab, env in (("env_fagemm_on", None), ("env_fagemm_off", {"GGML_CUDA_FA_GEMM": "0"})):
        add(lab, build("mtp", 2, ctx=32768), long_prompt, env, 64)
    save("suite_env", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "kv"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    {"kv": exp_kv, "content": exp_content, "family": exp_family,
     "prefill": exp_prefill, "ctx": exp_ctx, "combo": exp_combo,
     "env": exp_env}[which](rounds)
