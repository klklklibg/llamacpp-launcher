"""量化版本（权重位宽）的质量验证：同一批题、贪心解码，只改模型文件。

用法: python bench_quant_quality.py Q8_0 Q4_K_M
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_9b_quality as Q
import bench_quant as QQ
import bench_suite as B

OUT = os.path.join(B.LOGDIR, "quant_quality.json")


def start(model, tag):
    args = QQ.mk(model, "mtp", 2, ctx=8192, extra=["-bs"])
    log = open(os.path.join(B.LOGDIR, "qq_%s.log" % tag), "w",
               encoding="utf-8", errors="replace")
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 300:
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % B.PORT,
                                   timeout=2).read()
            return p, log, True
        except Exception:
            if p.poll() is not None:
                break
            time.sleep(0.4)
    return p, log, False


def main(argv):
    tags = argv[1:] or ["Q8_0", "Q4_K_M"]
    results = []
    for tag in tags:
        path = QQ.MODELS[tag]
        if not os.path.exists(path):
            print("!! 缺失 %s" % path, flush=True)
            continue
        print("\n=== 权重 %s ===" % tag, flush=True)
        p, log, ok = start(path, tag)
        try:
            if not ok:
                print("启动失败", flush=True)
                continue
            summary, rows = Q.run(tag, B.PORT)
            summary["gb"] = round(os.path.getsize(path) / 1073741824, 2)
            results.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        finally:
            p.terminate()
            try:
                p.wait(timeout=25)
            except subprocess.TimeoutExpired:
                p.kill()
            log.close()
            time.sleep(3)
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n===== 权重量化质量 =====")
    for r in results:
        print("%-7s %.2fGB  acc %.1f%%  %d/%d  tok/q %.0f  by_cat %s"
              % (r["tag"], r["gb"], r["accuracy"], r["correct"], r["n"],
                 r["tokens_per_question"],
                 {k: v["acc"] for k, v in r["by_cat"].items()}))


if __name__ == "__main__":
    main(sys.argv)
