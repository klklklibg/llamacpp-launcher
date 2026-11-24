"""KV cache 量化的质量验证：同一批题、贪心解码，只改 -ctk/-ctv。

用法: python bench_kv_quality.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_9b_quality as Q
import bench_suite as B

OUT = os.path.join(B.LOGDIR, "kv_quality.json")


def start(ctk, ctv, tag):
    args = B.build("mtp", 2, ctx=8192, ctk=ctk, ctv=ctv)
    log = open(os.path.join(B.LOGDIR, "kvq_%s.log" % tag), "w",
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


def main():
    results = []
    for tag, ctk, ctv in (("f16", "f16", "f16"),
                          ("q8_0", "q8_0", "q8_0"),
                          ("q4_0", "q4_0", "q4_0")):
        print("\n=== KV %s ===" % tag, flush=True)
        p, log, ok = start(ctk, ctv, tag)
        try:
            if not ok:
                print("启动失败", flush=True)
                continue
            summary, rows = Q.run(tag, B.PORT)
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
    print("\n===== KV 量化质量 =====")
    for r in results:
        print("%-6s acc %.1f%%  %d/%d  tok/q %.0f  by_cat %s"
              % (r["tag"], r["accuracy"], r["correct"], r["n"],
                 r["tokens_per_question"],
                 {k: v["acc"] for k, v in r["by_cat"].items()}))


if __name__ == "__main__":
    main()
