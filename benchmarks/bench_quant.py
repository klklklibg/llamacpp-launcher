"""量化版本 x 投机配置矩阵。

用法:
    python bench_quant.py                 # 全部
    python bench_quant.py Q4_K_M          # 只测某个量化
"""
import os
import sys

import bench_suite as B

MODELS = {
    "Q8_0": "<FILL_PATH>",
    "Q6_K": "<FILL_PATH>",
    "Q5_K_M": "<FILL_PATH>",
    "Q4_K_M": "<FILL_PATH>",
}


def mk(model, spec=None, n_max=2, ctx=8192, extra=None):
    """复刻 bench_suite.build，但允许替换模型文件。"""
    a = B.build(spec, n_max, ctx=ctx, extra=extra)
    i = a.index("-m")
    a[i + 1] = model
    return a


def main(argv):
    want = argv[1:] if len(argv) > 1 else list(MODELS)
    rounds = 2
    out = []
    for tag in want:
        path = MODELS[tag]
        if not os.path.exists(path):
            print("!! 缺失 %s" % path, flush=True)
            continue
        gb = os.path.getsize(path) / 1073741824
        print("\n########## %s  %.2f GB ##########" % (tag, gb), flush=True)
        cases = [
            ("%s_base_math" % tag, mk(path), "math", None),
            ("%s_base_essay" % tag, mk(path), "essay", None),
            ("%s_mtp2_math" % tag, mk(path, "mtp", 2), "math", None),
            ("%s_mtp2bs_math" % tag, mk(path, "mtp", 2, extra=["-bs"]), "math", None),
            ("%s_mtp2bs_essay" % tag, mk(path, "mtp", 2, extra=["-bs"]), "essay", None),
            ("%s_mtp3_math" % tag, mk(path, "mtp", 3), "math", None),
        ]
        for lab, a, prompt, env in cases:
            r = B.run(lab, a, prompt, rounds, env=env)
            if r:
                r["quant"] = tag
                r["gb"] = round(gb, 2)
                out.append(r)
    if out:
        B.save("bench_quant", out)


if __name__ == "__main__":
    main(sys.argv)
