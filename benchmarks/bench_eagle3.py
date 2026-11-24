"""EAGLE3 vs MTP 对照（目标模型 Q4_K_M）。

用法: python bench_eagle3.py
"""
import json
import os
import sys

import bench_quant as QQ
import bench_suite as B

EAGLE = "<FILL_PATH>"
OUT = os.path.join(B.LOGDIR, "eagle3_vs_mtp.json")


def mk(model, spec, n_max=3, ctx=8192, p_min=0.0):
    a = B.build(None, ctx=ctx)
    i = a.index("-m")
    a[i + 1] = model
    if spec == "eagle3":
        a += ["-md", EAGLE, "--spec-type", "draft-eagle3",
              "--spec-draft-n-max", str(n_max)]
    elif spec == "mtp":
        a += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max)]
    if p_min:
        a += ["--spec-draft-p-min", str(p_min)]
    return a


def main():
    m = QQ.MODELS["Q4_K_M"]
    cases = [("base", mk(m, None))]
    for n in (1, 2, 3, 4, 5, 6):
        cases.append(("eagle3_n%d" % n, mk(m, "eagle3", n)))
    cases.append(("eagle3_n6_p50", mk(m, "eagle3", 6, p_min=0.5)))
    for n in (2, 3):
        cases.append(("mtp_n%d" % n, mk(m, "mtp", n)))
    out = []
    for lab, a in cases:
        for prompt in ("math", "essay"):
            r = B.run("%s_%s" % (lab, prompt), a, prompt, 2)
            if r:
                r["case"] = lab
                r["prompt"] = prompt
                out.append(r)
    B.save("eagle3_vs_mtp", out)


if __name__ == "__main__":
    main()
