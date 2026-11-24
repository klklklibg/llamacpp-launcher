"""selftest.py - 无需人工点击的逻辑自检。

跑法：python.exe selftest.py
覆盖：模型扫描、命令行拼装、预设应用、UI↔配置同步、日志解析。
"""

from __future__ import annotations

import os
import sys

import llmcore as core

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def main() -> int:
    print("== 后端探测 ==")
    backends = core.discover_backends()
    check("至少发现 1 个可用后端", len(backends) >= 1, f"got {len(backends)}")
    for b in backends:
        print(f"   - {b.label:26} {b.version[:24]:26} caps={len(b.caps)}")
    main_be = next((b for b in backends
                    if "pascal" in b.label.lower() or "p100" in b.label.lower()), backends[0])
    check("主后端支持 --spec-type", "spec_type" in main_be.caps)

    print("== 模型扫描 ==")
    roots = core.load_config().get("model_roots") or core.DEFAULT_ROOTS
    models = core.scan_models(roots, progress=None, max_depth=4)
    check("扫到模型", len(models) > 0, f"got {len(models)}")
    big = [m for m in models if m.size > 5 * 1024**3]
    check("有 >5GB 模型", len(big) > 0)
    if big:
        print(f"   - 例：{big[0].name}  {big[0].size_h}  {big[0].quant}")

    print("== 命令行拼装 ==")
    cfg = core.load_config()
    cfg["backend_path"] = main_be.path
    cfg["params"].update({"port": 8123, "ctx": 8192, "ubatch": 1024, "fa": "on",
                          "parallel": 1, "ctk": "f16", "ctv": "f16",
                          "spec_enabled": False, "extra": ""})
    m = models[0] if models else None
    args = core.build_args(cfg, m, main_be.caps)
    joined = " ".join(args)
    check("含 -fa on", "-fa on" in joined, joined)
    check("含 -np 1", "-np 1" in joined, joined)
    check("含 --jinja", "--jinja" in joined)
    check("不含已废弃的 --draft-max", "--draft-max" not in joined)

    cfg["params"].update({"spec_enabled": True, "spec_model": r"C:\fake\draft.gguf",
                          "spec_type": "draft-simple", "spec_n_max": 4})
    spec_args = " ".join(core.build_args(cfg, m, main_be.caps))
    check("投机参数用新写法", "--spec-type draft-simple" in spec_args, spec_args)
    check("投机草稿长度", "--spec-draft-n-max 4" in spec_args, spec_args)

    cfg["params"].update({"ctk": "q8_0", "ctv": "q8_0", "load_mode": "mlock", "fit": True})
    adv = " ".join(core.build_args(cfg, m, main_be.caps))
    check("KV 量化参数", "-ctk q8_0 -ctv q8_0" in adv, adv)
    check("加载模式参数", "-lm mlock" in adv, adv)
    check("fit 参数", "-fit on" in adv, adv)

    print("== 视觉模块 ==")
    check("主后端已编译 mtmd (--mmproj)", "mmproj" in main_be.caps)
    cand = core.list_known_mmproj()
    check("已知投影候选 >= 3", len(cand) >= 3, f"got {len(cand)}")
    check("中缀命名也被识别",
          any(core._is_mmproj(os.path.basename(p)) and
              not os.path.basename(p).lower().startswith("mmproj") for p in cand),
          str(cand))
    bad = [m.name for m in models if core._is_mmproj(os.path.basename(m.path))]
    check("投影文件不再被当成模型", not bad, str(bad))
    vm = next((m for m in models if m.vision), None)
    check("至少一个模型自动配对投影（仅作候选）", vm is not None)
    check("默认配置 vision_enabled=False",
          core.DEFAULT_CONFIG["params"]["vision_enabled"] is False)
    if vm:
        base = {"params": dict(cfg["params"]), "backend_path": main_be.path}
        base["params"].update({"vision_enabled": False, "mmproj": "",
                               "mmproj_auto": False})
        off = " ".join(core.build_args(base, vm, main_be.caps))
        check("默认不加载视觉模块（即使模型已配对）", "--mmproj" not in off, off)

        on = {"params": dict(base["params"]), "backend_path": main_be.path}
        on["params"]["vision_enabled"] = True
        va = " ".join(core.build_args(on, vm, main_be.caps))
        check("显式开启后才下发 --mmproj", "--mmproj " in va, va)

        old = {"params": dict(on["params"]), "backend_path": main_be.path}
        oa = " ".join(core.build_args(old, vm, {"spec_type"}))
        check("旧后端不下发 --mmproj", "--mmproj" not in oa, oa)

        man = {"params": dict(on["params"]), "backend_path": main_be.path}
        man["params"]["mmproj"] = r"C:\fake\custom-mmproj.gguf"
        ma = " ".join(core.build_args(man, vm, main_be.caps))
        check("手动路径优先于自动配对", r"C:\fake\custom-mmproj.gguf" in ma, ma)

        au = {"params": dict(on["params"]), "backend_path": main_be.path}
        au["params"]["mmproj_auto"] = True
        aa = " ".join(core.build_args(au, vm, main_be.caps))
        check("--mmproj-auto 不与显式路径并存",
              "--mmproj-auto" in aa and "custom-mmproj" not in aa, aa)

        au_off = {"params": dict(au["params"]), "backend_path": main_be.path}
        au_off["params"]["vision_enabled"] = False
        ao = " ".join(core.build_args(au_off, vm, main_be.caps))
        check("总开关关闭时 auto 也不下发", "--mmproj-auto" not in ao, ao)

        txt = next((m for m in models if not m.vision), None)
        if txt:
            plain = {"params": dict(on["params"]), "backend_path": main_be.path}
            plain["params"]["mmproj"] = ""
            pa = " ".join(core.build_args(plain, txt, main_be.caps))
            check("纯文本模型即使开启也不下发", "--mmproj" not in pa, pa)

    print("== 日志解析 ==")
    sp = core.ServerProc(lambda l: None)
    sp._parse("0.02.131.152 I srv  llama_server: listening on http://127.0.0.1:8080")
    check("解析 listening", sp.listen_url == "http://127.0.0.1:8080", sp.listen_url)
    sp._parse("2.00.388.944 I slot print_timing: id  3 | task 0 | prompt eval time ="
              "      71.28 ms /    16 tokens (    4.45 ms per token,   224.47 tokens per second)")
    check("解析 prefill 速率", abs(sp.perf.pp_tps - 224.47) < 0.01, str(sp.perf.pp_tps))
    sp._parse("2.00.388.950 I slot print_timing: id  3 | task 0 |        eval time ="
              "     277.51 ms /    64 tokens (    4.40 ms per token,   227.01 tokens per second)")
    check("解析 decode 速率", abs(sp.perf.tg_tps - 227.01) < 0.01, str(sp.perf.tg_tps))

    print("== 预设（含 UI 同步）==")
    try:
        import launcher
        app = launcher.App()
        app.update()
        app._apply_preset("p100")
        app.update()
        fa_ui = app.v["fa"].get()
        fa_cfg = app.cfg["params"]["fa"]
        check("预设把 fa 设为 on（UI）", fa_ui == "on", fa_ui)
        check("预设把 fa 设为 on（cfg）", fa_cfg == "on", fa_cfg)
        check("hint 显示已应用", "p100" in app.lbl_hint.cget("text"))
        preview = app.txt_cmd.get("1.0", "end").strip()
        check("命令预览跟随预设", "-fa on" in preview, preview)

        app._apply_preset("long")
        app.update()
        check("长上下文预设 ctx=32768", app.cfg["params"]["ctx"] == 32768,
              str(app.cfg["params"]["ctx"]))
        check("长上下文预设 KV=q8_0", app.cfg["params"]["ctk"] == "q8_0")
        app.destroy()
    except Exception as e:
        check(f"UI 预设测试异常: {e}", False)

    print(f"\n结果：{PASS} 通过, {FAIL} 失败")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
