"""llmcore.py - llama.cpp 启动器核心：后端探测、模型扫描、进程管理、日志解析。

零第三方依赖，仅标准库。所有路径逻辑集中在 DEFAULT_ROOTS / KNOWN_BACKENDS，
新增目录不必改 UI。
"""

from __future__ import annotations

import json
import mmap
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable

# 关键：老构建缺 cublas64_12.dll 时，Windows 加载器会弹模态错误框并卡住探测。
# 关掉所有系统级错误对话框，让缺 DLL 的后端安静地失败。
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x8001 | 0x0002)
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
CACHE_PATH = os.path.join(HERE, "models.cache.json")
BENCH_LOG = os.path.join(HERE, "bench_matrix.log")
# 本机专属路径放这里（gitignore）。仓库只提交 paths.example.json。
PATHS_PATH = os.path.join(HERE, "paths.local.json")


def _load_paths() -> dict:
    try:
        with open(PATHS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_PATHS = _load_paths()

# 模型扫描根目录（本机配置，见 paths.example.json）
DEFAULT_ROOTS = list(_PATHS.get("model_roots", []))

# llama-server 目录 -> 展示名。每项可写 {"path":..., "label":..., "role":...}，
# role 用于自动后端路由：p100 / inkling / pascal / pxa。
KNOWN_BACKENDS = [
    (b["path"], b.get("label", ""))
    for b in _PATHS.get("backends", []) if b.get("path")
]

# 按 role 取后端路径；未配置返回空串，路由函数会自动跳过。
def _be(role: str) -> str:
    for b in _PATHS.get("backends", []):
        if b.get("role") == role:
            return b.get("path", "")
    return ""

BE_P100    = _be("p100")
BE_INKLING = _be("inkling")
BE_PASCAL  = _be("pascal")
BE_PXA     = _be("pxa")


def recommend_backend(model: Model | None) -> str | None:
    """按模型返回推荐后端绝对路径；无法判断时返回 None（保持用户选择）。"""
    if model is None:
        return None
    p = (model.path or "").lower()
    # PXQ/PXQN 是 PXA 私有张量类型，标准 llama.cpp 会直接报 unknown type，
    # 必须优先判，否则会被下面几条规则抢走。
    if "pxqn" in p or "pxq" in p:
        if BE_PXA and os.path.isfile(os.path.join(BE_PXA, "llama-server.exe")):
            return BE_PXA
    if ("inkling2" in p or "inkling-2" in p) and BE_INKLING:
        return BE_INKLING
    if "sensenova" in p and BE_PASCAL:
        return BE_PASCAL
    # 只推荐本机确实存在 exe 的后端，避免指向空目录
    if BE_P100 and os.path.isfile(os.path.join(BE_P100, "llama-server.exe")):
        return BE_P100
    if BE_PASCAL and os.path.isfile(os.path.join(BE_PASCAL, "llama-server.exe")):
        return BE_PASCAL
    return None


def recommend_kv(model: Model | None) -> str:
    """按模型返回推荐的 KV cache 量化类型。

    实测 2026-11-23 旧后端 + 35B-A3B（P100 16GB）：
      -ctk/-ctv f16 : 78 t/s
      -ctk/-ctv q8_0: 66 t/s（-15%），因 q8_0 KV 读写要额外量化 kernel。
    35B 模型本体仅 12.7GB，显存不紧张，没必要用 q8_0 换显存。
    Inkling-2-Mini 需 q8_0：49 层全 offload 时 f16 KV 会顶到显存上限。
    """
    if model is None:
        return "f16"
    p = (model.path or "").lower()
    # PXQN-OneCard 12.6GB + 131k 上下文按官方配置用 q4_0 KV 才装得下；
    # 32k 下实测 q4_0 解码 27.4 t/s、prefill 224，无掉速。
    if "pxqn" in p:
        return "q4_0"
    if "inkling2" in p or "inkling-2" in p:
        return "q8_0"
    return "f16"

# 已知存放 mmproj 视觉投影的目录（本机配置）。只作为「视觉」页的候选清单，
# 不做跨目录自动配对——投影必须与主模型架构严格对应，猜错会静默出乱码。
MMPROJ_HINTS = tuple(_PATHS.get("mmproj_hints", []))


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _quant_from_name(name: str) -> str:
    up = name.upper()
    for q in ("PTQ1_0", "PQ2_0", "Q2_K_XL", "Q3_K_XL", "Q4_K_XL", "Q5_K_XL", "Q6_K_XL",
              "IQ4_NL", "IQ3_S", "IQ3_XS", "IQ2_M", "Q8_0",
              "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M",
              "BF16", "F16", "NVFP4", "GSQ", "MTP", "3BIT"):
        if q in up:
            return q
    return ""


@dataclass
class Model:
    path: str
    name: str
    size: int
    quant: str = ""
    mtime: float = 0.0
    mmproj: str = ""
    mtp: bool = False
    vision: bool = False

    @property
    def size_h(self) -> str:
        return human_size(self.size)

    @property
    def dirname(self) -> str:
        return os.path.dirname(self.path)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("path", None)
        return d


@dataclass
class Backend:
    path: str          # 目录
    label: str = ""
    version: str = ""
    caps: set = field(default_factory=set)

    @property
    def exe(self) -> str:
        return os.path.join(self.path, "llama-server.exe")

    def to_dict(self) -> dict:
        return {"path": self.path, "label": self.label, "version": self.version}


# ---------------------------------------------------------------- 后端探测

CAP_FLAGS = {
    "flash_attn": ["-fa", "--flash-attn"],
    "spec_type": ["--spec-type"],
    "spec_draft": ["--spec-draft-model"],
    "load_mode": ["-lm", "--load-mode"],
    "fit": ["-fit", "--fit"],
    "jinja": ["--jinja"],
    "cache_type": ["-ctk", "--cache-type-k"],
    "metrics": ["--metrics"],
    "kv_unified": ["-kvu", "--kv-unified"],
    "no_host": ["--no-host"],
    "tensor_lazy": ["--tensor-read-lazy"],
    "cache_ram": ["-cram", "--cache-ram"],
    "ctx_checkpoints": ["-ctxcp", "--ctx-checkpoints"],
    "agent": ["-ag", "--agent"],
    # 视觉模块（mtmd）。--mmproj 存在即说明该后端编译了多模态投影支持；
    # --mmproj-auto 单独探测，旧后端没有这个开关时只能显式给路径。
    "mmproj": ["--mmproj"],
    "mmproj_auto": ["--mmproj-auto"],
}


def deps_ok(path: str) -> tuple[bool, str]:
    """预检依赖，避免把缺 DLL 的构建当成可用后端（否则会弹系统错误框）。"""
    exe = os.path.join(path, "llama-server.exe")
    if not os.path.isfile(exe):
        return False, "无 llama-server.exe"
    try:
        names = os.listdir(path)
    except OSError:
        return False, "目录不可读"
    low = {n.lower() for n in names}
    if "ggml-cuda.dll" in low:
        for need in ("cublas64_12.dll", "cudart64_12.dll"):
            if need not in low:
                return False, f"缺 {need}"
    if not any(n.startswith(("ggml", "llama")) and n.endswith(".dll") for n in low):
        return False, "缺运行时 DLL"
    return True, ""


def probe_backend(path: str, label: str = "") -> Backend | None:
    ok, _why = deps_ok(path)
    if not ok:
        return None
    be = Backend(path=path, label=label or os.path.basename(path))
    try:
        r = subprocess.run([os.path.join(path, "llama-server.exe"), "--version"],
                           capture_output=True, text=True, timeout=20,
                           creationflags=0x08000000)
        text = (r.stdout or "") + (r.stderr or "")
        for line in text.splitlines():
            if "version:" in line:
                be.version = line.split("version:", 1)[1].strip()
                break
    except Exception:
        pass
    try:
        r = subprocess.run([os.path.join(path, "llama-server.exe"), "--help"],
                           capture_output=True, text=True, timeout=15,
                           creationflags=0x08000000)
        helptext = (r.stdout or "") + (r.stderr or "")
        for cap, flags in CAP_FLAGS.items():
            if any(f in helptext for f in flags):
                be.caps.add(cap)
    except Exception:
        pass
    return be


_CAPS_CACHE: dict[str, set] = {}


def caps_for(path: str) -> set:
    """取某后端支持的特性集，带缓存；未知时返回空集表示"按新版处理"。"""
    if path in _CAPS_CACHE:
        return _CAPS_CACHE[path]
    be = probe_backend(path)
    caps = be.caps if be else set()
    _CAPS_CACHE[path] = caps
    return caps


# ------------------------------------------------------------ MTP 头探测
# llama.cpp 的 --spec-type 默认是 none（common.h: types = {TYPE_NONE}），
# draft-mtp 必须显式指定。模型能不能用 MTP，取决于 GGUF 里有没有 nextn 预测头：
# 本机实测该 key 位于文件第 1.5 KB（kv[30] qwen35.nextn_predict_layers），
# 因此只扫开头若干 MB 即可，不必解析整个文件（llama-gguf r 要 12 秒）。
_MTP_CACHE: dict[tuple, bool] = {}
_MTP_SCAN_FAST = 8 << 20     # 8 MB：命中率已足够
_MTP_SCAN_SLOW = 32 << 20    # 兜底扫描
_MTP_MARKERS = (b"nextn_predict_layers", b".nextn.")


def _file_sig(path: str) -> tuple:
    try:
        st = os.stat(path)
    except OSError:
        return (path, 0, 0.0)
    return (path, st.st_size, st.st_mtime)


def has_mtp_file(path: str) -> bool:
    """判断 GGUF 是否带 MTP(nextn) 预测头，带缓存。读失败一律返回 False。"""
    if not path or not os.path.isfile(path):
        return False
    sig = _file_sig(path)
    if sig in _MTP_CACHE:
        return _MTP_CACHE[sig]
    ok = False
    try:
        with open(path, "rb") as f:
            if f.read(4) == b"GGUF":
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    for limit in (_MTP_SCAN_FAST, _MTP_SCAN_SLOW):
                        end = min(len(mm), limit)
                        if any(mm.find(m, 0, end) >= 0 for m in _MTP_MARKERS):
                            ok = True
                            break
                finally:
                    try:
                        mm.close()
                    except BufferError:
                        pass
    except (OSError, ValueError):
        ok = False
    _MTP_CACHE[sig] = ok
    return ok


def model_has_mtp(model: "Model | None") -> bool:
    return bool(model and has_mtp_file(model.path))


def discover_backends() -> list[Backend]:
    out: list[Backend] = []
    seen = set()
    # 已知目录优先
    for path, label in KNOWN_BACKENDS:
        if path in seen:
            continue
        be = probe_backend(path, label)
        if be:
            out.append(be)
            _CAPS_CACHE[path] = be.caps
            seen.add(path)
    # 兜底：从配置里的额外目录
    cfg = load_config()
    for path in cfg.get("extra_backends", []):
        if path in seen:
            continue
        be = probe_backend(path)
        if be:
            out.append(be)
            seen.add(path)
    return out


# ---------------------------------------------------------------- 模型扫描

def _is_mmproj(fn: str) -> bool:
    """视觉投影文件：mmproj 可以出现在文件名的任意位置。

    实测本机存在 Ternary-Bonsai-2-27B-mmproj-BF16.gguf 这类中缀命名，
    旧版只认 startswith("mmproj")，会把它漏掉并当成独立模型列出来。
    """
    low = fn.lower()
    return low.endswith(".gguf") and "mmproj" in low


def _find_mmproj(gguf_path: str) -> str:
    """只在主模型同目录配对投影。多个候选取体积最大的（通常是 BF16 原版）。

    刻意不做跨目录自动配对：投影必须与主模型架构严格对应，
    按名字相似度猜会把 Qwen3.5-0.8B 配到 9B 的 merger 上，且不会报错。
    跨目录的场景由「视觉」页手动指定或 --mmproj-auto 交给后端处理。
    """
    d = os.path.dirname(gguf_path)
    try:
        cands = [os.path.join(d, fn) for fn in os.listdir(d) if _is_mmproj(fn)]
    except OSError:
        return ""
    return max(cands, key=os.path.getsize) if cands else ""


def list_known_mmproj() -> list[str]:
    """「视觉」页下拉候选：hints 目录 + 各模型根目录下的 mmproj 文件。"""
    dirs = list(MMPROJ_HINTS) + [r for r in DEFAULT_ROOTS if os.path.isdir(r)]
    out: list[str] = []
    seen: set[str] = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        try:
            for fn in os.listdir(d):
                if not _is_mmproj(fn):
                    continue
                full = os.path.join(d, fn)
                key = os.path.normcase(full)
                if key not in seen:
                    seen.add(key)
                    out.append(full)
        except OSError:
            continue
    return sorted(out)


def scan_models(roots: Iterable[str], progress: Callable[[str], None] | None = None,
                max_depth: int = 5) -> list[Model]:
    models: list[Model] = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        base_depth = root.rstrip("\\/").count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - base_depth >= max_depth:
                dirnames[:] = []
            # 排除名字以 .gguf 结尾的【目录】（不是文件）。这类目录是 Unsloth 官方
            # 动态量化包的下载容器，里面是 Desktop 专用布局（arch=inkling2,
            # block_count=48，但只有 24 个物理层张量），主线 llama.cpp 加载会报
            # blk.24.attn_q.weight not found。这些是正版完整文件，只是布局不兼容，
            # 磁盘文件一律不动；llama.cpp 用的 Mainline 版在同级的 Mainline/ 目录。
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and not d.lower().endswith(".gguf")]
            for fn in filenames:
                if not fn.lower().endswith(".gguf"):
                    continue
                if _is_mmproj(fn):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                if st.st_size < 50 * 1024 * 1024:  # 50MB 以下多是投影/测试件
                    continue
                name = os.path.splitext(fn)[0]
                models.append(Model(
                    path=full, name=name, size=st.st_size,
                    quant=_quant_from_name(name), mtime=st.st_mtime,
                    mtp="MTP" in fn.upper(),
                ))
        if progress:
            progress(f"已扫描 {root}")
    # 补 mmproj / vision
    for m in models:
        m.mmproj = _find_mmproj(m.path)
        m.vision = bool(m.mmproj)
    models.sort(key=lambda m: (-m.mtime, m.name))
    return models


def load_model_cache() -> list[Model]:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [Model(**m) for m in raw]
    except Exception:
        return []


def save_model_cache(models: list[Model]) -> None:
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in models], f, ensure_ascii=False, indent=1)
    except Exception:
        pass


# ---------------------------------------------------------------- 配置

DEFAULT_CONFIG = {
    "backend_path": _PATHS.get("backend_path", ""),
    "auto_backend": True,   # 按模型架构自动选最快后端（Inkling→master，其余→Pascal）
    "model_roots": DEFAULT_ROOTS,
    "extra_backends": [],
    "last_model": "",
    "recent": [],
    "presets": {},          # 用户自定义预设
    "params": {             # 上次使用的参数
        "ctx": 8192, "ngl": 99, "batch": 2048, "ubatch": 1024,
        "fa": "on", "ctk": "f16", "ctv": "f16",
        "threads": 6, "threads_batch": 0,
        "host": "127.0.0.1", "port": 8080, "parallel": 1,
        "load_mode": "none", "fit": False, "fit_ctx": 4096,
        "jinja": True, "metrics": True, "no_mmap": False,
        "api_key": "", "alias": "", "extra": "",
        "spec_enabled": True, "spec_model": "", "spec_type": "auto",
        "spec_n_max": 2, "spec_ngl": 99,
        "reasoning_budget": 0,
        # 视觉默认关闭：实测挂载 mmproj 后 decode 掉约 18%，必须显式开启。
        "vision_enabled": False, "mmproj": "", "mmproj_auto": False,
    },
}


def load_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user = json.load(f)
        cfg.update({k: v for k, v in user.items() if k != "params"})
        cfg["params"].update(user.get("params", {}))
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return cfg


def save_config(cfg: dict) -> None:
    tmp = CONFIG_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_PATH)
    except Exception:
        pass


# ---------------------------------------------------------------- 命令行构建

SPEC_NOTE = ""   # 最近一次 build_args 对投机参数的说明（供 UI 提示）


def build_args(cfg: dict, model: Model | None, caps: set | None = None) -> list[str]:
    p = cfg["params"]
    if caps is None:
        caps = caps_for(cfg.get("backend_path", ""))
    modern = ("spec_type" in caps) or not caps
    a: list[str] = []
    if model:
        a += ["-m", model.path]
        # 视觉投影默认完全不加载。实测挂载 mmproj 后 token 生成速度掉约 18%，
        # 因此同目录扫描到的 mmproj 只作为「视觉」页候选，绝不自动进入命令行；
        # 必须由用户显式勾选 vision_enabled 才下发任何 --mmproj* 参数。
        # 后端没编 mtmd 时不下发，否则旧构建会直接以 unknown argument 退出。
        if p.get("vision_enabled") and (("mmproj" in caps) or not caps):
            if p.get("mmproj_auto") and "mmproj_auto" in caps:
                a += ["--mmproj-auto"]
            else:
                mm = (p.get("mmproj") or "").strip() or (model.mmproj if model.vision else "")
                if mm:
                    a += ["--mmproj", mm]
    a += ["-ngl", str(p["ngl"])]
    a += ["-c", str(p["ctx"])]
    a += ["-b", str(p["batch"]), "-ub", str(p["ubatch"])]
    if p["threads"]:
        a += ["-t", str(p["threads"])]
    if p.get("threads_batch"):
        a += ["-tb", str(p["threads_batch"])]
    if "flash_attn" in caps or not caps:
        a += ["-fa", str(p["fa"])]
    if p["ctk"] != "f16" or p["ctv"] != "f16":
        a += ["-ctk", p["ctk"], "-ctv", p["ctv"]]
    # 只对明确支持 -lm 的后端传；旧后端传了会直接退出
    if p.get("load_mode") and p["load_mode"] != "auto" and "load_mode" in caps:
        a += ["-lm", p["load_mode"]]
    if p.get("fit") and ("fit" in caps or not caps):
        a += ["-fit", "on", "-fitc", str(p.get("fit_ctx", 4096))]
    if p.get("jinja"):
        a += ["--jinja"]
    if p.get("metrics") and ("metrics" in caps or not caps):
        a += ["--metrics"]
    if p.get("api_key"):
        a += ["--api-key", p["api_key"]]
    if p.get("alias"):
        a += ["-a", p["alias"]]
    a += ["-np", str(p.get("parallel", 1) or 1)]
    a += ["--host", p["host"], "--port", str(p["port"])]
    # 投机解码
    # draft-mtp 用模型自带的 nextn 预测头，ngram 系列靠上下文匹配，都不需要 -md。
    # spec_type=auto：模型带 nextn 头才启用 MTP，否则静默不下发任何投机参数。
    global SPEC_NOTE
    SPEC_NOTE = ""
    stype = (p.get("spec_type") or "").strip()
    if stype == "auto" and p.get("spec_enabled"):
        if model_has_mtp(model):
            stype = "draft-mtp"
            SPEC_NOTE = "auto：检测到 nextn 头，启用 MTP"
        else:
            stype = ""
            SPEC_NOTE = "auto：该模型无 nextn 头，未启用投机"
    if p.get("spec_enabled") and stype:
        # 显式选了 draft-mtp 但模型其实没有 MTP 头：不能下发，否则直接报错退出
        if stype == "draft-mtp" and model is not None and not model_has_mtp(model):
            SPEC_NOTE = "draft-mtp：该模型无 nextn 头，已跳过"
            stype = ""
    # PXA 引擎的投机写法不同：--spec-type mtp:n_max=N，且不认 --spec-draft-n-max。
    # 实测 27B PXQN：开 MTP +4.4%（27.17 vs 25.87），是同机唯一为正的投机实现。
    if p.get("spec_enabled") and stype and (cfg.get("backend_path") or "").lower().startswith(r"d:\pxa"):
        if stype == "draft-mtp":
            n = max(1, int(p.get("spec_n_max") or 1))
            a += ["--spec-type", f"mtp:n_max={n}"]
            SPEC_NOTE = f"PXA：MTP n_max={n}"
        stype = ""
    if p.get("spec_enabled") and stype:
        needs_model = not (stype == "draft-mtp" or stype.startswith("ngram"))
        if p.get("spec_model") or not needs_model:
            if modern:
                if needs_model:
                    a += ["-md", p["spec_model"]]
                if "spec_type" in caps:
                    a += ["--spec-type", stype]
                if needs_model and p.get("spec_ngl", 99) != 99:
                    a += ["-ngld", str(p["spec_ngl"])]
                if p.get("spec_n_max"):
                    a += ["--spec-draft-n-max", str(p["spec_n_max"])]
            else:
                a += ["-md", p["spec_model"], "--draft-max", str(p.get("spec_n_max", 3))]
    if p.get("extra"):
        a += p["extra"].split()
    return a


def quote_cmd(parts: list[str]) -> str:
    out = []
    for x in parts:
        out.append(f'"{x}"' if (" " in x or "\t" in x) else x)
    return " ".join(out)


def write_bat(cfg: dict, model: Model | None, path: str) -> str:
    be = cfg["backend_path"]
    exe = os.path.join(be, "llama-server.exe")
    args = build_args(cfg, model)
    lines = [
        "@echo off",
        "chcp 65001 >nul",
        "title llama.cpp server",
        f'cd /d "{be}"',
        f'"{exe}" {quote_cmd(args)}',
        "echo.",
        "echo [server exited]",
        "pause",
    ]
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------- 进程管理

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# 新老版本日志兼容
RE_PROMPT_EVAL = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens?.*?([\d.]+)\s*tokens per second")
RE_EVAL = re.compile(
    r"^.*\beval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs?|tokens?).*?([\d.]+)\s*tokens per second")
RE_PROMPT_EVAL_ALT = re.compile(r"prompt eval time.*?([\d.]+) tokens per second")
RE_EVAL_ALT = re.compile(r"^\s*eval time.*?([\d.]+) tokens per second")
RE_LOAD_MS = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")
RE_MODEL_LOADED = re.compile(r"llama_server:\s*model loaded")
RE_LISTEN = re.compile(r"listening on\s+(https?://\S+)")
RE_ERR = re.compile(r"\b(error|failed|CUDA error|out of memory|OOM)\b", re.I)


@dataclass
class Perf:
    load_ms: float = 0.0
    pp_tps: float = 0.0
    tg_tps: float = 0.0
    pp_ms: float = 0.0
    tg_ms: float = 0.0
    pp_tokens: int = 0
    tg_tokens: int = 0


class ServerProc:
    """管理一个 llama-server 子进程，线程安全地把日志交给 UI。"""

    def __init__(self, on_line: Callable[[str], None],
                 on_state: Callable[[str, str], None] | None = None):
        self.proc: subprocess.Popen | None = None
        self.on_line = on_line
        self.on_state = on_state or (lambda s, m: None)
        self.perf = Perf()
        self.listen_url = ""
        self.started_at = 0.0
        self._lock = threading.Lock()
        self._logfile = None
        self.log_path = ""

    # -- 生命周期
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cfg: dict, model: Model | None) -> bool:
        if self.is_running():
            return False
        be = cfg["backend_path"]
        exe = os.path.join(be, "llama-server.exe")
        if not os.path.isfile(exe):
            self.on_state("error", f"找不到 {exe}")
            return False
        args = build_args(cfg, model)
        self.perf = Perf()
        self.listen_url = ""
        self.started_at = time.time()
        try:
            self._logfile = open(os.path.join(HERE, "server.log"), "w",
                                 encoding="utf-8", errors="replace")
            self.log_path = self._logfile.name
            env = os.environ.copy()
            env["LLAMA_LOG_COLORS"] = "0"
            # Inkling2's Q3_K MoE becomes incoherent with the GP100 fp16 MMVQ kernel
            # (measured: same tg but wrong answers), while 35B-A3B gains +27% from it.
            # Disable it only for Inkling2 so the launcher needs no per-model manual env vars.
            if model and ("inkling2" in (model.path or "").lower() or "inkling-2" in (model.path or "").lower()):
                env["GGML_CUDA_GP100_MMVQ_DISABLE"] = "1"
            self.proc = subprocess.Popen(
                [exe] + args, cwd=be, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                bufsize=1, env=env, creationflags=0x08000000,
            )
        except Exception as e:
            self.on_state("error", f"启动失败: {e}")
            return False
        self.on_state("loading", "")
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._watcher, daemon=True).start()
        return True

    def _reader(self) -> None:
        proc = self.proc
        if not proc or not proc.stdout:
            return
        for raw in proc.stdout:
            line = ANSI_RE.sub("", raw.rstrip("\r\n"))
            if not line:
                continue
            if self._logfile:
                try:
                    self._logfile.write(line + "\n")
                    self._logfile.flush()
                except Exception:
                    pass
            self._parse(line)
            self.on_line(line)

    def _parse(self, line: str) -> None:
        m = RE_PROMPT_EVAL.search(line)
        if m:
            self.perf.pp_ms = float(m.group(1))
            self.perf.pp_tokens = int(m.group(2))
            self.perf.pp_tps = float(m.group(3))
            return
        m = RE_EVAL.search(line)
        if m:
            self.perf.tg_ms = float(m.group(1))
            self.perf.tg_tokens = int(m.group(2))
            self.perf.tg_tps = float(m.group(3))
            return
        m = RE_PROMPT_EVAL_ALT.search(line)
        if m and not self.perf.pp_tps:
            self.perf.pp_tps = float(m.group(1))
            return
        m = RE_EVAL_ALT.search(line)
        if m and not self.perf.tg_tps:
            self.perf.tg_tps = float(m.group(1))
            return
        m = RE_LOAD_MS.search(line)
        if m:
            self.perf.load_ms = float(m.group(1))
            return
        m = RE_LISTEN.search(line)
        if m:
            self.listen_url = m.group(1)
            if not self.perf.load_ms and self.started_at:
                self.perf.load_ms = (time.time() - self.started_at) * 1000.0
            self.on_state("ready", self.listen_url)
            return
        if RE_MODEL_LOADED.search(line) and not self.perf.load_ms and self.started_at:
            self.perf.load_ms = (time.time() - self.started_at) * 1000.0
        if RE_ERR.search(line):
            self.on_state("warn", line.strip()[:160])

    def _watcher(self) -> None:
        while self.is_running():
            time.sleep(0.5)
        code = self.proc.poll() if self.proc else -1
        if self._logfile:
            try:
                self._logfile.close()
            except Exception:
                pass
        self.on_state("stopped", f"退出码 {code}")

    def stop(self, timeout: float = 8.0) -> None:
        proc = self.proc
        if not proc or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=timeout)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


# ---------------------------------------------------------------- 环境探测

def port_free(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.4)
    try:
        return s.connect_ex((host if host not in ("0.0.0.0", "") else "127.0.0.1", port)) != 0
    except Exception:
        return True
    finally:
        s.close()


def http_json(url: str, timeout: float = 2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def gpu_stats() -> dict:
    """nvidia-smi 单次查询，失败返回空 dict。"""
    exe = shutil.which("nvidia-smi") or r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    try:
        r = subprocess.run(
            [exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, creationflags=0x08000000)
        parts = [x.strip() for x in (r.stdout or "").strip().split(",")]
        if len(parts) >= 7:
            return {"name": parts[0], "util": parts[1], "mem_used": parts[2],
                    "mem_total": parts[3], "temp": parts[4], "power": parts[5],
                    "clock": parts[6], "ok": True}
    except Exception:
        pass
    return {"ok": False}


def server_slots(port: int, host: str = "127.0.0.1") -> dict | None:
    return http_json(f"http://{host}:{port}/slots")


def read_bench_results() -> list[dict]:
    """解析 bench_matrix.log 成结构化结果，供 UI 展示。"""
    if not os.path.isfile(BENCH_LOG):
        return []
    out, cur = [], None
    with open(BENCH_LOG, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("###"):
                cur = {"tag": line[3:].strip(), "rows": []}
                out.append(cur)
            elif line.startswith("|") and cur is not None:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) >= 7 and cells[5] in ("pp512", "tg64"):
                    try:
                        cur["rows"].append({"test": cells[5], "tps": float(cells[6].split("±")[0])})
                    except ValueError:
                        pass
    return [x for x in out if x["rows"]]
