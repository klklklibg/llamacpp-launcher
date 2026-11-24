"""launcher.py - LlamaCpp 启动器主程序（Tk 界面）。

用法:
    pythonw.exe launcher.py
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import tkinter.simpledialog as simpledialog
from tkinter import filedialog, messagebox, ttk

import llmcore as core

# ------------------------------------------------------------------ 主题

BG = "#12141a"
PANEL = "#191c24"
PANEL2 = "#21252f"
LINE = "#2c3140"
FG = "#e8eaf0"
FG_DIM = "#8b93a7"
ACCENT = "#4c9aff"
OK = "#3ecf8e"
WARN = "#f5a524"
ERR = "#f65b5b"
MONO = "Consolas"

FONT = ("Microsoft YaHei UI", 9)
FONT_B = ("Microsoft YaHei UI", 9, "bold")
FONT_TITLE = ("Microsoft YaHei UI", 12, "bold")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = core.load_config()
        self.models: list[core.Model] = core.load_model_cache()
        self.backends: list[core.Backend] = []
        self.selected: core.Model | None = None
        self.q: queue.Queue = queue.Queue()
        self.server = core.ServerProc(self._on_line, self._on_state)
        self.state = "idle"
        self.gpu: dict = {"ok": False}

        self.title("LlamaCpp 启动器")
        try:
            self.iconbitmap(os.path.join(core.HERE, "assets", "llama.ico"))
        except Exception:
            pass
        self.configure(bg=BG)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(1340, max(1100, sw - 120))
        h = min(900, max(700, sh - 160))
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(10, (sh - h) // 2 - 20)}")
        self.minsize(1100, 700)
        self._init_style()
        self._build()
        self._bind_vars()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(80, self._drain)
        self.after(1200, self._tick_gpu)
        if self.models:
            self._fill_models()
        # 后台加载
        threading.Thread(target=self._boot, daemon=True).start()

    # ---------------------------------------------------------- 样式
    def _init_style(self) -> None:
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground=FG, fieldbackground=PANEL2,
                     bordercolor=LINE, lightcolor=LINE, darkcolor=LINE,
                     font=FONT)
        st.configure("TFrame", background=BG)
        st.configure("Panel.TFrame", background=PANEL)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Panel.TLabel", background=PANEL, foreground=FG)
        st.configure("Dim.TLabel", background=PANEL, foreground=FG_DIM, font=("Microsoft YaHei UI", 8))
        st.configure("Head.TLabel", background=PANEL, foreground=ACCENT,
                     font=("Microsoft YaHei UI", 9, "bold"))
        st.configure("Title.TLabel", background=BG, foreground=FG, font=FONT_TITLE)
        st.configure("TButton", background=PANEL2, foreground=FG, borderwidth=0,
                     focusthickness=0, padding=(10, 6))
        st.map("TButton", background=[("active", "#2d3342"), ("disabled", PANEL)],
               foreground=[("disabled", "#4a5162")])
        st.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                     font=FONT_B, padding=(14, 7))
        st.map("Accent.TButton", background=[("active", "#3d86e0"), ("disabled", "#2a3d5c")],
               foreground=[("disabled", "#7d8ba3")])
        st.configure("Danger.TButton", background="#7a2b2b", foreground="#ffd9d9",
                     font=FONT_B, padding=(14, 7))
        st.map("Danger.TButton", background=[("active", "#963434")])
        st.configure("TEntry", fieldbackground=PANEL2, foreground=FG,
                     insertcolor=FG, borderwidth=0, padding=4)
        st.configure("TCombobox", fieldbackground=PANEL2, background=PANEL2,
                     foreground=FG, arrowcolor=FG, borderwidth=0, padding=3)
        st.map("TCombobox", fieldbackground=[("readonly", PANEL2)],
               foreground=[("readonly", FG)])
        self.option_add("*TCombobox*Listbox.background", PANEL2)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        st.configure("TCheckbutton", background=PANEL, foreground=FG)
        st.map("TCheckbutton", background=[("active", PANEL)])
        st.configure("TSpinbox", fieldbackground=PANEL2, foreground=FG,
                     arrowcolor=FG, borderwidth=0, padding=3)
        st.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                     foreground=FG, borderwidth=0, rowheight=24)
        st.configure("Treeview.Heading", background=PANEL2, foreground=FG_DIM,
                     borderwidth=0, font=("Microsoft YaHei UI", 8))
        st.map("Treeview", background=[("selected", "#2b4a7a")],
               foreground=[("selected", "#ffffff")])
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", background=PANEL, foreground=FG_DIM,
                     padding=(14, 6), borderwidth=0)
        st.map("TNotebook.Tab", background=[("selected", PANEL2)],
               foreground=[("selected", FG)])
        st.configure("Vertical.TScrollbar", background=PANEL2, troughcolor=BG,
                     borderwidth=0, arrowcolor=FG_DIM)

    # ---------------------------------------------------------- 布局
    def _build(self) -> None:
        root = ttk.Frame(self, style="TFrame")
        root.pack(fill="both", expand=True, padx=10, pady=(8, 6))
        root.columnconfigure(0, weight=0, minsize=345)
        root.columnconfigure(1, weight=0, minsize=330)
        root.columnconfigure(2, weight=1)
        root.rowconfigure(1, weight=1)

        # ---- 顶部（两行：标题+GPU / 后端+版本）
        top = ttk.Frame(root, style="TFrame")
        top.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="LlamaCpp 启动器", style="Title.TLabel"
                  ).grid(row=0, column=0, sticky="w")
        self.lbl_gpu = tk.Label(top, text="GPU 探测中…", bg=BG, fg=FG_DIM,
                                font=("Microsoft YaHei UI", 8))
        self.lbl_gpu.grid(row=0, column=2, sticky="e")

        tk.Label(top, text="后端", bg=BG, fg=FG_DIM,
                 font=("Microsoft YaHei UI", 8)).grid(row=1, column=0, sticky="w", pady=(6, 0))
        be_frame = tk.Frame(top, bg=BG)
        be_frame.grid(row=1, column=1, sticky="w", padx=(6, 8), pady=(6, 0))
        self.cb_backend = ttk.Combobox(be_frame, state="readonly", width=30, values=[])
        self.cb_backend.grid(row=0, column=0, sticky="w")
        self.cb_backend.bind("<<ComboboxSelected>>", self._on_backend_pick)
        self.var_auto_be = tk.BooleanVar(value=bool(self.cfg.get("auto_backend", True)))
        tk.Checkbutton(be_frame, text="自动", variable=self.var_auto_be,
                       bg=BG, fg=FG_DIM, activebackground=BG, highlightthickness=0,
                       font=("Microsoft YaHei UI", 8),
                       command=self._on_auto_be_toggle).grid(row=0, column=1, padx=(6, 0))
        self.lbl_ver = tk.Label(top, text="", bg=BG, fg=FG_DIM,
                                font=("Microsoft YaHei UI", 8), anchor="e")
        self.lbl_ver.grid(row=1, column=2, sticky="e", pady=(6, 0))

        # ---- 左：模型
        left = ttk.Frame(root, style="Panel.TFrame")
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        head = ttk.Frame(left, style="Panel.TFrame")
        head.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        ttk.Label(head, text="模型", style="Head.TLabel").pack(side="left")
        self.lbl_count = ttk.Label(head, text="", style="Dim.TLabel")
        self.lbl_count.pack(side="left", padx=6)
        ttk.Button(head, text="刷新", width=5, command=self._rescan).pack(side="right")

        self.var_search = tk.StringVar()
        e = ttk.Entry(left, textvariable=self.var_search)
        e.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        e.insert(0, "")
        self.var_search.trace_add("write", lambda *_: self._fill_models())

        wrap = ttk.Frame(left, style="Panel.TFrame")
        wrap.grid(row=2, column=0, sticky="nsew", padx=(10, 4), pady=(0, 6))
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(wrap, columns=("q", "s"), show="tree headings",
                                 selectmode="browse")
        self.tree.heading("#0", text="名称", anchor="w")
        self.tree.heading("q", text="量化", anchor="w")
        self.tree.heading("s", text="大小", anchor="e")
        self.tree.column("#0", width=196, anchor="w")
        self.tree.column("q", width=62, anchor="w", stretch=False)
        self.tree.column("s", width=70, anchor="e", stretch=False)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_pick)
        self.tree.bind("<Double-1>", lambda e: self._start())

        self.lbl_model = tk.Label(left, text="未选择模型", bg=PANEL, fg=FG_DIM,
                                  font=("Microsoft YaHei UI", 8), anchor="w",
                                  justify="left", wraplength=315)
        self.lbl_model.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

        # ---- 中：参数
        mid = ttk.Frame(root, style="Panel.TFrame")
        mid.grid(row=1, column=1, sticky="nsew", padx=(0, 8))
        mid.rowconfigure(1, weight=1)
        mid.columnconfigure(0, weight=1)
        nh = ttk.Frame(mid, style="Panel.TFrame")
        nh.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        ttk.Label(nh, text="参数", style="Head.TLabel").pack(side="left")
        self.nb = ttk.Notebook(mid)
        self.nb.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        self.tab_base = ttk.Frame(self.nb, style="Panel.TFrame")
        self.tab_adv = ttk.Frame(self.nb, style="Panel.TFrame")
        self.tab_spec = ttk.Frame(self.nb, style="Panel.TFrame")
        self.tab_vision = ttk.Frame(self.nb, style="Panel.TFrame")
        self.nb.add(self.tab_base, text="基础")
        self.nb.add(self.tab_adv, text="高级")
        self.nb.add(self.tab_spec, text="投机")
        self.nb.add(self.tab_vision, text="视觉")

        self.v = {}
        self._build_base(self.tab_base)
        self._build_adv(self.tab_adv)
        self._build_spec(self.tab_spec)
        self._build_vision(self.tab_vision)

        # ---- 右：运行
        right = ttk.Frame(root, style="Panel.TFrame")
        right.grid(row=1, column=2, sticky="nsew")
        right.rowconfigure(4, weight=1)
        right.columnconfigure(0, weight=1)

        rh = ttk.Frame(right, style="Panel.TFrame")
        rh.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        self.dot = tk.Canvas(rh, width=10, height=10, bg=PANEL, highlightthickness=0)
        self.dot.pack(side="left")
        self._dot_id = self.dot.create_oval(1, 1, 9, 9, fill="#4a5162", outline="")
        self.lbl_state = tk.Label(rh, text="空闲", bg=PANEL, fg=FG_DIM, font=FONT_B)
        self.lbl_state.pack(side="left", padx=6)
        self.lbl_url = tk.Label(rh, text="", bg=PANEL, fg=ACCENT, font=("Consolas", 8))
        self.lbl_url.pack(side="left", padx=6)
        ttk.Button(rh, text="WebUI", width=6, command=self._open_web).pack(side="right")
        ttk.Button(rh, text="复制 API", width=8, command=self._copy_api).pack(side="right", padx=4)

        # 性能条
        perf = tk.Frame(right, bg=PANEL2, height=52)
        perf.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        perf.pack_propagate(False)
        self.perf_labels = {}
        for i, (key, text) in enumerate([("pp", "Prefill"), ("tg", "Decode"),
                                         ("load", "加载"), ("mem", "显存")]):
            cell = tk.Frame(perf, bg=PANEL2)
            cell.pack(side="left", expand=True, fill="both", padx=(10 if i == 0 else 4, 4), pady=6)
            tk.Label(cell, text=text, bg=PANEL2, fg=FG_DIM,
                     font=("Microsoft YaHei UI", 7), anchor="w").pack(anchor="w")
            lb = tk.Label(cell, text="—", bg=PANEL2, fg=FG, font=("Consolas", 10, "bold"), anchor="w")
            lb.pack(anchor="w")
            self.perf_labels[key] = lb

        # 按钮
        bar = ttk.Frame(right, style="Panel.TFrame")
        bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))
        self.btn_start = ttk.Button(bar, text="启动服务", style="Accent.TButton",
                                    command=self._start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="停止", style="Danger.TButton",
                                   width=8, command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(bar, text="导出 .bat", command=self._export_bat).pack(side="right")

        cmdwrap = tk.Frame(right, bg=PANEL)
        cmdwrap.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 6))
        self.txt_cmd = tk.Text(cmdwrap, height=2, bg="#0e1015", fg="#9fb4d0",
                               font=(MONO, 8), bd=0, wrap="word", padx=6, pady=4)
        self.txt_cmd.pack(fill="x")
        self.txt_cmd.configure(state="disabled")

        logwrap = tk.Frame(right, bg=PANEL)
        logwrap.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0, 6))
        logwrap.rowconfigure(1, weight=1)
        logwrap.columnconfigure(0, weight=1)
        lh = tk.Frame(logwrap, bg=PANEL)
        lh.grid(row=0, column=0, sticky="ew", pady=(0, 3))
        tk.Label(lh, text="日志", bg=PANEL, fg=FG_DIM,
                 font=("Microsoft YaHei UI", 8)).pack(side="left")
        self.var_autoscroll = tk.BooleanVar(value=True)
        tk.Checkbutton(lh, text="自动滚动", variable=self.var_autoscroll,
                       bg=PANEL, fg=FG_DIM, selectcolor=PANEL2, bd=0,
                       activebackground=PANEL, activeforeground=FG,
                       font=("Microsoft YaHei UI", 8)).pack(side="right")
        self.txt_log = tk.Text(logwrap, bg="#0e1015", fg="#c8d0e0", font=(MONO, 8),
                               bd=0, wrap="none", padx=6, pady=4)
        self.txt_log.grid(row=1, column=0, sticky="nsew")
        lsb = ttk.Scrollbar(logwrap, orient="vertical", command=self.txt_log.yview)
        lsb.grid(row=1, column=1, sticky="ns")
        self.txt_log.configure(yscrollcommand=lsb.set)
        self.txt_log.tag_configure("err", foreground=ERR)
        self.txt_log.tag_configure("warn", foreground=WARN)
        self.txt_log.tag_configure("ok", foreground=OK)
        self.txt_log.tag_configure("dim", foreground="#5c6478")

        # ---- 底：预设
        bottom = ttk.Frame(root, style="TFrame")
        bottom.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Label(bottom, text="预设", foreground=FG_DIM, background=BG,
                  font=("Microsoft YaHei UI", 8)).pack(side="left", padx=(2, 6))
        for name, key in [("P100 最优", "p100"), ("长上下文 32K", "long"),
                          ("极速小模型", "fast"), ("省显存 4bit KV", "lowmem")]:
            ttk.Button(bottom, text=name, width=13,
                       command=lambda k=key: self._apply_preset(k)).pack(side="left", padx=3)
        ttk.Button(bottom, text="保存当前为预设…", command=self._save_preset).pack(side="left", padx=(10, 3))
        self.lbl_hint = tk.Label(bottom, text="", bg=BG, fg=FG_DIM, font=("Microsoft YaHei UI", 8))
        self.lbl_hint.pack(side="right", padx=6)

    # ---------------------------------------------------------- 参数面板
    def _row(self, parent, r: int, label: str, widget, hint: str = "") -> None:
        tk.Label(parent, text=label, bg=PANEL, fg=FG, font=FONT, anchor="w"
                 ).grid(row=r, column=0, sticky="w", padx=(10, 6), pady=3)
        widget.grid(row=r, column=1, sticky="ew", padx=(0, 10), pady=3)
        if hint:
            tk.Label(parent, text=hint, bg=PANEL, fg=FG_DIM,
                     font=("Microsoft YaHei UI", 7), anchor="w"
                     ).grid(row=r + 1, column=1, sticky="w", padx=(0, 10), pady=(0, 3))
        parent.columnconfigure(1, weight=1)

    def _spin(self, key: str, frm: int, to: int, inc: int = 1, width: int = 9):
        var = tk.StringVar(value=str(self.cfg["params"].get(key, frm)))
        self.v[key] = var
        return ttk.Spinbox(self.nb_master, from_=frm, to=to, increment=inc,
                           textvariable=var, width=width)

    def _build_base(self, tab) -> None:
        tab.columnconfigure(1, weight=1)
        self.nb_master = tab
        r = 0
        self._row(tab, r, "上下文 -c", self._spin("ctx", 512, 262144, 512)); r += 2
        self._row(tab, r, "GPU 层数 -ngl", self._spin("ngl", 0, 999)); r += 2
        self._row(tab, r, "批大小 -b", self._spin("batch", 64, 16384, 64)); r += 2
        self._row(tab, r, "物理批 -ub", self._spin("ubatch", 32, 8192, 32),
                  "1024 与 512 持平（26.07 vs 26.03 t/s）"); r += 3
        self._row(tab, r, "CPU 线程 -t", self._spin("threads", 1, 32)); r += 2
        self._row(tab, r, "端口", self._spin("port", 1024, 65535, 1)); r += 2

        fa = tk.StringVar(value=self.cfg["params"].get("fa", "auto"))
        self.v["fa"] = fa
        self._row(tab, r, "Flash Attn -fa",
                  ttk.Combobox(tab, textvariable=fa, values=["auto", "on", "off"],
                               state="readonly"))
        r += 2
        ctk = tk.StringVar(value=self.cfg["params"].get("ctk", "f16"))
        ctv = tk.StringVar(value=self.cfg["params"].get("ctv", "f16"))
        self.v["ctk"], self.v["ctv"] = ctk, ctv
        kf = tk.Frame(tab, bg=PANEL)
        kf.grid(row=r, column=1, sticky="ew", padx=(0, 10), pady=3)
        ttk.Combobox(kf, textvariable=ctk, width=7, state="readonly",
                     values=["f16", "q8_0", "q4_0", "q5_1", "iq4_nl"]).pack(side="left")
        tk.Label(kf, text="K / V", bg=PANEL, fg=FG_DIM).pack(side="left", padx=4)
        ttk.Combobox(kf, textvariable=ctv, width=7, state="readonly",
                     values=["f16", "q8_0", "q4_0", "q5_1", "iq4_nl"]).pack(side="left")
        tk.Label(tab, text="KV 量化", bg=PANEL, fg=FG, anchor="w"
                 ).grid(row=r, column=0, sticky="w", padx=(10, 6), pady=3)
        tk.Label(tab, text="K/V 必须同类型，混用掉 37%。q8_0 反而慢 3%；\n"
                           "q4_0 实测最优：速度持平 f16、30 题质量完全一致。\n"
                           "ctx 上限（Q8_0 权重）：f16 128K / q8_0 192K / q4_0 256K。\n"
                           "换 Q4_K_M 权重后省 3.4 GB：f16 即可开 196K，且更快 12%。",
                 bg=PANEL, fg=FG_DIM, font=("Microsoft YaHei UI", 7),
                 justify="left", anchor="w"
                 ).grid(row=r + 1, column=1, sticky="w", padx=(0, 10), pady=(0, 3))
        r += 3
        jinja = tk.BooleanVar(value=self.cfg["params"].get("jinja", True))
        self.v["jinja"] = jinja
        ttk.Checkbutton(tab, text="启用 --jinja（推荐，工具调用/模板）",
                        variable=jinja).grid(row=r, column=0, columnspan=2,
                                             sticky="w", padx=10, pady=3)

    def _build_adv(self, tab) -> None:
        tab.columnconfigure(1, weight=1)
        r = 0
        lm = tk.StringVar(value=self.cfg["params"].get("load_mode", "none"))
        self.v["load_mode"] = lm
        self._row(tab, r, "加载模式 -lm",
                  ttk.Combobox(tab, textvariable=lm, state="readonly",
                               values=["auto", "mmap", "mlock", "mmap+mlock", "none", "dio"]),
                  "none：实测内存 6.9→1.1 GB；auto=mmap 会映射 5.9 GB 文件页"); r += 3
        fit = tk.BooleanVar(value=self.cfg["params"].get("fit", False))
        self.v["fit"] = fit
        ttk.Checkbutton(tab, text="-fit on  自动按显存调整上下文",
                        variable=fit).grid(row=r, column=0, columnspan=2,
                                           sticky="w", padx=10, pady=3)
        r += 1
        par = tk.StringVar(value=str(self.cfg["params"].get("parallel", 1)))
        self.v["parallel"] = par
        self._row(tab, r, "并发槽 -np", ttk.Spinbox(tab, from_=1, to=16, textvariable=par, width=9)); r += 2
        alias = tk.StringVar(value=self.cfg["params"].get("alias", ""))
        self.v["alias"] = alias
        self._row(tab, r, "模型别名 -a", ttk.Entry(tab, textvariable=alias)); r += 2
        key = tk.StringVar(value=self.cfg["params"].get("api_key", ""))
        self.v["api_key"] = key
        self._row(tab, r, "API Key", ttk.Entry(tab, textvariable=key), "留空则不鉴权"); r += 3
        met = tk.BooleanVar(value=self.cfg["params"].get("metrics", True))
        self.v["metrics"] = met
        ttk.Checkbutton(tab, text="开启 /metrics 端点", variable=met).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=10, pady=2)
        r += 1
        extra = tk.StringVar(value=self.cfg["params"].get("extra", ""))
        self.v["extra"] = extra
        self._row(tab, r, "额外参数", ttk.Entry(tab, textvariable=extra),
                  "原样追加，如 --cache-ram 4096")

    def _build_spec(self, tab) -> None:
        tab.columnconfigure(1, weight=1)
        r = 0
        en = tk.BooleanVar(value=self.cfg["params"].get("spec_enabled", False))
        self.v["spec_enabled"] = en
        ttk.Checkbutton(tab, text="启用投机解码（draft 模型加速 decode）",
                        variable=en).grid(row=r, column=0, columnspan=2,
                                          sticky="w", padx=10, pady=(8, 4))
        r += 1
        sm = tk.StringVar(value=self.cfg["params"].get("spec_model", ""))
        self.v["spec_model"] = sm
        brow = tk.Frame(tab, bg=PANEL)
        brow.grid(row=r, column=0, columnspan=2, sticky="ew", padx=10, pady=3)
        brow.columnconfigure(0, weight=1)
        ttk.Entry(brow, textvariable=sm).grid(row=0, column=0, sticky="ew")
        ttk.Button(brow, text="浏览", width=6,
                   command=self._pick_draft).grid(row=0, column=1, padx=(4, 0))
        tk.Label(tab, text="draft 模型 -md", bg=PANEL, fg=FG_DIM,
                 font=("Microsoft YaHei UI", 7), anchor="w"
                 ).grid(row=r + 1, column=0, sticky="w", padx=10)
        r += 2
        stp = tk.StringVar(value=self.cfg["params"].get("spec_type", "auto"))
        self.v["spec_type"] = stp
        self._row(tab, r, "投机类型", ttk.Combobox(
            tab, textvariable=stp, state="readonly",
            values=["auto", "draft-mtp", "draft-simple", "draft-eagle3",
                    "ngram-simple", "ngram-mod", "ngram-map-k", "draft-dflash"]),
            "auto：检测到 nextn 头才开 MTP") ; r += 3
        nmax = tk.StringVar(value=str(self.cfg["params"].get("spec_n_max", 2)))
        self.v["spec_n_max"] = nmax
        self._row(tab, r, "草稿长度", ttk.Spinbox(tab, from_=1, to=16, textvariable=nmax, width=9),
                  "越大潜在收益高，接受率低时反而慢"); r += 3
        ngld = tk.StringVar(value=str(self.cfg["params"].get("spec_ngl", 99)))
        self.v["spec_ngl"] = ngld
        self._row(tab, r, "draft 层数", ttk.Spinbox(tab, from_=0, to=999, textvariable=ngld, width=9))
        r += 2
        tk.Label(tab, text="MTP 最优：n_max=2，无需 -md。收益看内容：\n"
                           "数学 +59%／代码改写 +44%／论述 +36%／闲聊 +26%。\n"
                           "n_max≥3 必崩（51.0）——GP100 MMVQ 在 4 列换模板，\n"
                           "不是 cuBLAS 的问题：关掉 MMVQ 后 n3 反而略优于 n2。\n"
                           "DFlash：n2 打平 MTP 但多占 2.9 GB，n≥3 落后，n11 OOM。\n"
                           "EAGLE3：各 n_max 全输 MTP（最好 69.3），因为头训练在\n"
                           "Qwen3.5 base 上，与微调版 target 不同源，接受率仅 58%。\n"
                           "−bs 后端采样虽快 3.7%，但会在 code prompt 触发 CUDA\n"
                           "非法访存（且与 p_min 互斥），不要开。",
                 bg=PANEL, fg=FG_DIM, font=("Microsoft YaHei UI", 8),
                 justify="left").grid(row=r, column=0, columnspan=2,
                                      sticky="w", padx=10, pady=(8, 0))

    # ---------------------------------------------------------- 变量绑定
    def _build_vision(self, tab) -> None:
        tab.columnconfigure(1, weight=1)
        r = 0
        tk.Label(tab, text="视觉投影 (mmproj)", bg=PANEL, fg=FG, font=FONT_B,
                 anchor="w").grid(row=r, column=0, columnspan=2, sticky="w",
                                  padx=10, pady=(10, 2)); r += 1
        self.lbl_mtmd = tk.Label(tab, text="后端能力探测中…", bg=PANEL, fg=FG_DIM,
                                 font=("Microsoft YaHei UI", 8), anchor="w",
                                 wraplength=300, justify="left")
        self.lbl_mtmd.grid(row=r, column=0, columnspan=2, sticky="w", padx=10); r += 1

        ve = tk.BooleanVar(value=self.cfg["params"].get("vision_enabled", False))
        self.v["vision_enabled"] = ve
        ttk.Checkbutton(tab, text="启用视觉模块（默认关 · 实测 decode −18%）",
                        variable=ve).grid(row=r, column=0, columnspan=2,
                                          sticky="w", padx=10, pady=(10, 2)); r += 1
        tk.Label(tab, text="关闭时完全不加载 mmproj，命令行不含任何 --mmproj* 参数。",
                 bg=PANEL, fg=FG_DIM, font=("Microsoft YaHei UI", 7), anchor="w"
                 ).grid(row=r, column=0, columnspan=2, sticky="w", padx=10); r += 1

        mp = tk.StringVar(value=self.cfg["params"].get("mmproj", ""))
        self.v["mmproj"] = mp
        self._mmproj_values = core.list_known_mmproj()
        self.cb_mmproj = ttk.Combobox(tab, textvariable=mp,
                                      values=self._mmproj_values, width=28)
        self._row(tab, r, "投影 --mmproj", self.cb_mmproj,
                  "开启后留空 = 用同目录配对；跨目录必须手动选")
        r += 2
        self.btn_mmproj = ttk.Button(tab, text="浏览…", width=7,
                                     command=self._pick_mmproj)
        self.btn_mmproj.grid(row=r, column=1, sticky="w", padx=(0, 10), pady=2)
        r += 1

        au = tk.BooleanVar(value=self.cfg["params"].get("mmproj_auto", False))
        self.v["mmproj_auto"] = au
        self.chk_mmproj_auto = ttk.Checkbutton(
            tab, text="--mmproj-auto  交给后端自动匹配", variable=au)
        self.chk_mmproj_auto.grid(row=r, column=0, columnspan=2,
                                  sticky="w", padx=10, pady=3); r += 1
        tk.Label(tab, text="注意：auto 可能联网下载投影，本机默认关闭。",
                 bg=PANEL, fg=WARN, font=("Microsoft YaHei UI", 7), anchor="w"
                 ).grid(row=r, column=0, columnspan=2, sticky="w", padx=10); r += 1

        self.lbl_vision = tk.Label(tab, text="未选择模型", bg=PANEL, fg=FG_DIM,
                                   font=("Microsoft YaHei UI", 8), anchor="w",
                                   wraplength=300, justify="left")
        self.lbl_vision.grid(row=r, column=0, columnspan=2, sticky="w",
                             padx=10, pady=(12, 4))
        self._refresh_vision()

    def _refresh_vision(self) -> None:
        enabled = bool(self.cfg["params"].get("vision_enabled"))
        for w in (getattr(self, "cb_mmproj", None),
                  getattr(self, "btn_mmproj", None),
                  getattr(self, "chk_mmproj_auto", None)):
            if w is not None:
                try:
                    w.configure(state="normal" if enabled else "disabled")
                except Exception:
                    pass
        backs = getattr(self, "backends", [])
        caps: set = set()
        be = self.cfg.get("backend_path", "")
        for b in backs:
            if os.path.normcase(b.path) == os.path.normcase(be):
                caps = b.caps
                break
        if not backs:
            self.lbl_mtmd.configure(text="后端能力探测中…", fg=FG_DIM)
        else:
            ok = ("mmproj" in caps) or not caps
            suffix = " / --mmproj-auto" if "mmproj_auto" in caps else ""
            self.lbl_mtmd.configure(
                text=("已编译 mtmd：支持 --mmproj" + suffix) if ok
                     else "该后端未编译视觉模块，视觉参数会被自动跳过",
                fg=OK if ok else WARN)
        m = getattr(self, "selected", None)
        if not m:
            self.lbl_vision.configure(text="未选择模型", fg=FG_DIM)
            return
        if not enabled:
            self.lbl_vision.configure(
                text=(f"{m.name}\n视觉模块已关闭（默认）：不加载投影，"
                      "生成速度不受影响。需要看图再勾选上方开关。"), fg=FG_DIM)
            return
        manual = (self.cfg["params"].get("mmproj") or "").strip()
        if manual:
            self.lbl_vision.configure(text="手动投影：" + os.path.basename(manual), fg=OK)
        elif m.mmproj:
            self.lbl_vision.configure(text="已配对：" + os.path.basename(m.mmproj), fg=OK)
        else:
            self.lbl_vision.configure(
                text=(f"{m.name}\n同目录无 mmproj。该模型很可能是纯文本模型，"
                      "硬挂投影也不会产生视觉能力。"), fg=WARN)

    def _last_dir(self) -> str:
        """文件对话框的起始目录：优先当前模型目录，其次配置里的第一个模型根。"""
        if self.selected and self.selected.path:
            d = os.path.dirname(self.selected.path)
            if os.path.isdir(d):
                return d
        roots = self.cfg.get("model_roots") or core.DEFAULT_ROOTS
        for r in roots:
            if r and os.path.isdir(r):
                return r
        return os.path.expanduser("~")

    def _pick_mmproj(self) -> None:
        p = filedialog.askopenfilename(
            title="选择 mmproj 视觉投影", initialdir=self._last_dir(),
            filetypes=[("mmproj GGUF", "*mmproj*.gguf"), ("GGUF", "*.gguf")])
        if p:
            self.v["mmproj"].set(p)
            if p not in self._mmproj_values:
                self._mmproj_values.append(p)
                self.cb_mmproj.configure(values=self._mmproj_values)

    def _bind_vars(self) -> None:
        for k, var in self.v.items():
            var.trace_add("write", lambda *_a: self._on_param_change())

    def _sync_cfg_from_ui(self) -> None:
        p = self.cfg["params"]
        for k, var in self.v.items():
            if k in ("jinja", "fit", "metrics", "spec_enabled", "mmproj_auto", "vision_enabled"):
                p[k] = bool(var.get())
            elif k in ("port", "ctx", "ngl", "batch", "ubatch", "threads",
                       "parallel", "spec_n_max", "spec_ngl"):
                try:
                    p[k] = int(var.get())
                except Exception:
                    pass
            else:
                p[k] = var.get()

    def _sync_ui_from_cfg(self) -> None:
        p = self.cfg["params"]
        self._suppress = True
        try:
            for k, var in self.v.items():
                if k not in p:
                    continue
                val = p[k]
                try:
                    if k in ("jinja", "fit", "metrics", "spec_enabled", "mmproj_auto", "vision_enabled"):
                        var.set(bool(val))
                    else:
                        var.set(str(val))
                except Exception:
                    pass
        finally:
            self._suppress = False

    def _on_param_change(self) -> None:
        if getattr(self, "_suppress", False):
            return
        self._sync_cfg_from_ui()
        self._update_cmd_preview()
        self._refresh_vision()

    def _update_cmd_preview(self) -> None:
        be = self.cfg.get("backend_path", "")
        caps = set()
        for b in self.backends:
            if os.path.normcase(b.path) == os.path.normcase(be):
                caps = b.caps
                break
        args = core.build_args(self.cfg, self.selected, caps)
        cmd = "llama-server.exe " + core.quote_cmd(args)
        self.txt_cmd.configure(state="normal")
        self.txt_cmd.delete("1.0", "end")
        self.txt_cmd.insert("1.0", cmd)
        self.txt_cmd.configure(state="disabled")

    # ---------------------------------------------------------- 启动流程
    def _boot(self) -> None:
        backends = core.discover_backends()
        self.q.put(("backends", backends))
        models = core.scan_models(self.cfg.get("model_roots", core.DEFAULT_ROOTS),
                                  progress=lambda m: self.q.put(("hint", m)))
        core.save_model_cache(models)
        self.q.put(("models", models))

    def _rescan(self) -> None:
        self.lbl_hint.configure(text="扫描中…")
        threading.Thread(target=self._boot, daemon=True).start()

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "line":
                    self._append_log(payload)
                elif kind == "state":
                    self._set_state(*payload)
                elif kind == "models":
                    self.models = payload
                    self._fill_models()
                elif kind == "backends":
                    self._fill_backends(payload)
                elif kind == "hint":
                    self.lbl_hint.configure(text=payload)
                elif kind == "gpu":
                    self.gpu = payload
                    self._apply_gpu()
        except queue.Empty:
            pass
        self._refresh_perf()
        self.after(80, self._drain)

    def _fill_backends(self, backends: list[core.Backend]) -> None:
        self.backends = backends
        labels = [f"{b.label}  ·  {os.path.basename(b.path)}" for b in backends]
        self.cb_backend["values"] = labels
        if not backends:
            self.lbl_ver.configure(text="未找到 llama-server")
            self._refresh_vision()
            return
        cur = os.path.normcase(self.cfg.get("backend_path", ""))
        idx = 0
        for i, b in enumerate(backends):
            if os.path.normcase(b.path) == cur:
                idx = i
        self.cb_backend.current(idx)
        self.cfg["backend_path"] = backends[idx].path
        b = backends[idx]
        self.lbl_ver.configure(text=self._ver_text(b))
        self._update_cmd_preview()
        self._refresh_vision()
        # 后端探测完成、列表就绪后再按当前模型自动路由一次
        self._apply_auto_backend()

    @staticmethod
    def _ver_text(b) -> str:
        ver = (b.version or "ver ?").split("(")[0].strip()
        return f"{ver} · {len(b.caps)}/{len(core.CAP_FLAGS)} 项新特性"

    def _on_backend_pick(self, _evt=None) -> None:
        i = self.cb_backend.current()
        if 0 <= i < len(self.backends):
            b = self.backends[i]
            self.cfg["backend_path"] = b.path
            self.lbl_ver.configure(text=self._ver_text(b))
            self._update_cmd_preview()
            self._refresh_vision()

    def _on_auto_be_toggle(self) -> None:
        self.cfg["auto_backend"] = bool(self.var_auto_be.get())
        if self.cfg["auto_backend"]:
            self._apply_auto_backend()
        self._append_log("[后端] 自动选择" + ("已开启，按模型切到最快后端" if self.cfg["auto_backend"] else "已关闭，使用手动选择"), "dim")

    def _apply_auto_backend(self) -> None:
        """按当前模型自动选最快后端 + KV 量化；用户关掉「自动」则完全手动。"""
        if not getattr(self, "var_auto_be", None) or not self.var_auto_be.get():
            return
        m = getattr(self, "selected", None)
        changed = []

        # --- 后端 ---
        if getattr(self, "backends", None):
            rec = core.recommend_backend(m)
            if rec:
                paths = [os.path.normcase(b.path) for b in self.backends]
                try:
                    idx = paths.index(os.path.normcase(rec))
                    if (self.cb_backend.current() != idx or
                            os.path.normcase(self.cfg.get("backend_path", "")) != os.path.normcase(rec)):
                        self.cb_backend.current(idx)
                        self.cfg["backend_path"] = self.backends[idx].path
                        self.lbl_ver.configure(text=self._ver_text(self.backends[idx]))
                        changed.append("后端→" + self.backends[idx].label)
                except ValueError:
                    pass

        # --- KV cache 量化（实测 q8_0 拖慢 35B 15%，f16 最快）---
        kv = core.recommend_kv(m)
        if kv and "ctk" in self.v and "ctv" in self.v:
            if self.v["ctk"].get() != kv or self.v["ctv"].get() != kv:
                self.v["ctk"].set(kv)
                self.v["ctv"].set(kv)
                changed.append(f"KV→{kv}")

        if changed:
            self._sync_cfg_from_ui()
            self._update_cmd_preview()
            self._refresh_vision()
            self._append_log("[自动优化] " + "  ".join(changed), "dim")

    def _fill_models(self) -> None:
        kw = self.var_search.get().strip().lower()
        self.tree.delete(*self.tree.get_children())
        # 同名模型（同一份量化散落在两个目录）在列表里无法区分。目录前缀放最前，
        # 这样即使列宽把尾部截掉，前缀依然可见；单份的模型不加前缀，保持界面干净。
        dup: dict[str, int] = {}
        for m in self.models:
            dup[m.name] = dup.get(m.name, 0) + 1
        n = 0
        for m in self.models:
            if kw and kw not in m.name.lower() and kw not in m.path.lower():
                continue
            tag = "mtp" if m.mtp else ("vis" if m.vision else "")
            title = m.name
            if dup.get(m.name, 0) > 1:
                parent = os.path.basename(os.path.dirname(m.path))
                title = f"[{parent}] {m.name}"
            self.tree.insert("", "end", iid=m.path, text=title,
                             values=(m.quant or "—", m.size_h), tags=(tag,))
            n += 1
        self.tree.tag_configure("mtp", foreground="#c79bff")
        self.tree.tag_configure("vis", foreground="#8fd3ff")
        self.lbl_count.configure(text=f"{n} 个")
        if self.cfg.get("last_model") and self.tree.exists(self.cfg["last_model"]):
            self.tree.selection_set(self.cfg["last_model"])
            self.tree.see(self.cfg["last_model"])

    def _on_pick(self, _evt=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        path = sel[0]
        self.selected = next((m for m in self.models if m.path == path), None)
        if not self.selected:
            return
        m = self.selected
        self.cfg["last_model"] = m.path
        flags = []
        if m.mtp:
            flags.append("MTP")
        if m.vision:
            flags.append("视觉")
        extra = ("  ·  " + " / ".join(flags)) if flags else ""
        head = f"{m.size_h}  ·  {m.quant or '?'}{extra}"
        self.lbl_model.configure(text=f"{head}\n{m.path}", fg=FG_DIM)
        self._apply_auto_backend()
        self._update_cmd_preview()
        self._refresh_vision()

    # ---------------------------------------------------------- 运行控制
    def _start(self) -> None:
        if self.server.is_running():
            return
        if not self.selected:
            messagebox.showinfo("选择模型", "请先在左侧选择一个 .gguf 模型")
            return
        self._sync_cfg_from_ui()
        port = self.cfg["params"].get("port", 8080)
        if not core.port_free(port):
            if not messagebox.askyesno("端口占用",
                                       f"端口 {port} 已被占用，仍要尝试启动吗？"):
                return
        core.save_config(self.cfg)
        self._append_log(f"$ {self.txt_cmd.get('1.0', 'end').strip()}", "dim")
        ok = self.server.start(self.cfg, self.selected)
        if ok:
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self._remember_recent(self.selected.path)

    def _stop(self) -> None:
        self.server.stop()
        self._set_state("stopped", "已手动停止")
        self.btn_stop.configure(state="disabled")
        self.btn_start.configure(state="normal")
        self.lbl_url.configure(text="")

    def _remember_recent(self, path: str) -> None:
        rec = [p for p in self.cfg.get("recent", []) if p != path]
        rec.insert(0, path)
        self.cfg["recent"] = rec[:8]
        core.save_config(self.cfg)

    def _on_line(self, line: str) -> None:
        self.q.put(("line", line))

    def _on_state(self, state: str, msg: str) -> None:
        self.q.put(("state", (state, msg)))

    def _append_log(self, line: str, tag: str = "") -> None:
        if not tag:
            low = line.lower()
            if " error" in low or low.startswith("error") or "failed" in low:
                tag = "err"
            elif "warning" in low or "warn" in low:
                tag = "warn"
            elif "listening" in low or "all slots are idle" in low:
                tag = "ok"
        self.txt_log.insert("end", line + "\n", tag)
        if int(self.txt_log.index("end-1c").split(".")[0]) > 4000:
            self.txt_log.delete("1.0", "800.0")
        if self.var_autoscroll.get():
            self.txt_log.see("end")

    def _set_state(self, state: str, msg: str = "") -> None:
        # 忽略迟到的 ready 事件：进程已经停了就别把灯点回绿色
        if state in ("ready", "warn") and not self.server.is_running():
            return
        self.state = state
        color = {"loading": WARN, "ready": OK, "error": ERR,
                 "stopped": FG_DIM, "warn": WARN}.get(state, FG_DIM)
        text = {"loading": "加载中", "ready": "运行中", "error": "错误",
                "stopped": "已停止", "warn": "运行中"}.get(state, state)
        self.dot.itemconfigure(self._dot_id, fill=color)
        self.lbl_state.configure(text=text, fg=color)
        if state in ("ready", "warn"):
            self.lbl_url.configure(text=msg or self.server.listen_url)
            self.btn_stop.configure(state="normal")
            self.btn_start.configure(state="disabled")
        elif state in ("stopped", "error"):
            self.lbl_url.configure(text="")
            self.btn_stop.configure(state="disabled")
            self.btn_start.configure(state="normal")
        if msg and state in ("error", "warn"):
            self._append_log(msg, "err" if state == "error" else "warn")

    def _refresh_perf(self) -> None:
        p = self.server.perf
        self.perf_labels["pp"].configure(
            text=f"{p.pp_tps:.0f} t/s" if p.pp_tps else "—")
        self.perf_labels["tg"].configure(
            text=f"{p.tg_tps:.1f} t/s" if p.tg_tps else "—")
        self.perf_labels["load"].configure(
            text=f"{p.load_ms / 1000:.1f}s" if p.load_ms else "—")
        g = self.gpu
        if g.get("ok"):
            self.perf_labels["mem"].configure(
                text=f"{int(g['mem_used']) / 1024:.1f}G")

    def _tick_gpu(self) -> None:
        def work():
            g = core.gpu_stats()
            self.gpu = g
            self.q.put(("gpu", g))
        threading.Thread(target=work, daemon=True).start()
        self.after(2500, self._tick_gpu)

    def _apply_gpu(self) -> None:
        g = self.gpu
        if g.get("ok"):
            self.lbl_gpu.configure(
                text=f"{g['name']}   {g['util']}%   {int(g['mem_used'])}/{int(g['mem_total'])} MiB"
                     f"   {g['temp']}°C   {int(float(g['power']))}W")
        else:
            self.lbl_gpu.configure(text="GPU 不可用")

    # ---------------------------------------------------------- 预设
    PRESETS = {
        # 2026-11-08 实测（9B Q8_0 MTP n2, P100）：
        #   KV f16 67.2 / q8_0 65.0 / q4_0 67.3（q8_0 反而最慢）
        #   ctx 上限：f16 128K / q8_0 192K / q4_0 256K
        "p100": {"ctx": 8192, "ngl": 99, "batch": 2048, "ubatch": 1024,
                 "fa": "on", "ctk": "f16", "ctv": "f16", "threads": 6,
                 "load_mode": "none", "fit": False, "jinja": True, "metrics": True},
        "long": {"ctx": 65536, "ngl": 99, "batch": 2048, "ubatch": 1024,
                 "fa": "on", "ctk": "q4_0", "ctv": "q4_0", "threads": 6,
                 "load_mode": "none", "fit": False, "jinja": True, "metrics": True},
        "fast": {"ctx": 4096, "ngl": 99, "batch": 1024, "ubatch": 512,
                 "fa": "on", "ctk": "f16", "ctv": "f16", "threads": 6,
                 "load_mode": "none", "fit": True, "jinja": True, "metrics": True},
        "lowmem": {"ctx": 16384, "ngl": 99, "batch": 2048, "ubatch": 512,
                   "fa": "on", "ctk": "q4_0", "ctv": "q4_0", "threads": 6,
                   "load_mode": "none", "fit": True, "jinja": True, "metrics": True},
    }

    def _apply_preset(self, key: str) -> None:
        data = self.PRESETS.get(key) or self.cfg.get("presets", {}).get(key)
        if not data:
            return
        self.cfg["params"].update(data)
        self._sync_ui_from_cfg()
        self._sync_cfg_from_ui()
        self._update_cmd_preview()
        fa_now = self.cfg["params"].get("fa", "?")
        self.lbl_hint.configure(text=f"已应用预设：{key}  (fa={fa_now})")

    def _save_preset(self) -> None:
        self._sync_cfg_from_ui()
        name = simpledialog.askstring("保存预设", "预设名称：", parent=self)
        if not name:
            return
        self.cfg.setdefault("presets", {})[name] = dict(self.cfg["params"])
        core.save_config(self.cfg)
        self.lbl_hint.configure(text=f"预设 {name} 已保存")

    # ---------------------------------------------------------- 杂项
    def _pick_draft(self) -> None:
        p = filedialog.askopenfilename(
            title="选择 draft 模型", initialdir=self._last_dir(),
            filetypes=[("GGUF", "*.gguf")])
        if p:
            self.v["spec_model"].set(p)

    def _export_bat(self) -> None:
        self._sync_cfg_from_ui()
        if not self.selected:
            messagebox.showinfo("选择模型", "请先选择模型")
            return
        path = filedialog.asksaveasfilename(
            title="导出启动脚本", defaultextension=".bat",
            initialfile=f"启动-{self.selected.name[:20]}.bat",
            filetypes=[("批处理", "*.bat")])
        if not path:
            return
        core.write_bat(self.cfg, self.selected, path)
        self.lbl_hint.configure(text=f"已导出 {os.path.basename(path)}")

    def _open_web(self) -> None:
        url = self.server.listen_url or \
            f"http://{self.cfg['params']['host']}:{self.cfg['params']['port']}"
        if url and not url.startswith("http"):
            url = "http://" + url
        os.startfile(url)

    def _copy_api(self) -> None:
        host = self.cfg["params"]["host"]
        if host in ("0.0.0.0", ""):
            host = "127.0.0.1"
        url = f"http://{host}:{self.cfg['params']['port']}/v1"
        self.clipboard_clear()
        self.clipboard_append(url)
        self.lbl_hint.configure(text=f"已复制 {url}")

    def _on_close(self) -> None:
        try:
            self._sync_cfg_from_ui()
            core.save_config(self.cfg)
            if self.server.is_running():
                if not messagebox.askyesno("退出", "服务正在运行，停止并退出？"):
                    return
                self.server.stop()
        finally:
            self.destroy()


def _activate_existing() -> bool:
    """已有实例时把它的窗口提到前台，返回 True。"""
    import ctypes
    from ctypes import wintypes as wt
    found = []

    def cb(hwnd, _):
        n = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if n:
            b = ctypes.create_unicode_buffer(n + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, b, n + 1)
            if b.value == "LlamaCpp 启动器":
                found.append(hwnd)
        return True

    CB = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(cb)
    ctypes.windll.user32.EnumWindows(CB, 0)
    if found:
        ctypes.windll.user32.ShowWindow(found[0], 9)
        ctypes.windll.user32.SetForegroundWindow(found[0])
        return True
    return False


def main() -> None:
    import socket
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", 8788))
        lock.listen(1)
    except OSError:
        if _activate_existing():
            return
    if sys.platform == "win32":
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
