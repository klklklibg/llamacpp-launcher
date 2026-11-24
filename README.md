# llama.cpp 多后端启动器（P100 实测调优）

一个零第三方依赖（仅 Python 标准库 + Tk）的本地 `llama-server` 启动器。面向多后端共存、需要按模型架构自动选最优后端、并且关心实测吞吐的场景（本机为 Tesla P100 16GB + Ryzen 5 1600X）。

## 功能

- **多后端自动路由**：按模型文件名/架构自动选择最快的 `llama-server`（P100 fork / upstream master / Pascal / PXA），只推荐本机实际存在的后端。
- **参数按能力拼装**：探测每个后端 `--help` 支持的参数，换后端不会拼出无效 flag；缺 CUDA DLL 的目录自动跳过。
- **实测驱动的默认参数**：`-fa on`、`-ctk/-ctv f16`、投机解码默认关闭（本机实测无收益，见文档）。
- **模型/投影扫描、缓存、预设、批处理脚本导出、端口检测、GPU 状态、日志解析**。
- **配套投机解码实测脚本**：DFlash / EAGLE3 / MTP / ngram / 量化质量对比。

## 快速开始

```bash
# 1. 配置本机路径（仓库不提交私人路径）
copy paths.example.json paths.local.json
# 编辑 paths.local.json，填你的后端与模型目录

# 2. 启动（需要 Python 3 带 tkinter）
python launcher.py

# 3. 自检
python selftest.py
```

`paths.local.json` 里的 `backends[].role` 取 `p100` / `inkling` / `pascal` / `pxa`，供自动路由使用。不配置也能手动选后端。

## 实测数据（Tesla P100 16GB）

| 项 | 结论 |
|---|---|
| P100 fork vs 旧后端，35B-A3B prefill | 568 vs 459 t/s（+24%） |
| 同上 decode | 79.2 vs 77.8 t/s（+1.9%） |
| `-fa on` vs auto | prefill +1.9%，decode 持平 → 采用 |
| `-ctk/-ctv q8_0` | decode -15% → 显存不紧张时用 f16 |
| 投机解码（draft-simple / ngram） | 无收益甚至更慢 → 默认关闭 |
| PXA + PXQN 量化 | 27B decode +52%；但标准 GGUF -42% → 只对 PXQ 模型用 |

完整实验记录见 [`docs/`](docs/)：MTP 两轮研究、DFlash 实测报告、生成速度技术盘点。

## 目录

- `launcher.py` / `llmcore.py`：界面 + 核心逻辑
- `selftest.py`：无人工点击的逻辑自检
- `tools/`：GGUF metadata 解析、MTP 头探测、HF 断点续传下载、显存检查
- `benchmarks/`：各类吞吐/质量对照脚本（模型路径用占位符，运行时填）
- `docs/`：实测报告与技术笔记

## 许可

MIT，见 [LICENSE](LICENSE)。
