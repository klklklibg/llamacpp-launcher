# LlamaCPP 启动器

2026-11-06 部署。目标机器：Tesla P100-PCIE-16GB（TCC）+ Ryzen 5 1600X + 15.9 GB RAM。

## 1. 入口

| 方式 | 路径 |
|---|---|
| 桌面快捷方式 | `C:\Users\xiao\Desktop\LlamaCPP 启动器.lnk` |
| 批处理 | `D:\llamacpp-launcher\启动 LlamaCPP.bat` |
| 直接运行 | `D:\Python312\pythonw.exe D:\llamacpp-launcher\launcher.py` |

零第三方依赖（仅标准库 + 系统自带 Tk）。用 `D:\Python312` 而不是 PATH 里的
Python 3.13，因为后者没有装 tkinter。

单实例保护：重复双击只会把已有窗口提到前台。

## 2. 后端

默认 `D:\llama.cpp\bin`：

```
version: 0.3.0-dev
git: 4423f2a  cuda(pascal): make the half2 dp4a emulation opt-in, restoring MMQ prefill
CUDA arch: sm_60   GGML_CUDA_FA=ON   GGML_CUDA_GRAPHS=ON   Release
```

这是本机自编译的 Pascal 优化分支（PTQ1_0 三元量化 + LDG.128 访存 + MMQ 调优），
启动器会自动探测每个后端目录的 `--help`，按实际支持的参数拼命令行，所以换回
`bin-pascal-dp4a` / `bin-fp16fix` / Vulkan 备用构建也不会拼出无效参数。

探测时会先检查 `cublas64_12.dll` / `cudart64_12.dll` / `ggml-cuda.dll` 是否齐全。
`D:\llama.cpp-prism\bin` 缺 cublas，已被自动排除——这类缺 DLL 的目录直接跑会
弹系统错误框，现在连尝试都不会尝试。

## 3. 实测优化矩阵

模型：`Ternary-Bonsai-2-27B-PTQ1_0-2bit`（6.70 GiB），`llama-bench -p 512 -n 64 -r 3`，
P100 TCC，`-ngl 99`，`-t 6`。

| 配置 | pp512 (t/s) | tg64 (t/s) | 结论 |
|---|---:|---:|---|
| `-fa auto`（默认） | 230.28 | 26.03 | — |
| `-fa on` | **234.71** | 26.01 | prefill +1.9%，不损 decode → **采用** |
| `-fa on -ub 1024` | 234.71 | **26.07** | ubatch 与 512 持平，保留 1024 |
| `-fa on -ub 1024 -ctk q8_0 -ctv q8_0` | 233.64 | 23.40 | **decode −10%**，只在爆显存时用 |
| `-fa off` | 225.26 | 25.59 | 全面落后 |

对话级验证（27B，128 token，同一提示）：

| 配置 | 端到端 (t/s) | 说明 |
|---|---:|---|
| 基线 | 19.43 | 26.01 t/s 纯 decode |
| `draft-simple` + Qwen3.5-0.8B | **6.47** | 慢 3 倍，draft 自身成了瓶颈 |
| `ngram-simple` | 16.98 | 无收益 |
| `ngram-mod` | 19.53 | 与基线持平 |

**结论：投机解码在这台机器上不划算，默认关闭。** 启动器「投机」页保留全部开关，
换更快的主模型（或更强的 draft）时可以直接试，实测数据也写在页面上。

### 加载模式：`-lm none`（默认）

`-ngl 99` 把模型全放进显存后，默认的 mmap 仍会把整个 gguf 作为文件页映射进进程地址空间。
实测（27B，6.70 GiB，同一台机器）：

| `--load-mode` | 进程工作集 | 其中文件映射 | 系统可用 | 加载耗时 |
|---|---:|---:|---:|---:|
| `auto`（= mmap，旧默认） | 6.86 GB | 5.85 GB | 5.07 GB | 10 s |
| **`none`（现默认）** | **0.82 GB** | 0 | **11.07 GB** | **4.5 s** |
| `dio` | 0.79 GB | 0 | 11.10 GB | 5 s |
| `auto` + `--no-host` | 6.86 GB | 5.85 GB | 5.01 GB | 6 s |

- 那 6 GB 不是模型常驻内存，而是 mmap 文件页落在进程工作集里的映射。`none`/`dio` 不建立映射，
  读入临时缓冲、上传显存后释放。
- `--no-host` 对此**完全无效**，别指望它。
- `none` 反而比 mmap 加载更快：mmap 按需缺页，6.7 GB 要触发约 170 万次缺页中断。
- 推理后会从 0.82 涨到 1.10 GB 并稳定（compute / logits 缓冲），不随生成 token 数增长。
- **不要用 `mlock`**，它反向钉住全部内存。
- 该参数只对支持 `-lm` 的后端下发（旧后端传了会直接退出），由能力探测自动把关。

自查内存：`python tools/memcheck.py`，直接打印进程 WS / 私有 / 映射量与系统可用内存。

## 4. 预设

| 预设 | ctx | ub | FA | KV | 用途 |
|---|---:|---:|---|---|---|
| P100 最优 | 8192 | 1024 | on | f16 | 默认，实测最快 |
| 长上下文 32K | 32768 | 1024 | on | q8_0 | 省 KV 显存，接受 decode 掉速 |
| 极速小模型 | 4096 | 512 | on | f16 | 4B 级模型，配合 `-fit` |
| 省显存 4bit KV | 16384 | 512 | on | q4_0 | 大模型挤爆 16 GB 时 |

## 5. 功能

- 模型扫描：`D:\lmstudio`、`D:\models`、`D:\laya`、`D:\llama.cpp\models` 等，缓存到
  `models.cache.json`，启动秒开，点「刷新」重扫。
- 参数：ctx / ngl / b / ub / threads / port / FA / KV 类型 / load-mode / fit / 并发槽 /
  别名 / API Key / /metrics / 额外参数。
- 运行：状态灯、URL、Prefill/Decode/加载耗时/显存四个实时指标、彩色日志、
  WebUI 按钮、复制 OpenAI 兼容地址。
- 导出 `.bat`：把当前配置固化成单文件启动脚本（CRLF + UTF-8）。
- 视觉页：**默认关闭**。实测挂载 mmproj 后 decode 掉约 18%（GPU 侧用户实测），因此
  扫描到的同目录 mmproj 只作为候选清单，绝不自动进命令行；必须勾选「启用视觉模块」
  才会下发 `--mmproj` / `--mmproj-auto`。关闭时投影下拉与 auto 开关一并置灰。
  页内可看后端 mtmd 编译状态、手动指定跨目录投影；后端不支持时自动跳过，不拼无效参数。
- 配置落在 `config.json`，日志落在 `server.log`。

## 6. 已知限制

1. P100 处于 TCC 模式，不输出画面，全部显示走 RX590。
2. KV 量化（q8_0/q4_0）在 Pascal 上走的是非优化路径，decode 掉约 10%。
3. `-fa` 的收益只在 prefill 端（+1.9%），decode 基本不变——P100 没有 Tensor Core，
   FA 省的是中间显存和带宽，不是算力。
4. 端口默认 8080。启动器会检测占用并提示，不会静默失败。
5. **视觉能力由 mmproj 决定，与后端无关。** 后端 `D:\llama.cpp\bin` 已编译 mtmd
   （`--help` 实测含 `--mmproj` / `--mmproj-auto` / `--mtmd-batch-max-tokens`），
   本机有 6 个 mmproj 文件，但当前默认的 `Ornith-1.5-35B-A3B-GSQ-RCO-3bit.gguf`
   GGUF 架构是 `qwen35moe`、44 个 metadata 中视觉相关 0 个、同目录只有这一个文件——
   **它是纯文本 MoE，挂投影也不会有视觉**。要有视觉需换成已配对的 4 个模型之一：
   `Qwen3.5-0.8B-Q8_0` / `Qwen3.5-9B-Q4_K_M` / `Ornith-1.5-9B-MTP-NVFP4` /
   `Qwen3.8-27B-GSQ-RCO-IQ3_S`。`'视觉'` 页会直接显示当前模型是否已配对。
6. `--mmproj-auto` 可能联网下载投影，默认关闭；跨目录的投影（如
   `D:\llama.cpp-prism\models\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`）需在视觉页手动选，
   刻意不按文件名相似度自动配对，避免把 0.8B 配到 9B 的 merger 上。
7. **视觉模块的性能代价有条件，不是每 token 的固有开销。** 用户 GPU 侧实测挂载
   mmproj 后 decode −18%，故默认关闭。但本机另做了一组纯 CPU A/B（0.8B Q8_0，
   同一提示 / 64 token / 各 3 次取中位，只改 `--mmproj` 一个变量）：
   不加载 10.92 t/s，加载 11.13 t/s，**+1.9%（在噪声内，未复现掉速）**。
   两者不矛盾：CPU 路径下 mmproj 不占显存；GPU 侧更可能是投影占显存后挤压
   KV/offload 或改变 CUDA graph 所致。8095 占用 15.48 GiB、仅剩 794 MiB，
   本机无法在不中断服务的前提下复现 GPU 侧那组数字，故 18% 按用户实测采信、
   不视为普适值。
8. 即使启用视觉，Qwen-VL 类模型默认只有 ~146 image tokens，工程链路通但语义在瞎猜
   （红圆 / 绿圆 / 无图三组均答 `red`）。加 `--image-min-tokens 1024` 后变为
   `red` / `green` / `black`，可区分，但纯 CPU 单请求从 13.4 s 涨到 127 s。
   启动器暂未把该参数做成开关，需要时在「额外参数」里手填 `--image-min-tokens 1024`。

## 7. 文件

| 文件 | 说明 |
|---|---|
| `launcher.py` | 界面（Tk），约 850 行 |
| `llmcore.py` | 后端探测 / 模型扫描 / 命令行构建 / 进程与日志解析 |
| `assets/llama.ico` | 图标 |
| `bench_matrix.sh` / `bench_matrix.log` | 参数矩阵基准原始数据 |
| `bench_spec.py` / `bench_ngram.py` | 投机解码对比脚本 |
| `tools/memcheck.py` | 进程 / 系统内存自查（WS、私有、文件映射） |
| `启动 LlamaCPP.bat` | 双击入口 |
| `selftest.py` | 逻辑自检（后端探测 / 命令行拼装 / 视觉默认关闭与显式开启 / 预设同步 / 日志解析，35 项断言） |
| `screenshot.png` | 界面截图 |
