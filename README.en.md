# llama.cpp multi-backend launcher (tuned on Tesla P100)

A local `llama-server` launcher with **zero third-party dependencies** (Python standard library + Tk only). Built for machines with several coexisting llama.cpp backends, automatic per-architecture routing, and a focus on measured throughput. Development hardware: Tesla P100 16GB + Ryzen 5 1600X, Windows.

## Features

- **Automatic backend routing** — picks the fastest `llama-server` by model name/architecture (P100 fork / upstream master / Pascal / PXA). Only backends that actually exist on this machine are recommended.
- **Capability-based argument building** — probes each backend's `--help` so switching backends never emits an unsupported flag; directories missing CUDA DLLs are skipped.
- **Benchmark-driven defaults** — `-fa on`, `-ctk/-ctv f16`, speculative decoding off by default (no measured gain on this hardware; see `docs/`).
- Model/mmproj scanning and caching, presets, batch-script export, port checks, GPU status, and log parsing.
- **Benchmark scripts** for DFlash / EAGLE3 / MTP / ngram / quantization quality comparisons.

## Quick start

```bash
# 1. Configure local paths (private paths are never committed)
copy paths.example.json paths.local.json
# edit paths.local.json with your backend and model directories

# 2. Launch (needs Python 3 with tkinter)
python launcher.py

# 3. Run the self-test
python selftest.py
```

In `paths.local.json`, each `backends[].role` is one of `p100` / `inkling` / `pascal` / `pxa`, used by automatic routing. Manual backend selection still works without this file.

## Measured results (Tesla P100 16GB)

| Item | Result |
|---|---|
| P100 fork vs older backend, 35B-A3B prefill | 568 vs 459 t/s (+24%) |
| Same, decode | 79.2 vs 77.8 t/s (+1.9%) |
| `-fa on` vs auto | prefill +1.9%, decode unchanged -> adopted |
| `-ctk/-ctv q8_0` | decode -15% -> prefer f16 when VRAM allows |
| Speculative decoding (draft-simple / ngram) | no gain or slower -> off by default |
| PXA + PXQN quantization | 27B decode +52%, but standard GGUF -42% -> PXA only for PXQ models |

Full experiment notes are in [`docs/`](docs/) (two rounds of MTP research, DFlash report, generation-speed survey).

## Layout

- `launcher.py` / `llmcore.py` — GUI and core logic
- `selftest.py` — self-test without manual clicking
- `tools/` — GGUF metadata parser, MTP head probe, resumable HF downloader, VRAM check
- `benchmarks/` — throughput and quality comparison scripts (model paths are placeholders)
- `docs/` — measured reports and technical notes (Chinese)

## License

MIT — see [LICENSE](LICENSE).
