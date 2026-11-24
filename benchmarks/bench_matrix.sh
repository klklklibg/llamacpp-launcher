#!/bin/bash
BIN=/d/llama.cpp/bin/llama-bench.exe
M="<FILL_MODEL_PATH>"
LOG=/d/llamacpp-launcher/bench_matrix.log
: > "$LOG"
run() {
  local tag="$1"; shift
  echo "### $tag" >> "$LOG"
  "$BIN" -m "$M" -ngl 99 -p 512 -n 64 -r 3 "$@" 2>&1 | grep -E "^\|" >> "$LOG"
  echo "" >> "$LOG"
}
run "A_fa_auto_default" -fa auto
run "B_fa_on" -fa on
run "C_fa_on_ub1024" -fa on -ub 1024
run "D_fa_on_ub1024_kvq8" -fa on -ub 1024 -ctk q8_0 -ctv q8_0
run "E_fa_off" -fa off
echo "ALL_DONE" >> "$LOG"
