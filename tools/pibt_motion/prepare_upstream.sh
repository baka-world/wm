#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'usage: prepare_upstream.sh NEW_EMPTY_OUTPUT_DIRECTORY' >&2
  exit 2
fi
WM_PIBT_OUT="$1"
if [[ -e "$WM_PIBT_OUT" ]] && [[ -n "$(ls -A "$WM_PIBT_OUT")" ]]; then
  echo "refusing to overwrite $WM_PIBT_OUT" >&2
  exit 2
fi
mkdir -p "$WM_PIBT_OUT"
WM_PIBT_OUT="$(realpath "$WM_PIBT_OUT")"
git clone --branch public --single-branch https://github.com/Kei18/pibt2.git "$WM_PIBT_OUT/source"
git -C "$WM_PIBT_OUT/source" checkout --detach faab5b916649549f1cd563df8dbf6e4f6382f631
git -C "$WM_PIBT_OUT/source" submodule update --init --recursive third_party/googletest third_party/grid-pathfinding
git -C "$WM_PIBT_OUT/source" submodule status --recursive | tee "$WM_PIBT_OUT/submodules.txt"
cmake -S "$WM_PIBT_OUT/source" -B "$WM_PIBT_OUT/source/wm-readme-build" > "$WM_PIBT_OUT/readme-build.log" 2>&1
if cmake --build "$WM_PIBT_OUT/source/wm-readme-build" --parallel 2 >> "$WM_PIBT_OUT/readme-build.log" 2>&1; then
  echo 'unmodified README build succeeded' | tee "$WM_PIBT_OUT/readme-build-status.txt"
else
  echo 'unmodified README build failed; see readme-build.log' | tee "$WM_PIBT_OUT/readme-build-status.txt"
  # Do not mask unrelated source, dependency or compiler failures.
  if ! grep -q 'memset.*not a member' "$WM_PIBT_OUT/readme-build.log"; then
    cat "$WM_PIBT_OUT/readme-build.log" >&2
    exit 1
  fi
fi
# No upstream files are modified; the compiler is explicitly told to include
# the standard header omitted by the pinned grid-pathfinding source.
cmake -S "$WM_PIBT_OUT/source" -B "$WM_PIBT_OUT/source/wm-compatible-build" -DCMAKE_BUILD_TYPE=Release '-DCMAKE_CXX_FLAGS=-include cstring' > "$WM_PIBT_OUT/compatible-build.log" 2>&1
cmake --build "$WM_PIBT_OUT/source/wm-compatible-build" --parallel 2 >> "$WM_PIBT_OUT/compatible-build.log" 2>&1
# Upstream native tests use ../tests/instances relative to their build cwd.
(cd "$WM_PIBT_OUT/source/wm-compatible-build" && ./test) > "$WM_PIBT_OUT/native-tests.log" 2>&1
"$WM_PIBT_OUT/source/wm-compatible-build/mapf" -i "$WM_PIBT_OUT/source/instances/mapf/sample.txt" -s PIBT -o "$WM_PIBT_OUT/mapf-result.txt" -v > "$WM_PIBT_OUT/mapf.log" 2>&1
"$WM_PIBT_OUT/source/wm-compatible-build/mapd" -i "$WM_PIBT_OUT/source/instances/mapd/sample.txt" -s PIBT -o "$WM_PIBT_OUT/mapd-result.txt" -v > "$WM_PIBT_OUT/mapd.log" 2>&1
grep -q '^solved=1$' "$WM_PIBT_OUT/mapf-result.txt"
grep -q '^solved=1$' "$WM_PIBT_OUT/mapd-result.txt"
git -C "$WM_PIBT_OUT/source" diff --exit-code
git -C "$WM_PIBT_OUT/source/third_party/grid-pathfinding" diff --exit-code
git -C "$WM_PIBT_OUT/source/third_party/googletest" diff --exit-code
bash tools/pibt_motion/build_bridge.sh "$WM_PIBT_OUT/source" "$WM_PIBT_OUT/bridge"
nm -D "$WM_PIBT_OUT/bridge/libwm_pibt_bridge.so" | c++filt | grep -E 'PIBT::(run|funcPIBT)|wm_pibt_step' | tee "$WM_PIBT_OUT/linked-symbols.txt"
echo 'Upstream source unchanged; native examples, tests and PIBT bridge completed.'
