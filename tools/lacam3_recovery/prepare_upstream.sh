#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'usage: prepare_upstream.sh NEW_EMPTY_OUTPUT_DIRECTORY' >&2
  exit 2
fi
WM_OUT="$1"
if [[ -e "$WM_OUT" ]] && [[ -n "$(ls -A "$WM_OUT")" ]]; then
  echo "refusing to overwrite $WM_OUT" >&2
  exit 2
fi
mkdir -p "$WM_OUT"
WM_OUT="$(realpath "$WM_OUT")"
git clone https://github.com/Kei18/lacam3.git "$WM_OUT/source"
git -C "$WM_OUT/source" checkout --detach 1c3ecf8ade40298f16047544634e9c0c1cf3b93c
git -C "$WM_OUT/source" submodule update --init --recursive
test "$(git -C "$WM_OUT/source/third_party/argparse" rev-parse HEAD)" = '69dabd88a8e6680b1a1a18397eb3e165e4019ce6'
git -C "$WM_OUT/source" submodule status --recursive | tee "$WM_OUT/submodules.txt"
cmake --version > "$WM_OUT/versions.txt"
c++ --version >> "$WM_OUT/versions.txt"
cmake -S "$WM_OUT/source" -B "$WM_OUT/source/build" > "$WM_OUT/readme-build.log" 2>&1
cmake --build "$WM_OUT/source/build" --parallel 2 >> "$WM_OUT/readme-build.log" 2>&1
ctest --test-dir "$WM_OUT/source/build" --output-on-failure > "$WM_OUT/native-tests.log" 2>&1
(
  cd "$WM_OUT/source"
  timeout 90s ./build/main -i assets/random-32-32-10-random-1.scen -m assets/random-32-32-10.map -N 300 -v 3 -o "$WM_OUT/example-result.txt"
) > "$WM_OUT/example.log" 2>&1
grep -q '^solved=1$' "$WM_OUT/example-result.txt"
git -C "$WM_OUT/source" diff --exit-code
git -C "$WM_OUT/source/third_party/argparse" diff --exit-code
bash tools/lacam3_recovery/build_bridge.sh "$WM_OUT/source" "$WM_OUT/bridge" > "$WM_OUT/bridge-build.log" 2>&1
echo 'Official source unchanged; README example, native tests, and solve() bridge completed.'
