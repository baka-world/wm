#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'usage: build_bridge.sh PINNED_UPSTREAM_CHECKOUT EMPTY_BUILD_DIRECTORY' >&2
  exit 2
fi
WM_PIBT_SOURCE="$(realpath "$1")"
WM_PIBT_BUILD="$2"
test "$(git -C "$WM_PIBT_SOURCE" rev-parse HEAD)" = 'faab5b916649549f1cd563df8dbf6e4f6382f631'
test "$(git -C "$WM_PIBT_SOURCE/third_party/grid-pathfinding" rev-parse HEAD)" = 'f444df84459258d7b4d8ceffdbbf2e201b042a68'
git -C "$WM_PIBT_SOURCE" diff --quiet
git -C "$WM_PIBT_SOURCE/third_party/grid-pathfinding" diff --quiet
if [[ -e "$WM_PIBT_BUILD" ]] && [[ -n "$(ls -A "$WM_PIBT_BUILD")" ]]; then
  echo "refusing to overwrite nonempty $WM_PIBT_BUILD" >&2
  exit 2
fi
cmake -S native/pibt_bridge -B "$WM_PIBT_BUILD" -DPIBT_UPSTREAM="$WM_PIBT_SOURCE" -DCMAKE_BUILD_TYPE=Release
cmake --build "$WM_PIBT_BUILD" --parallel 2
