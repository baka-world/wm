#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'usage: build_bridge.sh PINNED_UPSTREAM_CHECKOUT EMPTY_BUILD_DIRECTORY' >&2
  exit 2
fi
WM_LACAM_SOURCE="$(realpath "$1")"
WM_LACAM_BUILD="$2"
test "$(git -C "$WM_LACAM_SOURCE" rev-parse HEAD)" = '1c3ecf8ade40298f16047544634e9c0c1cf3b93c'
git -C "$WM_LACAM_SOURCE" diff --quiet
if [[ -e "$WM_LACAM_BUILD" ]] && [[ -n "$(ls -A "$WM_LACAM_BUILD")" ]]; then
  echo "refusing to overwrite nonempty $WM_LACAM_BUILD" >&2
  exit 2
fi
cmake -S native/lacam3_bridge -B "$WM_LACAM_BUILD" -DLACAM3_UPSTREAM="$WM_LACAM_SOURCE" -DCMAKE_BUILD_TYPE=Release
cmake --build "$WM_LACAM_BUILD" --parallel 2
