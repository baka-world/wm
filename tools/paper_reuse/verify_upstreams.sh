#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'usage: verify_upstreams.sh NEW_EMPTY_OUTPUT_DIRECTORY' >&2
  exit 2
fi
WM_UPSTREAM_OUT="$1"
if [[ -e "$WM_UPSTREAM_OUT" ]] && [[ -n "$(ls -A "$WM_UPSTREAM_OUT")" ]]; then
  echo "refusing to overwrite $WM_UPSTREAM_OUT" >&2
  exit 2
fi
mkdir -p "$WM_UPSTREAM_OUT"
WM_ROOT="$(pwd)"
WM_OR_SHA=100f66e6242ab8bf8d32feb8f3bf086db66ae2b5
WM_MBRL_SHA=3f93cccfc8d635f74e335a2f07aab6e9a48fc021
curl -fsSL "https://raw.githubusercontent.com/google/or-tools/$WM_OR_SHA/examples/python/flexible_job_shop_sat.py" -o "$WM_UPSTREAM_OUT/flexible_job_shop_sat.py"
curl -fsSL "https://raw.githubusercontent.com/google/or-tools/$WM_OR_SHA/LICENSE" -o "$WM_UPSTREAM_OUT/ortools-LICENSE"
printf '%s  %s\n' ed9484f61c0d8064204b3aad32958a0e4e6eaeb377abe58a3b85f8681034c44f "$WM_UPSTREAM_OUT/flexible_job_shop_sat.py" | sha256sum -c -
printf '%s  %s\n' cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30 "$WM_UPSTREAM_OUT/ortools-LICENSE" | sha256sum -c -
timeout 60s python "$WM_UPSTREAM_OUT/flexible_job_shop_sat.py" > "$WM_UPSTREAM_OUT/ortools-example.log" 2>&1
grep -q 'status: OPTIMAL' "$WM_UPSTREAM_OUT/ortools-example.log"
git init -q "$WM_UPSTREAM_OUT/mbrl"
git -C "$WM_UPSTREAM_OUT/mbrl" remote add origin https://github.com/facebookresearch/mbrl-lib.git
git -C "$WM_UPSTREAM_OUT/mbrl" fetch --depth 1 origin "$WM_MBRL_SHA"
git -C "$WM_UPSTREAM_OUT/mbrl" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$WM_UPSTREAM_OUT/mbrl" rev-parse HEAD)" == "$WM_MBRL_SHA" ]]
(cd "$WM_UPSTREAM_OUT/mbrl" && python -m pytest -q \
  tests/core/test_models.py::test_activation_functions_guassian_mlp \
  tests/core/test_models.py::test_basic_ensemble_gaussian_forward \
  tests/core/test_models.py::test_model_trainer_batch_callback \
  tests/core/test_replay_buffer.py::test_replay_buffer_batched_add \
  tests/core/test_replay_buffer.py::test_bootstrap_iterator) > "$WM_UPSTREAM_OUT/mbrl-pytest.log" 2>&1
python "$WM_ROOT/tools/paper_reuse/upstream_mbrl_smoke.py" > "$WM_UPSTREAM_OUT/mbrl-model-smoke.log" 2>&1
[[ -z "$(git -C "$WM_UPSTREAM_OUT/mbrl" status --porcelain)" ]]
printf 'OR-Tools=%s; MBRL-Lib=%s; original sources unchanged\n' "$WM_OR_SHA" "$WM_MBRL_SHA"
