"""Reproducibility snapshot written before any PR31 training updates.

Stores exact available execution environment, package versions, source
hashes and lock files. No passwords, tokens, credentials or full process
environment are stored. Checkpoint resumption is handled by pr31_train.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


REPO_ROOT=Path(__file__).resolve().parents[3]
SOURCES=(
    "experiments/world_model_integration/posttraining/pr31_protocol.json",
    "experiments/world_model_integration/posttraining/pr31_dataset.py",
    "experiments/world_model_integration/posttraining/pr31_train.py",
    "experiments/world_model_integration/posttraining/pr31_evaluate.py",
    "experiments/world_model_integration/posttraining/repro_environment.py",
    "experiments/world_model_integration/build_dataset_v4.py",
    "experiments/world_model_integration/build_dataset.py",
    "experiments/world_model_integration/train_v4.py",
    "lab_transport/business_session_learning.py",
    "lab_transport/business_session_v3.py",
    "business_transport/session_features_v3.py",
    "business_transport/session_events.py",
    "business_transport/state_features.py",
    "business_transport/simulation.py",
    "requirements-paper-reuse.lock",
)


def run(args):
    p=subprocess.run(args,cwd=REPO_ROOT,check=True,capture_output=True,text=True)
    return p.stdout.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(out):
    import torch
    import numpy as np
    out=Path(out)
    if out.exists():
        raise FileExistsError("environment archive must be write-once")
    out.mkdir(parents=True)
    freeze=run([sys.executable,"-m","pip","freeze","--all"])
    (out/"pip-freeze.txt").write_text(freeze+"\n")
    source_manifest={}
    for name in SOURCES:
        path=REPO_ROOT/name
        if not path.is_file():
            raise FileNotFoundError(name)
        source_manifest[name]=sha(path)
    (out/"source_manifest.json").write_text(json.dumps(source_manifest,indent=2)+"\n")
    for name in ("requirements-paper-reuse.lock",):
        shutil.copy2(REPO_ROOT/name,out/name)
    relevant=("GITHUB_SHA","GITHUB_RUN_ID","GITHUB_REF_NAME",
              "RUNNER_OS","RUNNER_ARCH","CUDA_VISIBLE_DEVICES",
              "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
              "PYTHONHASHSEED")
    environment={
        "schema":"rhwm-pr31-repro-environment-v1",
        "python_version":sys.version,"python_executable":sys.executable,
        "platform":platform.platform(),"machine":platform.machine(),
        "processor":platform.processor(),"os_release":platform.freedesktop_os_release(),
        "pytorch":torch.__version__,
        "torch_cuda_version":torch.version.cuda,
        "torch_cuda_available":torch.cuda.is_available(),
        "torch_num_threads":torch.get_num_threads(),
        "numpy":np.__version__,
        "torch_config":torch.__config__.show(),
        "git_commit":run(["git","rev-parse","HEAD"]),
        "git_worktree_status":run(["git","status","--porcelain"]),
        "relevant_environment":{k:os.environ.get(k) for k in relevant},
        "pip_freeze_sha256":sha(out/"pip-freeze.txt"),
        "requirements_lock_sha256":sha(out/"requirements-paper-reuse.lock"),
        "source_manifest_sha256":sha(out/"source_manifest.json"),
    }
    (out/"environment.json").write_text(json.dumps(environment,indent=2,ensure_ascii=False)+"\n")
    try:
        cpuinfo=run(["lscpu"])
        (out/"lscpu.txt").write_text(cpuinfo+"\n")
    except (FileNotFoundError,subprocess.CalledProcessError):
        (out/"lscpu.txt").write_text("not available\n")
    print("PR31_ENVIRONMENT_SNAPSHOT",json.dumps({
        "git_commit":environment["git_commit"],
        "torch":environment["pytorch"],
        "numpy":environment["numpy"],
        "pip_sha256":environment["pip_freeze_sha256"],
        "source_manifest_sha256":environment["source_manifest_sha256"]
    }))
    return environment


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    snapshot(args.out)
