"""Run an experiment inside a Kaggle GPU notebook (T4 x2) with local Ollama servers.

One Ollama server per GPU, each serving the same model digest; the design's plans are
split into contiguous ranges, one worker per GPU. Each worker is a separate run
directory under /kaggle/working/results/ollama/. At the end the script prints the
--ranges value to resume in the next session and zips the results.

Usage (from the notebook cell, after cloning the repo to /tmp/repo):
    python /tmp/repo/kaggle/kaggle_runner.py --config configs/experiments/e1_drift_kaggle.yaml
    python /tmp/repo/kaggle/kaggle_runner.py --smoke
    python /tmp/repo/kaggle/kaggle_runner.py --ranges 120:282,402:565
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path("/kaggle/working/results")
DEFAULT_DIGEST = "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"


def sh(cmd: str, **kw) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def gpu_count() -> int:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return len([line for line in out.stdout.splitlines() if line.startswith("GPU")])


def start_ollama(gpu: int, port: int) -> subprocess.Popen:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OLLAMA_HOST=f"127.0.0.1:{port}",
               OLLAMA_KEEP_ALIVE="24h")
    log = open(f"/kaggle/working/ollama_gpu{gpu}.log", "w")
    proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            get_json(f"http://127.0.0.1:{port}/api/version")
            return proc
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"ollama on GPU {gpu} did not start; see ollama_gpu{gpu}.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/e1_drift_kaggle.yaml")
    ap.add_argument("--ranges", help="comma-separated start:end plan ranges, one per worker")
    ap.add_argument("--time-budget-hours", type=float, default=10.5)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--expected-digest", default=DEFAULT_DIGEST)
    ap.add_argument("--smoke", action="store_true", help="1 conversation per GPU, then stop")
    args = ap.parse_args()

    if sys.version_info < (3, 11):  # noqa: UP036 - runs on Kaggle's interpreter, not ours
        sys.exit(f"Python 3.11+ required, found {sys.version}")
    n_gpu = gpu_count()
    if n_gpu == 0:
        sys.exit("No GPU visible. Notebook settings -> Accelerator -> GPU T4 x2.")
    print(f"GPUs: {n_gpu}", flush=True)

    # 1. Ollama: install, one server per GPU, pull once, pin the exact weights.
    if shutil.which("ollama") is None:
        sh("apt-get -qq update && apt-get -qq install -y zstd > /dev/null")
        sh("curl -fsSL https://ollama.com/install.sh | sh > /dev/null")
    ports = [11434 + g for g in range(n_gpu)]
    servers = [start_ollama(g, p) for g, p in enumerate(ports)]
    sh(f"OLLAMA_HOST=127.0.0.1:{ports[0]} ollama pull {args.model}")
    tags = get_json(f"http://127.0.0.1:{ports[0]}/api/tags")["models"]
    digest = next(m["digest"] for m in tags if m["name"] == args.model)
    if args.expected_digest and digest != args.expected_digest:
        sys.exit(f"Model digest {digest} != expected {args.expected_digest}. "
                 "Different weights would not be comparable; stopping.")
    print(f"model {args.model} digest {digest} (matches expected)", flush=True)

    # 2. Package and pinned corpus (the config's corpus_sha256 is checked at plan time).
    sh(f"pip install -q -e {REPO}")
    if not (REPO / "data/corpus/wikitext103_test.jsonl").exists():
        sh("python scripts/fetch_corpus.py > /dev/null", cwd=REPO)

    # 3. Split the plans into one contiguous range per GPU.
    count = subprocess.run(
        [sys.executable, "-c", "import sys; from instruction_drift.config import load_config;"
         "from instruction_drift.runner import build_plans;"
         "print(len(build_plans(load_config(sys.argv[1]))[0]))", args.config],
        cwd=REPO, capture_output=True, text=True, check=True)
    total = int(count.stdout.strip())
    if args.ranges:
        ranges = [tuple(int(x) for x in r.split(":")) for r in args.ranges.split(",")]
    else:
        step = -(-total // n_gpu)
        ranges = [(s, min(total, s + step)) for s in range(0, total, step)]
    if args.smoke:
        ranges = [(a, a + 1) for a, _ in ranges]
    if len(ranges) > n_gpu:
        sys.exit(f"{len(ranges)} ranges but only {n_gpu} GPUs")
    print(f"valid plans: {total}; this session: {ranges}", flush=True)

    # 4. One worker per GPU, in parallel.
    out_root = OUT.parent / "results_smoke" if args.smoke else OUT
    workers = []
    for (a, b), port in zip(ranges, ports, strict=False):
        cmd = [sys.executable, "-m", "instruction_drift.cli", "experiment", "run",
               "--config", args.config, "--offset", str(a), "--max-conversations", str(b - a),
               "--time-budget-hours", str(args.time_budget_hours),
               "--output-root", str(out_root), "--base-url", f"http://127.0.0.1:{port}"]
        log = open(f"/kaggle/working/worker_{a}_{b}.log", "w")
        workers.append(((a, b), subprocess.Popen(cmd, cwd=REPO, stdout=log,
                                                 stderr=subprocess.STDOUT)))
    t0 = time.time()
    while any(p.poll() is None for _, p in workers):
        time.sleep(300)
        print(f"[{(time.time() - t0) / 3600:.1f} h] workers running...", flush=True)
    for s in servers:
        s.terminate()

    # 5. Report, resume ranges, and zip.
    remaining, failed = [], False
    for (a, b), p in workers:
        if p.returncode != 0:
            failed = True
            print(f"worker {a}:{b} FAILED (exit {p.returncode}); see worker_{a}_{b}.log")
            remaining.append(f"{a}:{b}")
            continue
    for summary_path in sorted(out_root.glob("ollama/*/summary.json")):
        s = json.loads(summary_path.read_text())
        if summary_path.stat().st_mtime < t0:
            continue
        end = s["next_offset"]
        print(f"{summary_path.parent.name}: conversations={s['conversations']} "
              f"aborted={s['aborted']} checkpoints={s['checkpoints']} "
              f"next_offset={s['next_offset']} stopped_early={s['stopped_early']}")
        match = [r for r in ranges if r[0] == s["plan_offset"]]
        if match and end < match[0][1]:
            remaining.append(f"{end}:{match[0][1]}")
    archive = shutil.make_archive(f"/kaggle/working/{out_root.name}_{int(t0)}", "zip", out_root)
    print(f"\nresults archive: {archive}")
    if args.smoke:
        print("Smoke test finished. Set SMOKE = False and run again for the real session.")
    elif remaining:
        print(f"\nNOT FINISHED. Next session: RANGES = \"{','.join(remaining)}\"")
    else:
        print("\nAll plans in this design are done.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
