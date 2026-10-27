#!/bin/bash
set -e
echo "==> Prefetching models (skips already-downloaded)..."
python /workbench/worker/models/prefetch.py
echo "==> Starting arq worker..."
# arq resolves 'worker.main' relative to cwd, matching how start.ps1 runs it from $root/worker
cd /workbench/worker
exec python -m arq worker.main.WorkerSettings
