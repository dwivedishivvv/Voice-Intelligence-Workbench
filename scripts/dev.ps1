<#
.SYNOPSIS
  Run the stack natively on Windows (Postgres + Redis stay in Docker).

.DESCRIPTION
  Docker's api/worker images still work — this is for running the Python side on the host,
  which is faster to iterate on and lets the worker use a local GPU.

  Open one terminal per component:

      .\scripts\dev.ps1 infra     # postgres + redis in Docker, waits for healthy
      .\scripts\dev.ps1 api       # uvicorn on :8000
      .\scripts\dev.ps1 worker    # arq job worker
      .\scripts\dev.ps1 web       # vite on :5174

  First run only:
      uv venv --python 3.11 .venv;        $env:VIRTUAL_ENV=".venv";        uv pip install -r pyproject.toml --extra api
      uv venv --python 3.11 .venv-worker; $env:VIRTUAL_ENV=".venv-worker"; uv pip install -r pyproject.toml --extra worker
      cd web; npm install; cd ..

  Python 3.11 is not optional: torch 2.4 and numpy 1.26 publish no 3.13 wheels.
  The web UI reads its API key from localStorage — in the browser console, once:
      localStorage.setItem("api_key", "<API_KEY from .env>")
#>
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("infra", "api", "worker", "web")]
  [string]$Component
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# .env points at Docker-internal hostnames; running on the host needs localhost instead.
# Process env wins over the .env file in pydantic-settings, so these override cleanly.
$env:POSTGRES_HOST = "localhost"
$env:REDIS_URL = "redis://localhost:6379/0"
$env:DATA_DIR = "./.runtime/data"
$env:PYTHONPATH = ".;worker"

# models/REGISTRY.yaml hardcodes absolute /models/... paths, which on Windows resolve to
# C:\models. Pointing MODEL_DIR anywhere else would leave verify_models() looking in a
# different place than the Settings page downloads to.
$env:MODEL_DIR = "C:/models"
$env:HF_HOME = "C:/models/.cache"
# pyannote ignores HF_HOME and uses its own cache var to resolve nested submodel refs
$env:PYANNOTE_CACHE = "C:/models/.cache/hub"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

switch ($Component) {
  "infra" {
    docker compose up -d postgres redis
    Write-Host "waiting for healthy..." -ForegroundColor DarkGray
    do {
      Start-Sleep -Seconds 2
      $state = docker compose ps --format "{{.Service}} {{.Status}}"
      $state | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    } until (($state -match "healthy").Count -ge 2)
    Write-Host "postgres + redis ready" -ForegroundColor Green
  }

  "api" {
    & .\.venv\Scripts\python.exe -m uvicorn api.app.main:app `
      --host 0.0.0.0 --port 8000 --reload --reload-dir api/app --reload-dir common
  }

  "worker" {
    # the worker must never reach the network for weights — a missing model should fail
    # loudly at startup, not silently download mid-race
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    & .\.venv-worker\Scripts\python.exe -m arq worker.main.WorkerSettings
  }

  "web" {
    Set-Location (Join-Path $root "web")
    npm run dev
  }
}
