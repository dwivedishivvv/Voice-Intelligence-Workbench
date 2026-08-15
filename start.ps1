<#
  Dev launcher. Postgres + Redis run in Docker (created from scratch if missing);
  api, worker and web run natively with reload.

    .\start.ps1               # bring everything up
    .\start.ps1 -Stop         # stop native servers, leave the datastores running
    .\start.ps1 -Stop -Down   # ...and stop the containers too

  ASCII-only on purpose: Windows PowerShell 5.1 decodes this file as ANSI and
  chokes on non-ASCII punctuation.
#>
param([switch]$Stop, [switch]$Down)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# HF libs read these from the process env, not .env. The model cache baked under
# MODEL_DIR/.cache is what makes pyannote's nested repo-id refs resolve offline.
$env:MODEL_DIR            = "C:/models"
$env:HF_HOME              = "C:/models/.cache"
$env:PYANNOTE_CACHE       = "C:/models/.cache/hub"   # pyannote ignores HF_HOME
$env:HF_HUB_OFFLINE       = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONPATH           = $root

$APP_PORTS = 8000, 5174

# uvicorn --reload runs the app in a spawned CHILD whose command line is a bare
# `python -c "from multiprocessing.spawn import ..."`, and `npm run dev` wraps vite in a
# cmd.exe. Killing only the process whose command line we recognise orphans those children,
# they keep the listening socket, the next start silently fails to bind, and you get the OLD
# code answering on :8000 -- which looks exactly like "my edit didn't take effect".
# So: kill whole process TREES, and repeat until the ports are actually free.
function Get-Descendants($rootId) {
  $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId
  $out = @(); $frontier = @($rootId)
  while ($frontier) {
    $kids = $all | Where-Object { $frontier -contains $_.ParentProcessId } | Select-Object -Expand ProcessId
    $kids = $kids | Where-Object { $out -notcontains $_ }
    if (-not $kids) { break }
    $out += $kids; $frontier = $kids
  }
  $out
}

function Stop-Servers {
  for ($attempt = 1; $attempt -le 3; $attempt++) {
    $roots = @()
    $roots += (Get-CimInstance Win32_Process |
               Where-Object { $_.CommandLine -match 'uvicorn api\.app\.main|arq worker\.main|vite' } |
               Select-Object -Expand ProcessId)
    foreach ($port in $APP_PORTS) {
      $roots += (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
                 Select-Object -Expand OwningProcess)
    }

    $targets = @()
    foreach ($r in ($roots | Sort-Object -Unique)) { $targets += $r; $targets += (Get-Descendants $r) }
    $targets = $targets | Sort-Object -Unique | Where-Object { $_ -and $_ -ne $PID }

    $killed = 0
    foreach ($procId in $targets) {
      $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
      if ($p) {
        Write-Host "stopping pid $procId ($($p.Name))"
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        $killed++
      }
    }

    # a listening socket outlives its process by a moment; rebinding too fast binds nothing
    Start-Sleep -Milliseconds 1200
    $held = @($APP_PORTS | Where-Object {
      Get-NetTCPConnection -State Listen -LocalPort $_ -ErrorAction SilentlyContinue })
    if (-not $held) { return }
    if ($killed -eq 0 -and $attempt -gt 1) { break }
  }

  $held = @($APP_PORTS | Where-Object {
    Get-NetTCPConnection -State Listen -LocalPort $_ -ErrorAction SilentlyContinue })
  if ($held) { Write-Warning "port(s) $($held -join ', ') still held - the next start may bind nothing." }
}

if ($Stop) {
  Stop-Servers
  if ($Down) { docker compose -f "$root/docker-compose.yml" stop }
  Write-Host "stopped." -ForegroundColor Green
  exit 0
}

# ---- datastores ----------------------------------------------------------
Write-Host "starting postgres + redis..." -ForegroundColor Cyan
docker compose -f "$root/docker-compose.yml" up -d
if ($LASTEXITCODE -ne 0) { throw "docker compose failed. Is Docker Desktop running?" }

# compose's own healthchecks are the source of truth; just wait them out
$deadline = (Get-Date).AddSeconds(120)
do {
  $states = docker compose -f "$root/docker-compose.yml" ps --format '{{.Health}}'
  if ($states -notcontains "starting" -and $states -notcontains "") { break }
  Start-Sleep 2
} while ((Get-Date) -lt $deadline)
if ($states -contains "unhealthy" -or (Get-Date) -ge $deadline) {
  throw "datastores did not become healthy: $states"
}
Write-Host "datastores healthy." -ForegroundColor Green

# ---- worker interpreter --------------------------------------------------
# worker/.venv is python 3.11 with the cu121 torch + ctranslate2 build. The global
# interpreter has torch+cpu, which silently runs the whole pipeline on CPU.
$workerPy = Join-Path $root "worker\.venv\Scripts\python.exe"
if (-not (Test-Path $workerPy)) {
  throw "missing $workerPy. Create it with: uv venv worker/.venv --python 3.11; uv pip install -p $workerPy -r worker/pyproject.toml"
}
$gpu = & $workerPy -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
if ($gpu) {
  Write-Host "GPU: $gpu" -ForegroundColor Green
} else {
  Write-Warning "worker venv has a CPU-only torch, so the pipeline will run on CPU."
}

# ---- native servers, one window each -------------------------------------
$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell" }

function Start-Window($title, $wd, $cmd) {
  $prelude = "`$Host.UI.RawUI.WindowTitle='$title'; Set-Location '$wd'; " +
             "`$env:MODEL_DIR='$env:MODEL_DIR'; `$env:HF_HOME='$env:HF_HOME'; " +
             "`$env:PYANNOTE_CACHE='$env:PYANNOTE_CACHE'; `$env:HF_HUB_OFFLINE='1'; " +
             "`$env:TRANSFORMERS_OFFLINE='1'; `$env:PYTHONPATH='$root'; "
  Start-Process $shell -ArgumentList @("-NoExit", "-Command", ($prelude + $cmd))
}

Start-Window "api"    $root          "python -m uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir api --reload-dir common"
Start-Window "worker" "$root\worker" "& '$workerPy' -m arq worker.main.WorkerSettings"
Start-Window "web"    "$root\web"    "npm run dev"

Write-Host ""
Write-Host "  api    http://localhost:8000/healthz" -ForegroundColor Green
Write-Host "  web    http://localhost:5174" -ForegroundColor Green
Write-Host "  worker loads ~2GB of models, first request waits on it" -ForegroundColor DarkGray
