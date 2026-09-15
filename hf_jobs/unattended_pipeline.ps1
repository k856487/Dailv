# Unattended distill -> train 3B -> merge -> AI-rate eval -> cancel HF jobs -> sleep
# Requires: $env:HF_TOKEN
param(
    [string]$DistillJobId = "6aa7936121047bf1b0386550",
    [int]$PollSec = 120,
    [switch]$SkipSleep
)

$ErrorActionPreference = "Continue"
$env:HF_HUB_DISABLE_EXPERIMENTAL_WARNING = "1"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$StatePath = "D:\hf_models\tools\dailv\unattended_state.json"
$LogPath = "D:\hf_models\tools\dailv\unattended_pipeline.log"
$ReportPath = "D:\hf_models\tools\dailv\unattended_eval_report.json"
$SummaryPath = "D:\hf_models\tools\dailv\unattended_result_report.md"
$MergedLocalDir = "D:\hf_models\paperpass\3b-merged"
$MergedRepo = "k86574/qwen25-3b-paperpass-rewrite-merged"
$LoraRepo = "k86574/qwen25-3b-paperpass-rewrite-lora"
$DistillDataset = "k86574/paperpass-rewrite-distill-3b"
New-Item -ItemType Directory -Force -Path (Split-Path $StatePath) | Out-Null

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $LogPath -Value $line -Encoding utf8
    Write-Host $line
}

function Save-State($obj) {
    ($obj | ConvertTo-Json -Depth 8) | Set-Content -Path $StatePath -Encoding utf8
}

function Ensure-Tun {
    try {
        powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\.cursor\scripts\sesame-tun.ps1" on | Out-Null
        Write-Log "TUN ensured ON"
    } catch {
        Write-Log "TUN enable warning: $_"
    }
}

function Test-ValidToken {
    Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
    python -u "$Root\pipeline_helper.py" validate 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $env:HF_TOKEN = (Get-Content "D:\hf_models\tools\dailv\.hf_token" -Raw).Trim()
        return $true
    }
    Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
    return $false
}

function Ensure-ValidToken {
    if (Test-ValidToken) { return $true }
    Write-Log "HF token invalid; waiting for D:\hf_models\tools\dailv\.hf_token update..."
    python -u "$Root\pipeline_helper.py" wait-token 60 2>&1 | ForEach-Object { Write-Log "  TOKEN $_" }
    $env:HF_TOKEN = (Get-Content "D:\hf_models\tools\dailv\.hf_token" -Raw).Trim()
    return $true
}

function Get-JobStage([string]$JobId) {
    if (-not (Test-ValidToken)) { return $null }
    $stage = python -u "$Root\pipeline_helper.py" job-stage $JobId 2>&1 | Select-Object -Last 1
    if ("$stage" -eq "NO_TOKEN") { return $null }
    return "$stage".Trim()
}

function Wait-Job([string]$JobId, [string]$Label) {
    Write-Log "WAIT $Label job=$JobId"
    $terminalOk = @("COMPLETED")
    $terminalBad = @("ERROR", "FAILED", "CANCELED", "CANCELLED")
    while ($true) {
        Ensure-ValidToken | Out-Null
        $stage = Get-JobStage $JobId
        Write-Log "  $Label stage=$stage"
        if ($stage -and ($terminalOk -contains $stage.ToUpper())) {
            return $true
        }
        if ($stage -and ($terminalBad -contains $stage.ToUpper())) {
            Write-Log "  $Label FAILED stage=$stage"
            hf jobs logs $JobId 2>&1 | Select-Object -Last 80 | ForEach-Object { Write-Log "  LOG $_" }
            return $false
        }
        Start-Sleep -Seconds $PollSec
    }
}

function Wait-DistillDataset {
    Write-Log "WAIT distill job=$DistillJobId (cloud job already running; token optional until submit)"
    $started = Get-Date
    $distillFlag = "D:\hf_models\tools\dailv\distill_done.flag"
    while ($true) {
        if (Test-Path $distillFlag) {
            Write-Log "Distill complete (flag file)"
            return $true
        }
        if (Test-ValidToken) {
            $ready = python -u "$Root\pipeline_helper.py" dataset-ready 2>&1 | Select-Object -Last 1
            if ("$ready" -eq "YES") {
                Write-Log "Distill dataset ready on Hub"
                return $true
            }
            $stage = Get-JobStage $DistillJobId
            Write-Log "  distill stage=$stage dataset=$ready"
            if ($stage -and @("ERROR","FAILED","CANCELED","CANCELLED") -contains $stage.ToUpper()) {
                return $false
            }
            if ($stage -and $stage.ToUpper() -eq "COMPLETED") {
                return $true
            }
        } else {
            $mins = [int]((Get-Date) - $started).TotalMinutes
            Write-Log "  token unavailable; distill assumed RUNNING on HF ($mins min elapsed)"
        }
        Start-Sleep -Seconds $PollSec
    }
}

function Submit-UvJob([string]$ScriptName, [string]$Flavor, [string]$Timeout, [string]$Name) {
    $id = python -u "$Root\submit_uv_job.py" $ScriptName $Flavor $Timeout $Name 2>&1 | Select-Object -Last 1
    if (-not $id -or ("$id" -notmatch '^[0-9a-f]{24}$')) {
        throw "Submit failed for $ScriptName : $id"
    }
    return "$id".Trim()
}

function Cancel-AllJobs {
    Write-Log "Canceling all non-terminal HF Jobs to stop billing..."
    $out = python -u "$Root\cancel_all_jobs.py" 2>&1
    Write-Log "$out"
}

function Invoke-SleepPC {
    # Disabled per user request — never suspend from this pipeline
    Write-Log "Sleep disabled (user requested); not suspending."
    return
}

function Write-SummaryReport($st) {
    $passed = $st.eval_passed
    $dropPct = if ($null -ne $st.mean_drop) { [math]::Round([double]$st.mean_drop * 100, 2) } else { "n/a" }
    $md = @"
# Paperpass 7B→3B 硬蒸馏无人值守结果

- 完成时间: $($st.finished_at)
- 蒸馏 Job: $($st.distill_job)
- 蒸馏数据集: $DistillDataset
- 3B 训练 Job: $($st.train_job)
- 3B LoRA: $LoraRepo
- 3B Merge Job: $($st.merge_job)
- 3B 合并模型 (Hub): $MergedRepo
- 3B 本地下载: $($st.merged_local)
- AI率评估: passed=$passed | mean_before=$($st.mean_ai_before) | mean_after=$($st.mean_ai_after) | mean_drop=${dropPct}pp (阈值≥5pp)
- 错误: $($st.error)
- 详细 JSON: $ReportPath
"@
    Set-Content -Path $SummaryPath -Value $md -Encoding utf8
    Write-Log "Wrote summary $SummaryPath"
}

function Download-MergedModel {
    Write-Log "Downloading merged model to $MergedLocalDir ..."
    New-Item -ItemType Directory -Force -Path $MergedLocalDir | Out-Null
    $env:HF_HOME = "D:\hf_models\cache"
    $env:HF_HUB_CACHE = "D:\hf_models\cache\hub"
    $env:HF_HUB_ENABLE_HF_TRANSFER = "0"
    # Prefer hf CLI (more resilient) then python fallback
    $ok = $false
    for ($i = 1; $i -le 3; $i++) {
        Write-Log "  download attempt $i/3"
        hf download $MergedRepo --local-dir $MergedLocalDir --token $env:HF_TOKEN 2>&1 | ForEach-Object { Write-Log "  DL $_" }
        if (($LASTEXITCODE -eq 0) -and (Test-Path "$MergedLocalDir\config.json") -and (Test-Path "$MergedLocalDir\model.safetensors")) {
            $ok = $true
            break
        }
        Start-Sleep -Seconds 10
    }
    if (-not $ok) {
        Write-Log "WARN: merged download incomplete; eval will use LoRA adapter path"
        return $null
    }
    Write-Log "Merged model ready at $MergedLocalDir"
    return $MergedLocalDir
}

function Test-HubRepoExists([string]$RepoId, [string]$RepoType = "model") {
    $out = python -c @"
from huggingface_hub import HfApi
import os,sys
api=HfApi(token=os.environ.get('HF_TOKEN'))
try:
    api.repo_info('$RepoId', repo_type='$RepoType')
    print('YES')
except Exception:
    print('NO')
"@ 2>&1 | Select-Object -Last 1
    return ("$out" -eq "YES")
}

# ---- main ----
Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
if (Test-ValidToken) { Write-Log "HF token OK" } else { Write-Log "HF token invalid at start; will wait before train/merge submit" }

Ensure-Tun

# Resume from previous state when present (avoid re-submitting completed jobs)
$prev = $null
if (Test-Path $StatePath) {
    try { $prev = Get-Content $StatePath -Raw -Encoding utf8 | ConvertFrom-Json } catch { $prev = $null }
}

$state = [ordered]@{
    started_at = if ($prev -and $prev.started_at) { $prev.started_at } else { (Get-Date).ToString("o") }
    distill_job = $DistillJobId
    train_job = if ($prev) { $prev.train_job } else { $null }
    merge_job = if ($prev) { $prev.merge_job } else { $null }
    distill_done = if ($prev -and $prev.distill_done) { [bool]$prev.distill_done } else { $false }
    train_done = if ($prev -and $prev.train_done) { [bool]$prev.train_done } else { $false }
    merge_done = if ($prev -and $prev.merge_done) { [bool]$prev.merge_done } else { $false }
    merged_local = if ($prev) { $prev.merged_local } else { $null }
    eval_passed = if ($prev) { $prev.eval_passed } else { $null }
    mean_drop = if ($prev) { $prev.mean_drop } else { $null }
    mean_ai_before = if ($prev) { $prev.mean_ai_before } else { $null }
    mean_ai_after = if ($prev) { $prev.mean_ai_after } else { $null }
    error = $null
    finished_at = $null
}
Save-State $state
Write-Log "=== Unattended pipeline START (resumable) ==="

try {
    # 1) Wait distill (job + dataset)
    if (-not $state.distill_done) {
        if (-not (Wait-DistillDataset)) {
            throw "Distill job failed"
        }
        $state.distill_done = $true
        Save-State $state
    } else {
        Write-Log "SKIP distill (already done)"
    }

    Ensure-ValidToken | Out-Null

    # Upload latest scripts (merge + train already on hub; refresh merge)
    Ensure-Tun
    python -c @"
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os
token=os.environ['HF_TOKEN']
api=HfApi(token=token)
repo='k86574/train-scripts'
create_repo(repo, repo_type='model', private=True, token=token, exist_ok=True)
base=Path(r'$Root')
for name in ['train_qwen25_3b_distill_qlora.py','merge_3b_lora.py','distill_generate_from_7b.py']:
    p=base/name
    if p.exists():
        api.upload_file(path_or_fileobj=str(p), path_in_repo=name, repo_id=repo, repo_type='model', token=token)
        print('uploaded', name)
"@
    Write-Log "Scripts refreshed on Hub"

    # 2) Train 3B
    if (-not $state.train_done) {
        if ($state.train_job) {
            Write-Log "RESUME wait train job $($state.train_job)"
            $trainId = "$($state.train_job)"
        } elseif (Test-HubRepoExists $LoraRepo) {
            Write-Log "SKIP train submit (LoRA repo already exists)"
            $state.train_done = $true
            Save-State $state
            $trainId = $null
        } else {
            $trainId = Submit-UvJob "train_qwen25_3b_distill_qlora.py" "l4x1" "3h" "paperpass-3b-distill-train"
            $state.train_job = $trainId
            Save-State $state
            Write-Log "Submitted train job $trainId"
        }
        if ($trainId) {
            if (-not (Wait-Job $trainId "train3b")) {
                throw "Train job failed"
            }
            $state.train_done = $true
            Save-State $state
        }
    } else {
        Write-Log "SKIP train (already done)"
    }

    # 3) Merge 3B
    if (-not $state.merge_done) {
        if ($state.merge_job) {
            Write-Log "RESUME wait merge job $($state.merge_job)"
            $mergeId = "$($state.merge_job)"
        } elseif (Test-HubRepoExists $MergedRepo) {
            Write-Log "SKIP merge submit (merged repo already exists)"
            $state.merge_done = $true
            Save-State $state
            $mergeId = $null
        } else {
            $mergeId = Submit-UvJob "merge_3b_lora.py" "l4x1" "2h" "paperpass-3b-merge"
            $state.merge_job = $mergeId
            Save-State $state
            Write-Log "Submitted merge job $mergeId"
        }
        if ($mergeId) {
            if (-not (Wait-Job $mergeId "merge3b")) {
                throw "Merge job failed"
            }
            $state.merge_done = $true
            Save-State $state
        }
    } else {
        Write-Log "SKIP merge (already done)"
    }

    # 3b) Download usable merged model locally (non-fatal if incomplete; eval can use LoRA)
    $mergedReady = $state.merged_local -and (Test-Path "$($state.merged_local)\model.safetensors")
    if (-not $mergedReady) {
        $dl = Download-MergedModel
        if ($dl) { $state.merged_local = $dl } else { $state.merged_local = $null }
        Save-State $state
    } else {
        Write-Log "SKIP download (merged already at $($state.merged_local))"
    }

    # 4) Local AI-rate eval (merged local if present, else LoRA; detector from ai率检测)
    if ($null -eq $state.eval_passed) {
        Ensure-Tun
        Write-Log "Running local AI-rate eval..."
        $env:HF_HOME = "D:\hf_models\cache"
        $env:HF_HUB_CACHE = "D:\hf_models\cache\hub"
        python -u "$Root\eval_ai_rate_3b.py" 2>&1 | ForEach-Object { Write-Log "EVAL $_" }
        $evalCode = $LASTEXITCODE
        if (Test-Path $ReportPath) {
            $rep = Get-Content $ReportPath -Raw -Encoding utf8 | ConvertFrom-Json
            $state.eval_passed = [bool]$rep.passed
            $state.mean_drop = $rep.mean_drop
            $state.mean_ai_before = $rep.mean_ai_before
            $state.mean_ai_after = $rep.mean_ai_after
        } else {
            $state.eval_passed = $false
            $state.error = "eval report missing; exit=$evalCode"
        }
        Save-State $state
        Write-Log ("Eval passed={0} mean_drop={1}" -f $state.eval_passed, $state.mean_drop)
    } else {
        Write-Log "SKIP eval (already have result passed=$($state.eval_passed))"
    }
}
catch {
    $state.error = "$_"
    Save-State $state
    Write-Log "PIPELINE ERROR: $_"
}
finally {
    Cancel-AllJobs
    $state.finished_at = (Get-Date).ToString("o")
    Save-State $state
    Write-SummaryReport $state
    Write-Log "=== Unattended pipeline END ==="
    Invoke-SleepPC
}
