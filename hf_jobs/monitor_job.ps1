# Temporary HF Job monitor (poll status + recent logs).
# Usage:
#   $env:HF_TOKEN = "hf_xxx"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\monitor_job.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\monitor_job.ps1" -JobId 6aa7936121047bf1b0386550 -IntervalSec 60

param(
    [string]$JobId = "6aa7936121047bf1b0386550",
    [int]$IntervalSec = 60,
    [int]$LogTail = 40
)

$ErrorActionPreference = "Continue"
$env:HF_HUB_DISABLE_EXPERIMENTAL_WARNING = "1"

if (-not $env:HF_TOKEN) {
    Write-Host "请先设置: `$env:HF_TOKEN = '你的Write token'" -ForegroundColor Red
    exit 1
}

$terminal = @("COMPLETED", "ERROR", "CANCELED", "CANCELLED", "FAILED")
Write-Host "Monitoring job $JobId every ${IntervalSec}s  (Ctrl+C to stop)" -ForegroundColor Cyan
Write-Host "URL: https://huggingface.co/jobs/k86574/$JobId"
Write-Host ""

while ($true) {
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "======== $ts ========" -ForegroundColor DarkGray

    $inspect = hf jobs inspect $JobId 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Host $inspect -ForegroundColor Yellow
    } else {
        # Try parse stage from JSON if available; fallback to table text
        $json = hf jobs inspect $JobId --json 2>$null
        if ($LASTEXITCODE -eq 0 -and $json) {
            try {
                $obj = $json | ConvertFrom-Json
                # CLI may return array or object
                if ($obj -is [System.Array]) { $obj = $obj[0] }
                $stage = $obj.status.stage
                if (-not $stage) { $stage = $obj.status }
                $run = $obj.durations.running_secs
                $total = $obj.durations.total_secs
                Write-Host ("stage={0}  running={1}s  total={2}s" -f $stage, $run, $total) -ForegroundColor Green
                Write-Host ("flavor={0}  name={1}" -f $obj.flavor, $obj.name)
            } catch {
                Write-Host $inspect
                $stage = $null
            }
        } else {
            Write-Host $inspect
            if ($inspect -match "COMPLETED|ERROR|CANCELED|CANCELLED|FAILED|RUNNING|SCHEDULING") {
                $stage = $Matches[0]
            } else {
                $stage = $null
            }
        }

        Write-Host "---- logs (tail $LogTail) ----" -ForegroundColor DarkCyan
        hf jobs logs $JobId 2>&1 | Select-Object -Last $LogTail

        if ($stage -and ($terminal -contains $stage.ToUpper())) {
            Write-Host ""
            Write-Host "Job finished: $stage" -ForegroundColor Yellow
            if ($stage.ToUpper() -eq "COMPLETED") {
                Write-Host "下一步: 提交 3B 训练 Job（告诉 Cursor 即可）" -ForegroundColor Green
            }
            exit 0
        }
    }

    Start-Sleep -Seconds $IntervalSec
}
