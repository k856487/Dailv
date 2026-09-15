$ErrorActionPreference = "Continue"
$env:HF_HUB_DISABLE_EXPERIMENTAL_WARNING = "1"
$tokFile = "D:\hf_models\tools\dailv\.hf_token"
if (-not $env:HF_TOKEN -and (Test-Path $tokFile)) {
    $env:HF_TOKEN = (Get-Content $tokFile -Raw).Trim()
}

function Get-Stage([string]$JobId) {
    $json = hf jobs inspect $JobId --json 2>$null
    if (-not $json) { return "unknown" }
    $tmp = [System.IO.Path]::GetTempFileName()
    [System.IO.File]::WriteAllText($tmp, $json)
    try {
        return (python -c "import json,sys; o=json.load(open(sys.argv[1],encoding='utf-8-sig')); o=o[0] if isinstance(o,list) else o; print(o['status']['stage'])" $tmp).Trim()
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
}

$jobId = "6aa7936121047bf1b0386550"
while ($true) {
    Start-Sleep -Seconds 300
    $stage = Get-Stage $jobId
    $state = Get-Content "D:\hf_models\tools\dailv\unattended_state.json" -Raw -ErrorAction SilentlyContinue
    $prog = (hf jobs logs $jobId 2>$null | Select-String -Pattern "progress \d+/699|Done generating|Pushed dataset" | Select-Object -Last 1)
    Write-Host ("STATUS distill={0} {1}" -f $stage, $prog)
    if ($stage -match "COMPLETED|ERROR|FAILED|CANCELED") {
        Write-Host "AGENT_LOOP_TICK_distill distill_terminal=$stage"
        break
    }
    if ($state -match '"train_job":\s*"[0-9a-f]') {
        Write-Host "AGENT_LOOP_TICK_distill train_job_set"
        break
    }
}
