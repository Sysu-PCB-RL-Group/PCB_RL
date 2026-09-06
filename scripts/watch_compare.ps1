# scripts/watch_compare.ps1
# Monitor run_compare A/B experiment progress and estimate ETA.
# Usage (from PCB_RL root in PowerShell):
#   .\scripts\watch_compare.ps1
#   .\scripts\watch_compare.ps1 -Total 10000
# Continuous refresh every 30s:
#   while($true){ cls; .\scripts\watch_compare.ps1; Start-Sleep 30 }
param([int]$Total = 10000)

$proj = Resolve-Path (Join-Path $PSScriptRoot "..")
$compareDir = Join-Path $proj "result\compare"

if (-not (Test-Path $compareDir)) {
    "Not started yet: $compareDir not found"; return
}

$base = Get-ChildItem $compareDir -Directory |
    Where-Object { $_.Name -like "*ran-pang*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $base) {
    "Not started yet: no ran-pang dir under result/compare. Launch run_compare.py first."; return
}

$logs = Get-ChildItem $base.FullName -Recurse -Filter training_log.txt |
    Sort-Object LastWriteTime -Descending
if (-not $logs) {
    "Dir exists but no training_log.txt yet (A group may be in BC pretrain/warmup)."; return
}
$active = $logs | Select-Object -First 1
$phase = if ($active.FullName -like "*baseline*") { "B(baseline)" } else { "A(with_expert)" }

$epPattern = '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?.*\[Ep\s+(\d+)\]'
$lines = Get-Content $active.FullName -Encoding utf8 | Where-Object { $_ -match $epPattern }

function ParseTs($s) {
    $m = [regex]::Match($s, '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?')
    if ($m.Success) { [datetime]::Parse($m.Value) } else { $null }
}

if ($lines.Count -lt 2) {
    $tail = Get-Content $active.FullName -Encoding utf8 -Tail 1
    "${phase}: no episode lines yet (likely BC pretrain/warmup). Last line:`n$tail"
} else {
    $last = $lines[-1]; $prev = $lines[-2]
    $mL = [regex]::Match($last, $epPattern)
    $mP = [regex]::Match($prev, $epPattern)
    $tLast = ParseTs $last; $done = [int]$mL.Groups[1].Value
    $tPrev = ParseTs $prev; $prevN = [int]$mP.Groups[1].Value
    $span = ($tLast - $tPrev).TotalSeconds
    $epInt = $done - $prevN
    if ($span -gt 0 -and $epInt -gt 0) {
        $secPerEp = $span / $epInt
        $remain = ($Total - $done) * $secPerEp
        $eta = $tLast.AddSeconds($remain)
        "${phase}: done $done/$Total (sec/ep {0:N2}, ~{1:N1} ep/min)" -f $secPerEp, (60 / $secPerEp)
        if ($done -ge $Total) {
            "${phase}: RL reached $Total, waiting for training_summary.json..."
        } else {
            "ETA ~{0:N1} min (~{1:N2} h) at {2:HH:mm:ss}" -f ($remain / 60), ($remain / 3600), $eta
        }
    } else {
        "${phase}: parsed $done/$Total but adjacent lines too close to estimate ETA; check later."
    }
}

$aDone = Test-Path (Join-Path $base.FullName "with_expert\training_summary.json")
$bDone = Test-Path (Join-Path $base.FullName "baseline\training_summary.json")
$infA  = Test-Path (Join-Path $base.FullName "inference_eval\with_expert\inference_eval.json")
$infB  = Test-Path (Join-Path $base.FullName "inference_eval\baseline\inference_eval.json")
""
"Done status -> Atrain:$aDone  Btrain:$bDone  infA:$infA  infB:$infB"
if ($aDone -and $bDone -and $infA -and $infB) {
    "ALL DONE. Artifacts in: $($base.FullName)"
}
