# ═══════════════════════════════════════════════════════════════════════════════
# start_night_run.ps1 — Yövahti: käynnistää yöajon kun mallit valmiit ja GPU vapaa
# ═══════════════════════════════════════════════════════════════════════════════
# Odottaa kunnes:
#   1) qwen3:30b-thinking (mikä tahansa "thinking"-qwen-tagi) näkyy `ollama list`:issä
#   2) qwen3-coder:30b on asennettu
#   3) yksikään muu `python main.py` -ajo ei ole käynnissä (GPU vapaa, ei VRAM-konfliktia)
# Sitten: patchaa night_run.yaml todellisella thinking-tagilla (jos eroaa), validoi
#         konffin ja käynnistää `python main.py config/night_run.yaml`.
#
# Ajetaan taustalla. Turvallinen jättää valvomatta yön yli (read-only kunnes ehdot täyttyvät).
# ═══════════════════════════════════════════════════════════════════════════════

$ErrorActionPreference = "SilentlyContinue"
$proj = "c:\Projektit\learn_multiagent"
$py   = "C:\Users\tobia\condamini3\envs\langgraph-rl\python.exe"
$cfg  = "config/night_run.yaml"
Set-Location $proj
$env:PYTHONUTF8 = "1"

function Log($m) { Write-Output ("[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] " + $m) }

Log "Yovahti kaynnistyi. Odotetaan: qwen3:30b-thinking + qwen3-coder:30b + vapaa GPU..."
$deadline = (Get-Date).AddHours(12)
$tag = $null
while ((Get-Date) -lt $deadline) {
    $lines = & ollama list 2>$null
    $haveCoder = (@($lines | Where-Object { $_ -match 'qwen3-coder:30b' })).Count -gt 0
    $thinkingLine = $lines | Where-Object { $_ -match 'thinking' } | Select-Object -First 1
    $running = (@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                 Where-Object { $_.CommandLine -match 'main\.py' })).Count
    if ($thinkingLine -and $haveCoder -and $running -eq 0) {
        $tag = ($thinkingLine.Trim() -split '\s+')[0]
        break
    }
    Start-Sleep -Seconds 120
}

if (-not $tag) { Log "AIKAKATKAISU (12h): thinking-mallia ei nakynyt. Yoajoa EI kaynnistetty."; exit 1 }
Log ("Loytyi thinking-tagi: " + $tag + "  (coder valmis, GPU vapaa)")

# Patchaa night_run.yaml jos todellinen tagi eroaa oletuksesta
if ($tag -ne "qwen3:30b-thinking") {
    Log ("Patchataan night_run.yaml: qwen3:30b-thinking -> " + $tag)
    $content = Get-Content $cfg -Raw
    $content = $content.Replace("qwen3:30b-thinking", $tag)
    Set-Content -Path $cfg -Value $content -Encoding utf8
}

# Validoi konffi ennen kuin sitoudutaan yön mittaiseen ajoon
$val = & $py -c "from src.config_loader import load_config; load_config('config/night_run.yaml'); print('CFG_OK')" 2>&1 | Out-String
if ($val -notmatch "CFG_OK") { Log ("Konffin validointi EPAONNISTUI, yoajoa ei kaynnistetty: " + $val); exit 1 }
Log "Konffi validoitu OK. Annetaan VRAM:n asettua 20s..."
Start-Sleep -Seconds 20

Log "=== KAYNNISTETAAN YOAJO: config/night_run.yaml ==="
& $py main.py $cfg
Log ("=== Yoajo paattyi (process exit) " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " ===")
