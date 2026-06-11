# ═══════════════════════════════════════════════════════════════════════════════
# run_robot_experiment.ps1 — Ketjuta robottikäsi-koe: SEEDED -> BLIND
# ═══════════════════════════════════════════════════════════════════════════════
# 1) robot_arm_seeded  (kyvykkyysdemo: HER annettu -> pitäisi ratkaista + video)
# 2) robot_arm_blind   (älytesti: ei vihjettä -> osaako keksiä HER:in itse?)
# Ajaa peräkkäin (sama GPU + mallit), turvallinen jättää yön yli.
# ═══════════════════════════════════════════════════════════════════════════════
$ErrorActionPreference = "Continue"
$proj = "c:\Projektit\learn_multiagent"
$py   = "C:\Users\tobia\condamini3\envs\langgraph-rl\python.exe"
Set-Location $proj
$env:PYTHONUTF8 = "1"
function Log($m) { Write-Output ("[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] " + $m) }

Log "=== ROBOT EXPERIMENT START ==="
Log ">>> RUN 1/2: SEEDED (capability demo - HER pre-seeded in Codex)"
& $py main.py config/robot_arm_seeded.yaml
Log ">>> SEEDED finished. Starting RUN 2/2: BLIND (intelligence test - no HER hint)"
& $py main.py config/robot_arm_blind.yaml
Log "=== BOTH RUNS COMPLETE ==="
