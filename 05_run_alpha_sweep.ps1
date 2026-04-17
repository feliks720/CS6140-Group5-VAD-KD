# 05_run_alpha_sweep.ps1
# Run alpha sweep experiments for all student architectures.
#
# Alpha controls the balance between KD loss and BCE loss:
#   L = alpha * KL_div(soft) + (1-alpha) * BCE(hard)
#
# Uses ground-truth labels for the BCE hard target component.
#
# Usage:
#   .\05_run_alpha_sweep.ps1                    # Full sweep (all models, all alphas)
#   .\05_run_alpha_sweep.ps1 -Student tiny_cnn  # Single model sweep
#   .\05_run_alpha_sweep.ps1 -EvalOnly          # Skip training, just evaluate

param(
    [string]$Student = "",
    [switch]$EvalOnly
)

$ErrorActionPreference = "Stop"

# Configuration
$Alphas = @(0.3, 0.5, 0.7, 0.9)
$Temperature = 4.0
$Epochs = 30

if ($Student -ne "") {
    $Students = @($Student)
} else {
    $Students = @("tiny_cnn", "mlp", "tiny_transformer")
}

Write-Host "============================================================"
Write-Host "  Alpha Sweep Experiments"
Write-Host "  Students: $($Students -join ', ')"
Write-Host "  Alphas: $($Alphas -join ', ')"
Write-Host "  Temperature: $Temperature"
Write-Host "  GT labels: yes"
Write-Host "============================================================"

if (-not $EvalOnly) {
    foreach ($student in $Students) {
        foreach ($alpha in $Alphas) {
            $expName = "${student}_T${Temperature}_a${alpha}"
            Write-Host ""
            Write-Host "============================================================"
            Write-Host "  Training: $expName (with GT labels)"
            Write-Host "============================================================"

            python 03_train_kd.py `
                --student $student `
                --temperature $Temperature `
                --alpha $alpha `
                --epochs $Epochs `
                --use_gt_labels

            if ($LASTEXITCODE -ne 0) {
                Write-Host "  ERROR: Training failed for $expName" -ForegroundColor Red
                continue
            }
            Write-Host "  Done: $expName" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "  Running comprehensive evaluation..."
Write-Host "============================================================"

python 04_evaluate.py --eval_teacher --splits dev eval

Write-Host ""
Write-Host "  All results saved to results/comprehensive_eval.json"
Write-Host "============================================================"
