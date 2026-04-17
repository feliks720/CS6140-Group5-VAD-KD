#!/bin/bash
# 05_run_alpha_sweep.sh
# Run alpha sweep experiments for all student architectures.
#
# Alpha controls the balance between KD loss and BCE loss:
#   L = alpha * KL_div(soft) + (1-alpha) * BCE(hard)
#
# Uses ground-truth labels for the BCE hard target component.
#
# Usage:
#   bash 05_run_alpha_sweep.sh              # Full sweep (all models, all alphas)
#   bash 05_run_alpha_sweep.sh tiny_cnn     # Single model sweep

set -e

# Configuration
ALPHAS=(0.3 0.5 0.7 0.9)
TEMPERATURE=4.0
EPOCHS=30
STUDENTS=("tiny_cnn" "mlp" "tiny_transformer")

# Allow filtering to a single student
if [ -n "$1" ]; then
    STUDENTS=("$1")
fi

echo "============================================================"
echo "  Alpha Sweep Experiments"
echo "  Students: ${STUDENTS[*]}"
echo "  Alphas: ${ALPHAS[*]}"
echo "  Temperature: ${TEMPERATURE}"
echo "  GT labels: yes"
echo "============================================================"

for student in "${STUDENTS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        exp_name="${student}_T${TEMPERATURE}_a${alpha}"
        echo ""
        echo "============================================================"
        echo "  Training: ${exp_name} (with GT labels)"
        echo "============================================================"

        python 03_train_kd.py \
            --student "$student" \
            --temperature "$TEMPERATURE" \
            --alpha "$alpha" \
            --epochs "$EPOCHS" \
            --use_gt_labels

        echo "  Done: ${exp_name}"
    done
done

echo ""
echo "============================================================"
echo "  Alpha sweep complete!"
echo "  Running comprehensive evaluation..."
echo "============================================================"

python 04_evaluate.py --eval_teacher --splits dev eval

echo ""
echo "  All results saved to results/comprehensive_eval.json"
echo "============================================================"
