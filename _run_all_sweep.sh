#!/bin/bash
# Run all alpha sweep experiments sequentially
# Uses GT labels, T=4.0, 30 epochs

PYTHON=".venv/Scripts/python.exe"
COMMON="--temperature 4 --epochs 30 --use_gt_labels"

echo "Starting alpha sweep: 3 models x 4 alphas = 12 runs"
echo "============================================================"

for student in tiny_cnn mlp tiny_transformer; do
    for alpha in 0.3 0.5 0.7 0.9; do
        echo ""
        echo ">>> Training: ${student} alpha=${alpha} (GT labels)"
        $PYTHON 03_train_kd.py --student $student --alpha $alpha $COMMON 2>&1 | tail -15
        echo "<<< Done: ${student} alpha=${alpha}"
    done
done

echo ""
echo "============================================================"
echo "All training complete. Running evaluation..."
echo "============================================================"
$PYTHON 04_evaluate.py --eval_teacher --splits dev eval 2>&1
