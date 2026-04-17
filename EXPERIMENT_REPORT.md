# Experiment Report: VAD Knowledge Distillation
## CS6140 Machine Learning - Group 5

### 1. Overview

This project applies **Knowledge Distillation (KD)** to compress a pretrained CRDNN-based Voice Activity Detection (VAD) teacher model into lightweight student architectures. We evaluate three student models (TinyCNN, MLP, TinyTransformer) and study the effect of distillation temperature on performance.

- **Dataset:** LibriParty (train: 250 files, dev: 50 files, eval: 50 files, ~240 min per split)
- **Teacher:** SpeechBrain pretrained CRDNN (`speechbrain/vad-crdnn-libriparty`)
- **Hardware:** NVIDIA RTX 3090, Windows 10, Python 3.12, PyTorch 2.5.1+CUDA 12.4
- **Training:** 30 epochs, batch_size=16, lr=0.001, Adam optimizer with weight decay 1e-4

---

### 2. Baseline Results

#### 2.1 Teacher Model (CRDNN)

| Metric | Value |
|--------|-------|
| Parameters | 109,744 |
| Model size | 0.43 MB |
| RTF (eval set) | 0.005 (200x real-time) |
| Avg speech ratio | 45.6% |
| Eval files | 50 sessions |
| Total audio | 240.8 min |

The CRDNN teacher serves as the upper bound for student model quality and as the source of soft labels for KD training.

#### 2.2 Energy VAD Baseline

| Metric | Value |
|--------|-------|
| Method | Hysteresis thresholding on log-energy |
| Activation threshold | 0.1 |
| Deactivation threshold | 0.05 |
| RTF | 0.0073 |
| Avg speech ratio | 7.4% |

The energy VAD severely under-detects speech (7.4% vs teacher's 45.6%), demonstrating that simple signal-level features are insufficient for VAD in noisy, multi-speaker conditions like LibriParty. This motivates the use of neural approaches.

---

### 3. Knowledge Distillation Results

#### 3.1 Student Architecture Comparison (T=4, alpha=0.7)

| Model | Params | Size (MB) | Best F1 | Precision | Recall | Accuracy | DER | Best Epoch |
|-------|--------|-----------|---------|-----------|--------|----------|-----|------------|
| **Teacher (CRDNN)** | 109,744 | 0.43 | -- | -- | -- | -- | -- | -- |
| **TinyCNN** | 14,913 | 0.057 | 0.7707 | 0.7774 | 0.7640 | 0.8270 | 0.3703 | 24 |
| **MLP** | 81,921 | 0.313 | 0.7841 | 0.7651 | 0.8040 | 0.8315 | 0.3476 | 19 |
| **TinyTransformer** | 389,633 | 1.486 | **0.8122** | 0.7542 | **0.8800** | **0.8452** | **0.2962** | 24 |

**Key observations:**

1. **TinyTransformer achieves the best F1 (0.8122)** with the lowest DER (0.2962) and highest recall (0.88). However, it is 3.5x larger than the teacher in parameter count -- it trades compression for quality.

2. **TinyCNN is the most efficient** at 14,913 parameters (7.4x compression vs teacher), achieving F1=0.77 in only 57 KB. This makes it the most practical choice for edge deployment.

3. **MLP offers a middle ground** -- 1.3x compression with F1=0.78, higher recall (0.804) than TinyCNN, and faster convergence (epoch 19).

4. All three KD students significantly outperform the energy VAD baseline, validating that knowledge distillation successfully transfers the teacher's ability to handle noisy audio.

#### 3.2 Compression Ratios vs Teacher

| Model | Param Ratio | Size Ratio | F1 Retention (eval GT) |
|-------|-------------|------------|------------------------|
| TinyCNN | **7.4x smaller** | **7.5x smaller** | 85.1% (0.81 / 0.95) |
| MLP | 1.3x smaller | 1.4x smaller | **89.9%** (0.86 / 0.95) |
| TinyTransformer | 3.5x larger | 3.5x larger | 89.7% (0.85 / 0.95) |

> F1 retention = student eval F1 (GT) / teacher eval F1 (GT=0.9517). Now computable with ground-truth evaluation (Section 7).

---

### 4. Temperature Sweep (TinyCNN, alpha=0.7)

| Temperature | Best F1 | Precision | Recall | Accuracy | DER | Best Epoch |
|-------------|---------|-----------|--------|----------|-----|------------|
| T=1 | **0.7759** | **0.8104** | 0.7443 | **0.8365** | **0.3626** | 17 |
| T=2 | 0.7720 | 0.7634 | 0.7808 | 0.8245 | 0.3678 | 23 |
| T=4 | 0.7707 | 0.7774 | 0.7640 | 0.8270 | 0.3703 | 24 |
| T=8 | 0.7667 | 0.7595 | 0.7741 | 0.8208 | 0.3764 | 22 |

**Key observations:**

1. **Lower temperatures perform slightly better** for TinyCNN on this task. T=1 achieves the highest F1 (0.7759) and lowest DER (0.3626), suggesting that the teacher's hard decisions are already quite informative for this binary classification task.

2. **T=1 converges fastest** (best at epoch 17 vs 22-24 for higher T), consistent with sharper supervision providing stronger gradients.

3. **T=1 has highest precision (0.8104) but lowest recall (0.7443)**, while higher temperatures shift the precision-recall trade-off toward recall. This is expected: higher T softens the teacher distribution, giving the student more nuanced "uncertainty" information but weaker decision boundaries.

4. The F1 differences across temperatures are small (0.77 range), indicating that TinyCNN's limited capacity is the bottleneck, not the distillation temperature.

---

### 5. Training Configuration

All experiments used the following shared hyperparameters:

```yaml
optimizer: Adam
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 16
epochs: 30
alpha: 0.7          # weight for KD loss (0.3 for BCE)
feature: log-mel     # 80-dim log-mel spectrogram
frame_length: 25 ms
frame_shift: 10 ms
context_frames: 15   # ~150ms context window
scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
early_stopping: patience=7
```

**Loss function:** `L = alpha * KL(softened_teacher || softened_student) * T^2 + (1 - alpha) * BCE(student, hard_labels)`

---

### 6. Files and Artifacts

```
results/
├── teacher_baseline.json           # Step 2: CRDNN inference stats
├── energy_vad_baseline.json        # Step 3: Energy VAD stats
├── teacher_gt_eval.json            # Teacher vs GT evaluation
├── comprehensive_eval.json         # All models, all splits, GT + pseudo
├── tiny_cnn_T1.0_a0.7/            # Temperature sweep (pseudo labels)
│   ├── best_model.pt
│   ├── config.json
│   └── tensorboard/
├── tiny_cnn_T2.0_a0.7/
├── tiny_cnn_T4.0_a0.7/
├── tiny_cnn_T8.0_a0.7/
├── mlp_T4.0_a0.7/
├── tiny_transformer_T4.0_a0.7/
├── tiny_cnn_T4.0_a0.3_gt/         # Alpha sweep (GT labels)
├── tiny_cnn_T4.0_a0.5_gt/
├── tiny_cnn_T4.0_a0.7_gt/
├── tiny_cnn_T4.0_a0.9_gt/
├── mlp_T4.0_a0.3_gt/
├── mlp_T4.0_a0.5_gt/
├── mlp_T4.0_a0.7_gt/
└── mlp_T4.0_a0.9_gt/

data/soft_labels/                   # Teacher posteriors (Step 4)
├── train/
├── dev/
└── eval/
```

---

### 7. Ground-Truth Evaluation

Previous results (Sections 3-4) used thresholded teacher posteriors as pseudo ground-truth labels. We now evaluate all models against **actual ground-truth annotations** from LibriParty metadata, which provides per-speaker utterance timestamps for each session.

#### 7.1 Teacher vs Ground Truth

The teacher CRDNN's soft labels evaluated against actual ground-truth speech annotations:

| Split | F1 | Precision | Recall | Accuracy | DER | Speech Ratio (GT) | Speech Ratio (Teacher) |
|-------|-----|-----------|--------|----------|-----|-------------------|----------------------|
| **dev** | **0.9355** | 0.9114 | 0.9608 | 0.9522 | 0.0919 | 36.1% | 38.1% |
| **eval** | **0.9517** | 0.9777 | 0.9270 | 0.9597 | 0.0888 | 42.8% | 40.6% |

The teacher achieves **F1 > 0.93** on both splits, confirming it is a high-quality source for knowledge distillation. The eval set has higher speech ratio (42.8%) than dev (36.1%).

#### 7.2 Student Evaluation on Eval Set (GT Labels)

All models evaluated on the **eval split** with ground-truth annotations:

| Model | Params | Size | F1 (GT) | Prec (GT) | Recall (GT) | Acc (GT) | DER (GT) | CPU Latency |
|-------|--------|------|---------|-----------|-------------|----------|----------|-------------|
| **Teacher (CRDNN)** | 109,744 | 0.43 MB | 0.9517 | 0.9777 | 0.9270 | 0.9597 | 0.0888 | -- |
| **TinyTransformer** | 389,633 | 1.49 MB | **0.8533** | 0.8438 | **0.8629** | **0.8730** | **0.2565** | 2.2 ms |
| **MLP** | 81,921 | 0.31 MB | 0.8558 | **0.9561** | 0.7745 | 0.8883 | 0.2521 | 10.0 ms |
| **TinyCNN (T=2)** | 14,913 | 0.06 MB | 0.8178 | 0.9350 | 0.7267 | 0.8615 | 0.3110 | 1.2 ms |
| **TinyCNN (T=4)** | 14,913 | 0.06 MB | 0.8096 | 0.9424 | 0.7097 | 0.8572 | 0.3228 | 1.2 ms |
| **TinyCNN (T=1)** | 14,913 | 0.06 MB | 0.7971 | 0.9518 | 0.6857 | 0.8507 | 0.3403 | 1.1 ms |
| **TinyCNN (T=8)** | 14,913 | 0.06 MB | 0.8174 | 0.9383 | 0.7241 | 0.8616 | 0.3115 | 1.4 ms |

**Key observations:**

1. **All students achieve F1 > 0.80 on eval with GT labels**, significantly better than the dev-set pseudo-label results reported earlier.

2. **MLP achieves best F1 on eval (0.8558)** with extremely high precision (0.9561), edging out TinyTransformer (0.8533) which has best recall (0.8629).

3. **TinyCNN at 57 KB achieves F1=0.82** on eval — remarkably competitive given 7.4x compression vs teacher.

4. **Eval set is "easier" than dev**: All models score higher on eval than dev, likely because the eval set has higher speech ratio (42.8% vs 36.1%) and perhaps less overlap complexity.

5. **Teacher ceiling is F1=0.95** — students reach 85-90% of teacher quality, demonstrating effective knowledge transfer.

#### 7.3 Dev vs Eval Comparison (GT Labels)

| Model | dev F1 | eval F1 | dev DER | eval DER |
|-------|--------|---------|---------|----------|
| Teacher | 0.9355 | 0.9517 | 0.0919 | 0.0888 |
| TinyTransformer | 0.8185 | 0.8533 | 0.2666 | 0.2565 |
| MLP | 0.7813 | 0.8558 | 0.3373 | 0.2521 |
| TinyCNN (T=1) | 0.7826 | 0.7971 | 0.3417 | 0.3403 |
| TinyCNN (T=2) | 0.7758 | 0.8178 | 0.3475 | 0.3110 |
| TinyCNN (T=4) | 0.7760 | 0.8096 | 0.3490 | 0.3228 |
| TinyCNN (T=8) | 0.7680 | 0.8174 | 0.3604 | 0.3115 |

---

### 8. Alpha Sweep

Alpha (α) controls the balance between KD loss (soft teacher targets) and BCE loss (hard labels):

```
L = α * KL_div(softened_teacher, softened_student) * T² + (1-α) * BCE(student, hard_labels)
```

We sweep α ∈ {0.3, 0.5, 0.7, 0.9} with T=4.0, using **ground-truth annotations** as hard labels for BCE.

#### 8.1 TinyCNN Alpha Sweep (T=4, GT labels)

| Alpha | dev F1 (GT) | dev DER | eval F1 (GT) | eval DER | Precision (eval) | Recall (eval) |
|-------|-------------|---------|--------------|----------|-------------------|---------------|
| **α=0.3** | **0.7847** | **0.3374** | 0.8104 | 0.3219 | 0.9414 | 0.7118 |
| α=0.5 | 0.7833 | 0.3396 | 0.8119 | 0.3190 | 0.9369 | 0.7158 |
| α=0.7 | 0.7759 | 0.3486 | 0.8119 | 0.3194 | 0.9375 | 0.7154 |
| α=0.9 | 0.7694 | 0.3585 | 0.8092 | 0.3237 | 0.9375 | 0.7102 |

#### 8.2 MLP Alpha Sweep (T=4, GT labels)

| Alpha | dev F1 (GT) | dev DER | eval F1 (GT) | eval DER | Precision (eval) | Recall (eval) |
|-------|-------------|---------|--------------|----------|-------------------|---------------|
| **α=0.3** | **0.7949** | **0.3190** | 0.8502 | 0.2609 | 0.9520 | 0.7678 |
| α=0.5 | 0.7894 | 0.3297 | 0.8354 | 0.2834 | 0.9361 | 0.7546 |
| α=0.7 | 0.7863 | 0.3324 | 0.8445 | 0.2695 | 0.9423 | 0.7645 |
| α=0.9 | 0.7845 | 0.3341 | 0.8517 | 0.2585 | 0.9533 | 0.7697 |

#### 8.3 Alpha Sweep Analysis

**Key findings:**

1. **Lower alpha performs best on dev set** for both architectures. α=0.3 (more weight on hard GT labels) gives the highest dev F1 for both TinyCNN (0.7847) and MLP (0.7949).

2. **Eval set shows a different pattern**: MLP actually improves with higher α (0.8517 at α=0.9 vs 0.8502 at α=0.3), suggesting that the soft teacher knowledge generalizes better to unseen data, while hard labels improve fitting to the training distribution.

3. **TinyCNN is insensitive to alpha**: F1 varies only 0.001-0.003 on eval across all alpha values (0.809-0.812), confirming that model capacity — not loss weighting — is the bottleneck for this architecture.

4. **MLP benefits more from alpha tuning**: The F1 range on eval is larger (0.835-0.852), indicating that the larger model has enough capacity to exploit differences in the loss balance.

5. **Precision remains very high across all alpha values** (>0.93 for all models), while recall varies more. This means alpha primarily controls the false negative rate.

6. **Comparison with pseudo-label training**: The original MLP (α=0.7, pseudo labels) achieved eval F1=0.8558, slightly higher than MLP α=0.7 with GT labels (0.8445). This suggests the teacher's soft labels and thresholded labels are well-aligned for this task, and GT labels don't always outperform pseudo labels — the teacher is accurate enough (F1=0.95) that its pseudo labels are near-optimal supervision.

---

### 9. Reproducing the Results

```powershell
# Step 4: Generate soft labels (once)
python 03_train_kd.py --generate_labels --data_dir data/LibriParty/dataset

# Step 5a: Train all three students (T=4, alpha=0.7, pseudo labels — original)
python 03_train_kd.py --student tiny_cnn --temperature 4 --alpha 0.7 --epochs 30
python 03_train_kd.py --student mlp --temperature 4 --alpha 0.7 --epochs 30
python 03_train_kd.py --student tiny_transformer --temperature 4 --alpha 0.7 --epochs 30

# Step 5b: Temperature sweep (TinyCNN)
foreach ($T in 1, 2, 4, 8) {
    python 03_train_kd.py --student tiny_cnn --temperature $T --alpha 0.7 --epochs 30
}

# Step 6: Alpha sweep with ground-truth labels
.\05_run_alpha_sweep.ps1

# Step 7: Comprehensive evaluation (all models, all splits, GT + pseudo)
python 04_evaluate.py --eval_teacher --splits dev eval
```
