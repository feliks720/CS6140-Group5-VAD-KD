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

| Model | Param Ratio | Size Ratio | F1 Retention |
|-------|-------------|------------|--------------|
| TinyCNN | **7.4x smaller** | **7.5x smaller** | -- |
| MLP | 1.3x smaller | 1.4x smaller | -- |
| TinyTransformer | 3.5x larger | 3.5x larger | -- |

> Note: F1 retention cannot be computed because the teacher was not evaluated with the same frame-level metrics (it was run as an inference-only pipeline with speech ratio output). The student F1 scores are measured on the dev set against thresholded teacher soft labels.

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
├── tiny_cnn_T1.0_a0.7/            # Temperature sweep
│   ├── best_model.pt
│   ├── config.json
│   └── tensorboard/
├── tiny_cnn_T2.0_a0.7/
├── tiny_cnn_T4.0_a0.7/
├── tiny_cnn_T8.0_a0.7/
├── mlp_T4.0_a0.7/
└── tiny_transformer_T4.0_a0.7/

data/soft_labels/                   # Teacher posteriors (Step 4)
├── train/
├── dev/
└── eval/
```

---

### 7. Reproducing the Results

```powershell
# Step 4: Generate soft labels (once)
python 03_train_kd.py --generate_labels --data_dir data/LibriParty/dataset

# Step 5a: Train all three students (T=4, alpha=0.7)
python 03_train_kd.py --student tiny_cnn --temperature 4 --alpha 0.7 --epochs 30
python 03_train_kd.py --student mlp --temperature 4 --alpha 0.7 --epochs 30
python 03_train_kd.py --student tiny_transformer --temperature 4 --alpha 0.7 --epochs 30

# Step 5b: Temperature sweep (TinyCNN)
foreach ($T in 1, 2, 4, 8) {
    python 03_train_kd.py --student tiny_cnn --temperature $T --alpha 0.7 --epochs 30
}
```
