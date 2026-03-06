# VAD Knowledge Distillation Sub-Project
## CS6140 - Machine Learning | Northeastern University
## Group 5

### Project File Structure
```
vad-kd-project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup_data.ps1              # Download datasets (Windows PowerShell)
├── setup_data.sh               # Download datasets (Linux/Mac)
├── 01_baseline_inference.py    # Load pretrained CRDNN teacher & evaluate
├── 02_energy_vad_baseline.py   # Energy-based VAD baseline (Part 1 comparison)
├── 03_train_kd.py              # Knowledge Distillation training pipeline
├── models/
│   └── students.py             # Student architecture definitions
├── utils/
│   ├── dataset.py              # LibriParty data loading utilities
│   └── metrics.py              # Evaluation metrics (F1, DER, latency)
└── configs/
    └── kd_config.yaml          # Hyperparameter configuration
```

---

## Step 0: Environment Setup (Windows + RTX 3090)

```powershell
# Create venv (Python 3.12 recommended — 3.13/3.14 have SpeechBrain compatibility issues)
py -3.12 -m venv .venv
.venv\Scripts\activate

# Install PyTorch 2.5.1 with CUDA 12.4 (verified working with SpeechBrain 1.0.3)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

# Install SpeechBrain (pin huggingface_hub to avoid deprecated API errors)
pip install speechbrain==1.0.3 "huggingface_hub<0.26"

# Install remaining dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}  GPU: {torch.cuda.get_device_name(0)}')"
# Expected: PyTorch 2.5.1+cu124  CUDA: True  GPU: NVIDIA GeForce RTX 3090

# Verify SpeechBrain
python -c "from speechbrain.inference.VAD import VAD; print('SpeechBrain OK')"
```

## Step 1: Download Datasets

```powershell
# If PowerShell blocks the script, run first:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Standard setup: LibriParty only (~1.4 GB) — sufficient for KD experiments
.\setup_data.ps1

# Full setup with MUSAN (~10 GB extra, for noise augmentation)
.\setup_data.ps1 -Full
```

LibriParty directory structure after download:
```
data/LibriParty/dataset/
├── train/          # Training audio files
├── dev/            # Validation audio files
├── eval/           # Test audio files (note: "eval" not "test")
└── metadata/       # Ground truth annotations
```

## Step 2: Run Pretrained Teacher Baseline

```powershell
python 01_baseline_inference.py --audio_dir data/LibriParty/dataset/eval
```

Loads the pretrained CRDNN from HuggingFace and evaluates on LibriParty eval set.
Results saved to `results/teacher_baseline.json`.

**Baseline results (50 files, 240 min audio):**
- RTF = 0.005 (200x real-time on RTX 3090)
- Avg speech ratio: 45.6%

## Step 3: Energy VAD Baseline (Part 1 - Individual)

```powershell
# Default thresholds
python 02_energy_vad_baseline.py --audio_dir data/LibriParty/dataset/eval

# Tuned thresholds (lower activation for noisy audio)
python 02_energy_vad_baseline.py --audio_dir data/LibriParty/dataset/eval --activation_th 0.1 --deactivation_th 0.05
```

Compares energy-based VAD against the neural CRDNN baseline.
Results saved to `results/energy_vad_baseline.json`.

**Baseline results:** Energy VAD detects only 7.4% speech (vs teacher's 45.6%), demonstrating the need for neural VAD in noisy conditions.

## Step 4: Generate Teacher Soft Labels (required before KD training)

```powershell
python 03_train_kd.py --generate_labels --data_dir data/LibriParty/dataset
```

Runs the teacher CRDNN on all train/dev/eval audio and saves frame-level posteriors to `data/soft_labels/`. This only needs to be done once.

## Step 5: Train Student via Knowledge Distillation

```powershell
# Train Tiny CNN student
python 03_train_kd.py --student tiny_cnn --temperature 4 --alpha 0.7

# Train MLP student
python 03_train_kd.py --student mlp --temperature 4 --alpha 0.7

# Train Tiny Transformer student
python 03_train_kd.py --student tiny_transformer --temperature 4 --alpha 0.7

# Sweep temperatures
foreach ($T in 1, 2, 4, 8) {
    python 03_train_kd.py --student tiny_cnn --temperature $T --alpha 0.7
}
```

## Step 6: Evaluate All Models

```powershell
python 03_train_kd.py --eval_only --student tiny_cnn --checkpoint results/tiny_cnn_T4_a0.7/best_model.pt
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `list_audio_backends` error | Use Python 3.12, not 3.13/3.14 |
| `use_auth_token` TypeError | `pip install "huggingface_hub<0.26"` |
| Symlink permission error on Windows | Already handled via `LocalStrategy.COPY` in code |
| `No module named 'k2'` | Use `speechbrain==1.0.3` from PyPI, not develop branch |
| No .wav files found in test/ | LibriParty uses `eval/` not `test/` |
