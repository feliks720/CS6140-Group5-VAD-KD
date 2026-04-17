"""
03_train_kd.py
Knowledge Distillation training pipeline for VAD.

Teacher: Pretrained CRDNN from SpeechBrain (frozen)
Student: TinyCNN / MLPVAD / TinyTransformer

Loss = alpha * KL_div(soft_teacher, soft_student) + (1-alpha) * BCE(hard_label, student)

Usage:
    # Train
    python 03_train_kd.py --student tiny_cnn --temperature 4 --alpha 0.7

    # Sweep temperatures
    for T in 1 2 4 8; do
        python 03_train_kd.py --student tiny_cnn --temperature $T
    done

    # Evaluate only
    python 03_train_kd.py --eval_only --student tiny_cnn --checkpoint results/tiny_cnn_T4/best_model.pt
"""

import argparse
import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchaudio
import numpy as np
from tqdm import tqdm

from models.students import build_student, print_model_summary
from utils.metrics import (
    compute_vad_metrics, estimate_model_size_mb,
    measure_latency, print_metrics_table
)


# ======================== Teacher Wrapper ========================

class CRDNNTeacher(nn.Module):
    """
    Wrapper around SpeechBrain's pretrained CRDNN-VAD.
    Extracts frame-level posteriors (soft labels) for distillation.

    NOTE: The teacher is always in eval mode and frozen.
    We use the SpeechBrain inference interface to get posteriors.
    """

    def __init__(self, device="cpu"):
        super().__init__()
        from speechbrain.inference.VAD import VAD
        from speechbrain.utils.fetching import LocalStrategy
        self.vad = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="pretrained_models/vad-crdnn-libriparty",
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY,
        )
        self.device = device
        self.eval()

    @torch.no_grad()
    def get_posteriors_from_features(self, features):
        """
        Get speech posteriors from FBANK features.

        The pretrained CRDNN expects features computed by its own pipeline.
        For KD, we use a simpler approach: generate soft labels offline
        by running the teacher's full pipeline on audio files, then use
        those soft labels during student training.

        Args:
            features: (batch, time, n_mels) - not used directly by teacher

        Returns:
            posteriors: (batch, time, 1) speech probabilities
        """
        # This is a placeholder - in practice, you'll precompute teacher
        # soft labels offline (see generate_soft_labels below)
        raise NotImplementedError(
            "Use generate_soft_labels() to precompute teacher outputs, "
            "then load them during training."
        )

    @torch.no_grad()
    def get_posteriors_from_audio(self, audio_file):
        """
        Get frame-level speech posteriors from an audio file.
        This is the correct way to use the teacher.
        """
        prob_chunks = self.vad.get_speech_prob_file(audio_file)
        return prob_chunks.squeeze()  # (num_frames,)


def generate_soft_labels(teacher, audio_dir, output_dir, split="train"):
    """
    Pre-compute teacher soft labels for all audio files.
    This runs once and saves posteriors to disk for efficient training.

    Args:
        teacher: CRDNNTeacher instance
        audio_dir: Path to audio files (e.g., data/LibriParty/dataset/train)
        output_dir: Where to save soft labels
        split: 'train', 'dev', or 'test'
    """
    import glob

    audio_path = os.path.join(audio_dir, split)
    save_path = os.path.join(output_dir, split)
    os.makedirs(save_path, exist_ok=True)

    wav_files = sorted(glob.glob(os.path.join(audio_path, "**", "*.wav"), recursive=True))
    print(f"\nGenerating soft labels for {len(wav_files)} files ({split})...")

    for i, wav_file in enumerate(tqdm(wav_files)):
        basename = os.path.splitext(os.path.basename(wav_file))[0]
        out_file = os.path.join(save_path, f"{basename}.pt")

        if os.path.exists(out_file):
            continue

        posteriors = teacher.get_posteriors_from_audio(wav_file)
        torch.save(posteriors, out_file)

    print(f"Soft labels saved to {save_path}/")


# ======================== KD Loss ========================

class KDLoss(nn.Module):
    """
    Knowledge Distillation loss for binary VAD.

    L = alpha * KL_div(soft_teacher/T, soft_student/T) * T^2
      + (1 - alpha) * BCE(hard_label, student)

    For binary classification, we convert single-output logits to
    2-class distributions for KL divergence.
    """

    def __init__(self, alpha=0.7, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, student_logits, teacher_probs, hard_labels, mask=None):
        """
        Args:
            student_logits: (batch, time, 1) raw student output (before sigmoid)
            teacher_probs:  (batch, time) teacher posteriors in [0, 1]
            hard_labels:    (batch, time) binary ground truth {0, 1}
            mask:           (batch, time) binary mask for valid frames

        Returns:
            total_loss, kd_loss, bce_loss (for logging)
        """
        student_logits = student_logits.squeeze(-1)  # (batch, time)
        T = self.temperature

        # --- KD loss (soft targets) ---
        # Convert to 2-class log-probs for KL divergence
        # Teacher: prob -> [1-prob, prob] -> /T -> log_softmax
        teacher_2class = torch.stack([1 - teacher_probs, teacher_probs], dim=-1)  # (B, T_time, 2)
        teacher_soft = F.log_softmax(torch.log(teacher_2class.clamp(1e-7)) / T, dim=-1)

        # Student: logit -> [0, logit] (treat as 2-class logits) -> /T -> log_softmax
        student_2class = torch.stack(
            [torch.zeros_like(student_logits), student_logits], dim=-1
        )
        student_soft = F.log_softmax(student_2class / T, dim=-1)

        # KL divergence (sum over classes, mean over frames)
        kd_loss = F.kl_div(
            student_soft,
            teacher_soft.exp(),  # KL expects target as probs when log_target=False
            reduction="none",
        ).sum(dim=-1)  # (batch, time)

        kd_loss = kd_loss * (T ** 2)  # Scale by T^2

        # --- BCE loss (hard targets) ---
        bce_loss = self.bce(student_logits, hard_labels)  # (batch, time)

        # --- Apply mask ---
        if mask is not None:
            kd_loss = (kd_loss * mask).sum() / mask.sum().clamp(min=1)
            bce_loss = (bce_loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            kd_loss = kd_loss.mean()
            bce_loss = bce_loss.mean()

        # --- Combined loss ---
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * bce_loss

        return total_loss, kd_loss.item(), bce_loss.item()


# ======================== Training Loop ========================

class KDTrainer:
    """Training manager for Knowledge Distillation."""

    def __init__(self, student, optimizer, scheduler, kd_loss, device, save_dir):
        self.student = student.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.kd_loss = kd_loss
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))
        self.best_f1 = 0.0
        self.global_step = 0

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.student.train()
        total_loss = 0.0
        total_kd = 0.0
        total_bce = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for features, teacher_probs, hard_labels, lengths in pbar:
            features = features.to(self.device)
            teacher_probs = teacher_probs.to(self.device)
            hard_labels = hard_labels.to(self.device)

            # Create mask from lengths
            batch_size, max_len = features.shape[0], features.shape[1]
            mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1).to(self.device)
            mask = mask.float()

            # Forward pass
            student_logits = self.student(features)

            # Compute loss
            loss, kd_l, bce_l = self.kd_loss(
                student_logits, teacher_probs, hard_labels, mask
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_kd += kd_l
            total_bce += bce_l
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "kd": f"{kd_l:.4f}",
                "bce": f"{bce_l:.4f}",
            })

            # Log to tensorboard
            if self.global_step % 50 == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/kd_loss", kd_l, self.global_step)
                self.writer.add_scalar("train/bce_loss", bce_l, self.global_step)

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader, epoch):
        """Validate and compute metrics."""
        self.student.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for features, teacher_probs, hard_labels, lengths in dataloader:
            features = features.to(self.device)
            teacher_probs = teacher_probs.to(self.device)
            hard_labels = hard_labels.to(self.device)

            batch_size, max_len = features.shape[0], features.shape[1]
            mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1).to(self.device)
            mask = mask.float()

            student_logits = self.student(features)
            loss, _, _ = self.kd_loss(student_logits, teacher_probs, hard_labels, mask)
            total_loss += loss.item()
            num_batches += 1

            # Collect predictions for metrics
            probs = torch.sigmoid(student_logits.squeeze(-1))
            for i in range(batch_size):
                length = lengths[i].item()
                all_preds.append(probs[i, :length].cpu().numpy())
                all_labels.append(hard_labels[i, :length].cpu().numpy())

        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_vad_metrics(all_preds, all_labels)
        metrics["val_loss"] = round(total_loss / max(num_batches, 1), 4)

        # Log
        self.writer.add_scalar("val/loss", metrics["val_loss"], epoch)
        self.writer.add_scalar("val/f1", metrics["f1"], epoch)
        self.writer.add_scalar("val/precision", metrics["precision"], epoch)
        self.writer.add_scalar("val/recall", metrics["recall"], epoch)
        self.writer.add_scalar("val/der", metrics["der"], epoch)

        # Save best model
        if metrics["f1"] > self.best_f1:
            self.best_f1 = metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.student.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "f1": metrics["f1"],
                "metrics": metrics,
            }, os.path.join(self.save_dir, "best_model.pt"))
            print(f"  ** New best F1: {self.best_f1:.4f} (saved) **")

        return metrics

    def save_checkpoint(self, epoch):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_f1": self.best_f1,
            "global_step": self.global_step,
        }, os.path.join(self.save_dir, "last_checkpoint.pt"))


# ======================== KD Dataset ========================

def load_metadata_annotations(metadata_dir, split):
    """
    Load ground-truth speech annotations from LibriParty metadata JSON.

    Args:
        metadata_dir: Path to metadata/ directory (e.g., data/LibriParty/dataset/metadata)
        split: 'train', 'dev', or 'eval'

    Returns:
        dict mapping session_name -> list of (start_sec, stop_sec) speech segments
    """
    json_path = os.path.join(metadata_dir, f"{split}.json")
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r") as f:
        metadata = json.load(f)

    annotations = {}
    non_speech_keys = {"noises", "background"}

    for session_name, session_data in metadata.items():
        segments = []
        for key, utterances in session_data.items():
            if key in non_speech_keys:
                continue
            # key is a speaker ID, utterances is a list of dicts with start/stop
            if isinstance(utterances, list):
                for utt in utterances:
                    start = utt.get("start", utt.get("start_time", 0))
                    stop = utt.get("stop", utt.get("end", utt.get("end_time", 0)))
                    segments.append((float(start), float(stop)))
        # Sort by start time
        segments.sort(key=lambda x: x[0])
        annotations[session_name] = segments

    return annotations


def segments_to_frame_labels(segments, num_frames, sample_rate=16000, hop_length=160):
    """
    Convert time-based speech segments to frame-level binary labels.

    Args:
        segments: list of (start_sec, stop_sec) tuples
        num_frames: total number of feature frames
        sample_rate: audio sample rate
        hop_length: STFT hop length in samples

    Returns:
        Tensor of shape (num_frames,) with 1.0 for speech frames, 0.0 otherwise
    """
    labels = torch.zeros(num_frames)
    frame_duration = hop_length / sample_rate  # seconds per frame

    for start_sec, stop_sec in segments:
        start_frame = int(start_sec / frame_duration)
        stop_frame = int(stop_sec / frame_duration)
        start_frame = max(0, min(start_frame, num_frames))
        stop_frame = max(0, min(stop_frame, num_frames))
        labels[start_frame:stop_frame] = 1.0

    return labels


class KDDataset(torch.utils.data.Dataset):
    """
    Dataset that loads precomputed features, teacher soft labels, and hard labels.

    Expected directory structure:
        soft_labels_dir/{split}/{basename}.pt   (teacher posteriors)
        data_dir/{split}/*.wav                  (audio files)
        data_dir/metadata/{split}.json          (ground-truth annotations, optional)

    When use_gt_labels=True, hard labels come from ground-truth annotations
    instead of thresholded teacher posteriors.
    """

    def __init__(self, data_dir, soft_labels_dir, split="train",
                 n_mels=40, sample_rate=16000, max_duration_s=30.0,
                 use_gt_labels=False):
        self.split = split
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_s * sample_rate)
        self.hop_length = 160
        self.use_gt_labels = use_gt_labels

        import glob
        audio_path = os.path.join(data_dir, split)
        self.wav_files = sorted(glob.glob(
            os.path.join(audio_path, "**", "*.wav"), recursive=True
        ))

        self.soft_labels_dir = os.path.join(soft_labels_dir, split)

        # Load ground-truth annotations if requested
        self.gt_annotations = None
        if use_gt_labels:
            metadata_dir = os.path.join(data_dir, "metadata")
            self.gt_annotations = load_metadata_annotations(metadata_dir, split)
            if self.gt_annotations is not None:
                print(f"  Loaded GT annotations for {len(self.gt_annotations)} sessions")
            else:
                print(f"  WARNING: GT annotations not found at {metadata_dir}/{split}.json, "
                      f"falling back to thresholded teacher labels")

        # FBANK
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, hop_length=self.hop_length, n_mels=n_mels,
        )

        print(f"KDDataset ({split}): {len(self.wav_files)} files, "
              f"GT labels: {use_gt_labels and self.gt_annotations is not None}")

    def __len__(self):
        return len(self.wav_files)

    def _get_session_name(self, wav_path):
        """Extract session name from wav path (e.g., 'session_0' from '.../session_0/session_0_mixture.wav')."""
        basename = os.path.splitext(os.path.basename(wav_path))[0]  # session_0_mixture
        # Remove '_mixture' suffix if present
        if basename.endswith("_mixture"):
            return basename[:-len("_mixture")]
        return basename

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        basename = os.path.splitext(os.path.basename(wav_path))[0]

        # Load audio
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]

        # Features
        mel_spec = self.fbank(waveform)
        features = torch.log1p(mel_spec).squeeze(0).transpose(0, 1)  # (time, n_mels)
        num_frames = features.shape[0]

        # Teacher soft labels
        soft_label_path = os.path.join(self.soft_labels_dir, f"{basename}.pt")
        if os.path.exists(soft_label_path):
            teacher_probs = torch.load(soft_label_path, weights_only=True)
            # Align length
            if len(teacher_probs) > num_frames:
                teacher_probs = teacher_probs[:num_frames]
            elif len(teacher_probs) < num_frames:
                teacher_probs = F.pad(teacher_probs, (0, num_frames - len(teacher_probs)))
        else:
            # Fallback: use zeros (teacher not yet computed)
            teacher_probs = torch.zeros(num_frames)

        # Hard labels: ground-truth if available, else thresholded teacher
        if self.use_gt_labels and self.gt_annotations is not None:
            session_name = self._get_session_name(wav_path)
            if session_name in self.gt_annotations:
                hard_labels = segments_to_frame_labels(
                    self.gt_annotations[session_name],
                    num_frames,
                    self.sample_rate,
                    self.hop_length,
                )
            else:
                # Fallback for missing session
                hard_labels = (teacher_probs > 0.5).float()
        else:
            hard_labels = (teacher_probs > 0.5).float()

        return features, teacher_probs, hard_labels, num_frames


def collate_kd(batch):
    """Collate for KD dataset with variable lengths."""
    features, teacher_probs, hard_labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths)
    max_len = lengths.max().item()
    n_mels = features[0].shape[1]

    padded_features = torch.zeros(len(features), max_len, n_mels)
    padded_teacher = torch.zeros(len(features), max_len)
    padded_labels = torch.zeros(len(features), max_len)

    for i in range(len(features)):
        t = lengths[i].item()
        padded_features[i, :t, :] = features[i]
        padded_teacher[i, :t] = teacher_probs[i]
        padded_labels[i, :t] = hard_labels[i]

    return padded_features, padded_teacher, padded_labels, lengths


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for VAD")

    # Model
    parser.add_argument("--student", type=str, default="tiny_cnn",
                        choices=["tiny_cnn", "mlp", "tiny_transformer"])

    # KD hyperparameters
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    # Data
    parser.add_argument("--data_dir", type=str, default="data/LibriParty/dataset")
    parser.add_argument("--soft_labels_dir", type=str, default="data/soft_labels")

    # Ground truth & evaluation
    parser.add_argument("--use_gt_labels", action="store_true",
                        help="Use ground-truth annotations instead of thresholded teacher labels")
    parser.add_argument("--eval_split", type=str, default="dev",
                        choices=["dev", "eval"],
                        help="Which split to evaluate on (default: dev)")

    # Other
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--generate_labels", action="store_true",
                        help="Generate teacher soft labels before training")

    args = parser.parse_args()

    # Experiment name
    exp_name = f"{args.student}_T{args.temperature}_a{args.alpha}"
    if args.use_gt_labels and not args.eval_only:
        exp_name += "_gt"
    save_dir = os.path.join("results", exp_name)

    print("=" * 60)
    print(f"  VAD Knowledge Distillation")
    print(f"  Student: {args.student}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Alpha: {args.alpha}")
    print(f"  GT labels: {args.use_gt_labels}")
    print(f"  Eval split: {args.eval_split}")
    print(f"  Device: {args.device}")
    print(f"  Save dir: {save_dir}")
    print("=" * 60)

    # --- Step 0: Generate soft labels if needed ---
    if args.generate_labels:
        print("\n--- Generating teacher soft labels ---")
        teacher = CRDNNTeacher(device=args.device)
        for split in ["train", "dev", "eval"]:
            generate_soft_labels(teacher, args.data_dir, args.soft_labels_dir, split)
        print("Done! Now run without --generate_labels to train.")
        return

    # --- Build student ---
    student = build_student(args.student)
    total_params = print_model_summary(student, args.student)

    # --- Evaluation only ---
    if args.eval_only:
        assert args.checkpoint, "Provide --checkpoint for eval_only mode"
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        student.load_state_dict(ckpt["model_state_dict"])
        student.to(args.device)
        student.eval()

        print(f"\nLoaded checkpoint: {args.checkpoint}")
        print(f"Checkpoint F1 (training): {ckpt.get('f1', 'N/A')}")

        # Evaluate on specified split
        eval_dataset = KDDataset(args.data_dir, args.soft_labels_dir,
                                 split=args.eval_split,
                                 use_gt_labels=args.use_gt_labels)
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_kd, pin_memory=True,
        )

        # Run evaluation
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, teacher_probs, hard_labels, lengths in tqdm(eval_loader, desc=f"Eval ({args.eval_split})"):
                features = features.to(args.device)
                student_logits = student(features)
                probs = torch.sigmoid(student_logits.squeeze(-1))
                for i in range(features.shape[0]):
                    length = lengths[i].item()
                    all_preds.append(probs[i, :length].cpu().numpy())
                    all_labels.append(hard_labels[i, :length].numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_vad_metrics(all_preds, all_labels)

        label_type = "GT" if args.use_gt_labels else "pseudo"
        print_metrics_table(metrics, f"{args.student} ({args.eval_split}, {label_type} labels)")

        # Latency benchmark
        dummy_input = torch.randn(1, 1000, 40)  # ~10s of audio at 100fps
        latency = measure_latency(student, dummy_input, device="cpu")
        print(f"\nLatency (CPU, 1000 frames): {latency['avg_ms']:.1f} ms")
        print(f"Throughput: {latency['throughput_fps']:.0f} frames/sec")
        print(f"Model size: {estimate_model_size_mb(student):.3f} MB")

        # Save eval results
        eval_results = {
            "model": args.student,
            "checkpoint": args.checkpoint,
            "eval_split": args.eval_split,
            "label_type": label_type,
            "metrics": metrics,
            "latency": latency,
            "model_size_mb": estimate_model_size_mb(student),
        }
        eval_out = os.path.join(os.path.dirname(args.checkpoint),
                                f"eval_{args.eval_split}_{label_type}.json")
        with open(eval_out, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nResults saved to {eval_out}")
        return

    # --- Create dataloaders ---
    print("\nCreating datasets...")
    train_dataset = KDDataset(args.data_dir, args.soft_labels_dir, split="train",
                              use_gt_labels=args.use_gt_labels)
    val_dataset = KDDataset(args.data_dir, args.soft_labels_dir, split=args.eval_split,
                            use_gt_labels=args.use_gt_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_kd, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_kd, pin_memory=True,
    )

    # --- Setup training ---
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    kd_loss = KDLoss(alpha=args.alpha, temperature=args.temperature)

    trainer = KDTrainer(
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        kd_loss=kd_loss,
        device=args.device,
        save_dir=save_dir,
    )

    # --- Train ---
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader, epoch)

        if scheduler:
            scheduler.step()
            trainer.writer.add_scalar("train/lr",
                scheduler.get_last_lr()[0], epoch)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print_metrics_table(val_metrics, f"{args.student} (val)")

        trainer.save_checkpoint(epoch)

    # --- Final summary ---
    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best F1: {trainer.best_f1:.4f}")
    print(f"  Model saved: {save_dir}/best_model.pt")
    print(f"  Model size: {estimate_model_size_mb(student):.3f} MB")
    print(f"  Parameters: {total_params:,}")
    print("=" * 60)

    # Save experiment config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
