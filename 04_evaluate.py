"""
04_evaluate.py
Comprehensive evaluation of all trained student models.

Evaluates each checkpoint on dev and eval splits using both:
  - Ground-truth annotations (from metadata JSON)
  - Pseudo labels (thresholded teacher posteriors)

Also evaluates the teacher model directly against ground-truth.

Usage:
    # Evaluate all models on eval set with GT labels
    python 04_evaluate.py

    # Evaluate specific model
    python 04_evaluate.py --models tiny_cnn_T4.0_a0.7

    # Evaluate on dev set only
    python 04_evaluate.py --splits dev

    # Evaluate teacher model against GT
    python 04_evaluate.py --eval_teacher
"""

import argparse
import os
import json
import glob

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm

from models.students import build_student
from utils.metrics import (
    compute_vad_metrics, estimate_model_size_mb,
    measure_latency, print_metrics_table
)

# Reuse annotation loading from 03_train_kd
from importlib import import_module


def load_metadata_annotations(metadata_dir, split):
    """Load ground-truth speech annotations from LibriParty metadata JSON."""
    json_path = os.path.join(metadata_dir, f"{split}.json")
    if not os.path.exists(json_path):
        print(f"  WARNING: {json_path} not found")
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
            if isinstance(utterances, list):
                for utt in utterances:
                    start = utt.get("start", utt.get("start_time", 0))
                    stop = utt.get("stop", utt.get("end", utt.get("end_time", 0)))
                    segments.append((float(start), float(stop)))
        segments.sort(key=lambda x: x[0])
        annotations[session_name] = segments

    return annotations


def segments_to_frame_labels(segments, num_frames, sample_rate=16000, hop_length=160):
    """Convert time-based speech segments to frame-level binary labels."""
    labels = torch.zeros(num_frames)
    frame_duration = hop_length / sample_rate

    for start_sec, stop_sec in segments:
        start_frame = int(start_sec / frame_duration)
        stop_frame = int(stop_sec / frame_duration)
        start_frame = max(0, min(start_frame, num_frames))
        stop_frame = max(0, min(stop_frame, num_frames))
        labels[start_frame:stop_frame] = 1.0

    return labels


def get_session_name(wav_path):
    """Extract session name from wav path."""
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    if basename.endswith("_mixture"):
        return basename[:-len("_mixture")]
    return basename


def find_trained_models(results_dir):
    """Find all trained model directories with best_model.pt."""
    models = {}
    for d in sorted(glob.glob(os.path.join(results_dir, "*"))):
        if os.path.isdir(d):
            ckpt = os.path.join(d, "best_model.pt")
            config_path = os.path.join(d, "config.json")
            if os.path.exists(ckpt):
                name = os.path.basename(d)
                config = {}
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        config = json.load(f)
                models[name] = {"checkpoint": ckpt, "config": config, "name": name}
    return models


def infer_student_type(model_name):
    """Infer student architecture name from experiment directory name."""
    if model_name.startswith("tiny_cnn"):
        return "tiny_cnn"
    elif model_name.startswith("mlp"):
        return "mlp"
    elif model_name.startswith("tiny_transformer"):
        return "tiny_transformer"
    else:
        return None


def evaluate_student(student, wav_files, soft_labels_dir, gt_annotations,
                     split, device, sample_rate=16000, n_mels=40,
                     hop_length=160, max_duration_s=30.0):
    """
    Evaluate a student model on a set of audio files.

    Returns:
        dict with 'gt' and 'pseudo' metrics
    """
    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=400, hop_length=hop_length, n_mels=n_mels,
    )
    max_samples = int(max_duration_s * sample_rate)

    gt_preds, gt_labels = [], []
    pseudo_preds, pseudo_labels = [], []

    student.eval()
    with torch.no_grad():
        for wav_path in tqdm(wav_files, desc=f"  {split}", leave=False):
            basename = os.path.splitext(os.path.basename(wav_path))[0]
            session_name = get_session_name(wav_path)

            # Load audio
            waveform, sr = torchaudio.load(wav_path)
            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

            # Features
            mel_spec = fbank(waveform)
            features = torch.log1p(mel_spec).squeeze(0).transpose(0, 1)  # (time, n_mels)
            num_frames = features.shape[0]

            # Student prediction
            features_batch = features.unsqueeze(0).to(device)  # (1, time, n_mels)
            logits = student(features_batch)
            probs = torch.sigmoid(logits.squeeze(0).squeeze(-1)).cpu().numpy()  # (time,)

            # Load teacher soft labels for pseudo evaluation
            soft_label_path = os.path.join(soft_labels_dir, split, f"{basename}.pt")
            if os.path.exists(soft_label_path):
                teacher_probs = torch.load(soft_label_path, weights_only=True)
                if len(teacher_probs) > num_frames:
                    teacher_probs = teacher_probs[:num_frames]
                elif len(teacher_probs) < num_frames:
                    teacher_probs = F.pad(teacher_probs, (0, num_frames - len(teacher_probs)))
                pseudo_hard = (teacher_probs > 0.5).float().cpu().numpy()
                pseudo_preds.append(probs[:len(pseudo_hard)])
                pseudo_labels.append(pseudo_hard)

            # Ground-truth labels
            if gt_annotations and session_name in gt_annotations:
                gt_frame_labels = segments_to_frame_labels(
                    gt_annotations[session_name], num_frames, sample_rate, hop_length
                ).cpu().numpy()
                gt_preds.append(probs[:len(gt_frame_labels)])
                gt_labels.append(gt_frame_labels)

    results = {}

    if gt_preds:
        all_gt_preds = np.concatenate(gt_preds)
        all_gt_labels = np.concatenate(gt_labels)
        results["gt"] = compute_vad_metrics(all_gt_preds, all_gt_labels)
        results["gt"]["num_frames"] = len(all_gt_labels)
        results["gt"]["speech_ratio_gt"] = round(float(all_gt_labels.mean()), 4)
        results["gt"]["speech_ratio_pred"] = round(float((all_gt_preds > 0.5).mean()), 4)

    if pseudo_preds:
        all_pseudo_preds = np.concatenate(pseudo_preds)
        all_pseudo_labels = np.concatenate(pseudo_labels)
        results["pseudo"] = compute_vad_metrics(all_pseudo_preds, all_pseudo_labels)

    return results


def evaluate_teacher_vs_gt(data_dir, soft_labels_dir, split, sample_rate=16000,
                           hop_length=160, max_duration_s=30.0):
    """
    Evaluate teacher soft labels against ground-truth annotations.
    This gives an upper-bound for student performance trained with teacher labels.
    """
    metadata_dir = os.path.join(data_dir, "metadata")
    gt_annotations = load_metadata_annotations(metadata_dir, split)
    if gt_annotations is None:
        print(f"  No GT annotations for {split}")
        return None

    wav_files = sorted(glob.glob(
        os.path.join(data_dir, split, "**", "*.wav"), recursive=True
    ))

    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=400, hop_length=hop_length, n_mels=40,
    )
    max_samples = int(max_duration_s * sample_rate)

    all_teacher_preds = []
    all_gt_labels = []

    for wav_path in tqdm(wav_files, desc=f"  Teacher vs GT ({split})"):
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        session_name = get_session_name(wav_path)

        # Load audio to get frame count
        waveform, sr = torchaudio.load(wav_path)
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        mel_spec = fbank(waveform)
        num_frames = mel_spec.shape[-1]

        # Teacher predictions
        soft_label_path = os.path.join(soft_labels_dir, split, f"{basename}.pt")
        if not os.path.exists(soft_label_path):
            continue

        teacher_probs = torch.load(soft_label_path, weights_only=True)
        if len(teacher_probs) > num_frames:
            teacher_probs = teacher_probs[:num_frames]
        elif len(teacher_probs) < num_frames:
            teacher_probs = F.pad(teacher_probs, (0, num_frames - len(teacher_probs)))

        # GT labels
        if session_name not in gt_annotations:
            continue

        gt_frame_labels = segments_to_frame_labels(
            gt_annotations[session_name], num_frames, sample_rate, hop_length
        )

        min_len = min(len(teacher_probs), len(gt_frame_labels))
        all_teacher_preds.append(teacher_probs[:min_len].cpu().numpy())
        all_gt_labels.append(gt_frame_labels[:min_len].cpu().numpy())

    if not all_teacher_preds:
        return None

    all_preds = np.concatenate(all_teacher_preds)
    all_labels = np.concatenate(all_gt_labels)
    metrics = compute_vad_metrics(all_preds, all_labels)
    metrics["num_frames"] = len(all_labels)
    metrics["speech_ratio_gt"] = round(float(all_labels.mean()), 4)
    metrics["speech_ratio_teacher"] = round(float((all_preds > 0.5).mean()), 4)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Comprehensive VAD model evaluation")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing trained model results")
    parser.add_argument("--data_dir", type=str, default="data/LibriParty/dataset")
    parser.add_argument("--soft_labels_dir", type=str, default="data/soft_labels")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Specific model names to evaluate (default: all)")
    parser.add_argument("--splits", type=str, nargs="*", default=["dev", "eval"],
                        help="Splits to evaluate on")
    parser.add_argument("--eval_teacher", action="store_true",
                        help="Also evaluate teacher model against GT")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 70)
    print("  Comprehensive VAD Evaluation")
    print("=" * 70)

    # Find all trained models
    all_models = find_trained_models(args.results_dir)
    if args.models:
        all_models = {k: v for k, v in all_models.items() if k in args.models}

    print(f"\nFound {len(all_models)} trained models:")
    for name in all_models:
        print(f"  - {name}")

    # Load GT annotations for each split
    metadata_dir = os.path.join(args.data_dir, "metadata")
    gt_annotations = {}
    for split in args.splits:
        gt_annotations[split] = load_metadata_annotations(metadata_dir, split)
        if gt_annotations[split]:
            print(f"\nGT annotations for {split}: {len(gt_annotations[split])} sessions")
        else:
            print(f"\nWARNING: No GT annotations found for {split}")

    # === Evaluate teacher vs GT ===
    if args.eval_teacher:
        print("\n" + "=" * 70)
        print("  Teacher (CRDNN) vs Ground Truth")
        print("=" * 70)
        teacher_results = {}
        for split in args.splits:
            metrics = evaluate_teacher_vs_gt(
                args.data_dir, args.soft_labels_dir, split
            )
            if metrics:
                teacher_results[split] = metrics
                print_metrics_table(metrics, f"Teacher ({split}, GT labels)")

        # Save teacher results
        out_path = os.path.join(args.results_dir, "teacher_gt_eval.json")
        with open(out_path, "w") as f:
            json.dump(teacher_results, f, indent=2)
        print(f"\nTeacher results saved to {out_path}")

    # === Evaluate each student model ===
    all_results = {}

    for model_name, model_info in all_models.items():
        print(f"\n{'=' * 70}")
        print(f"  Evaluating: {model_name}")
        print(f"{'=' * 70}")

        # Determine student type
        student_type = model_info["config"].get("student", infer_student_type(model_name))
        if student_type is None:
            print(f"  WARNING: Cannot determine student type for {model_name}, skipping")
            continue

        # Build and load model
        student = build_student(student_type)
        ckpt = torch.load(model_info["checkpoint"], map_location=args.device, weights_only=False)
        student.load_state_dict(ckpt["model_state_dict"])
        student.to(args.device)
        student.eval()

        training_f1 = ckpt.get("f1", "N/A")
        training_epoch = ckpt.get("epoch", "N/A")
        model_size = estimate_model_size_mb(student)
        total_params = sum(p.numel() for p in student.parameters())

        print(f"  Architecture: {student_type}")
        print(f"  Parameters: {total_params:,}")
        print(f"  Size: {model_size:.3f} MB")
        print(f"  Training best F1: {training_f1} (epoch {training_epoch})")

        model_results = {
            "model_name": model_name,
            "student_type": student_type,
            "total_params": total_params,
            "model_size_mb": model_size,
            "training_f1": training_f1,
            "training_epoch": training_epoch,
            "config": model_info["config"],
            "splits": {},
        }

        # Evaluate on each split
        for split in args.splits:
            wav_files = sorted(glob.glob(
                os.path.join(args.data_dir, split, "**", "*.wav"), recursive=True
            ))
            if not wav_files:
                print(f"  No wav files found for {split}")
                continue

            split_results = evaluate_student(
                student, wav_files, args.soft_labels_dir,
                gt_annotations.get(split), split, args.device
            )

            model_results["splits"][split] = split_results

            # Print results
            if "gt" in split_results:
                print_metrics_table(split_results["gt"],
                                    f"{model_name} ({split}, GT labels)")
            if "pseudo" in split_results:
                print_metrics_table(split_results["pseudo"],
                                    f"{model_name} ({split}, pseudo labels)")

        # Latency benchmark
        dummy_input = torch.randn(1, 1000, 40)
        latency = measure_latency(student, dummy_input, device="cpu")
        model_results["latency_cpu"] = latency
        print(f"\n  CPU Latency (1000 frames): {latency['avg_ms']:.1f} ms "
              f"({latency['throughput_fps']:.0f} fps)")

        all_results[model_name] = model_results

    # === Save all results ===
    out_path = os.path.join(args.results_dir, "comprehensive_eval.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"  All results saved to {out_path}")

    # === Print summary table ===
    print(f"\n{'=' * 70}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 70}")

    # Header
    header = f"{'Model':<30s} {'Params':>8s} {'Size':>7s}"
    for split in args.splits:
        header += f" | {split} F1 (GT)  {split} DER (GT)"
    print(header)
    print("-" * len(header))

    for model_name, res in all_results.items():
        row = f"{model_name:<30s} {res['total_params']:>8,d} {res['model_size_mb']:>6.3f}M"
        for split in args.splits:
            split_data = res.get("splits", {}).get(split, {})
            gt_data = split_data.get("gt", {})
            f1 = gt_data.get("f1", "--")
            der = gt_data.get("der", "--")
            if isinstance(f1, float):
                row += f" | {f1:>10.4f}  {der:>12.4f}"
            else:
                row += f" | {f1:>10s}  {der:>12s}"
        print(row)

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
