"""
utils/metrics.py
Evaluation metrics for VAD models.
"""

import time
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_frame_metrics(predictions, targets, threshold=0.5):
    """
    Compute frame-level VAD metrics.

    Args:
        predictions: Tensor or array of shape (num_frames,) with values in [0, 1]
        targets: Tensor or array of shape (num_frames,) with binary labels {0, 1}
        threshold: Threshold to binarize predictions

    Returns:
        dict with precision, recall, f1, accuracy
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    pred_binary = (predictions > threshold).astype(int)
    targets = targets.astype(int)

    # Ensure same length (truncate to shorter)
    min_len = min(len(pred_binary), len(targets))
    pred_binary = pred_binary[:min_len]
    targets = targets[:min_len]

    f1 = f1_score(targets, pred_binary, zero_division=0)
    precision = precision_score(targets, pred_binary, zero_division=0)
    recall = recall_score(targets, pred_binary, zero_division=0)
    accuracy = (pred_binary == targets).mean()

    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
    }


def compute_detection_error_rate(predictions, targets, threshold=0.5):
    """
    Compute Detection Error Rate (DER) = P(miss) + P(false alarm).

    Args:
        predictions, targets: arrays of frame-level decisions

    Returns:
        dict with miss_rate, fa_rate, der
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    pred_binary = (predictions > threshold).astype(int)
    targets = targets.astype(int)

    min_len = min(len(pred_binary), len(targets))
    pred_binary = pred_binary[:min_len]
    targets = targets[:min_len]

    # Miss: target=1, pred=0
    speech_frames = targets.sum()
    miss = ((targets == 1) & (pred_binary == 0)).sum()
    miss_rate = miss / speech_frames if speech_frames > 0 else 0.0

    # False alarm: target=0, pred=1
    nonspeech_frames = (1 - targets).sum()
    fa = ((targets == 0) & (pred_binary == 1)).sum()
    fa_rate = fa / nonspeech_frames if nonspeech_frames > 0 else 0.0

    der = miss_rate + fa_rate

    return {
        "miss_rate": round(float(miss_rate), 4),
        "fa_rate": round(float(fa_rate), 4),
        "der": round(float(der), 4),
    }


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def estimate_model_size_mb(model):
    """Estimate model size in MB (FP32)."""
    params = count_parameters(model)
    size_mb = params["total_params"] * 4 / (1024 ** 2)
    return round(size_mb, 3)


def measure_latency(model, input_tensor, n_runs=50, warmup=10, device="cpu"):
    """
    Measure average inference latency.

    Args:
        model: nn.Module
        input_tensor: example input tensor
        n_runs: number of timing runs
        warmup: number of warmup runs (not timed)
        device: 'cpu' or 'cuda'

    Returns:
        dict with avg_ms, std_ms, throughput_frames_per_sec
    """
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Timed runs
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    avg_ms = np.mean(times)
    std_ms = np.std(times)

    # Throughput: frames per second
    num_frames = input_tensor.shape[1]
    frames_per_sec = num_frames / (avg_ms / 1000)

    return {
        "avg_ms": round(avg_ms, 3),
        "std_ms": round(std_ms, 3),
        "throughput_fps": round(frames_per_sec, 0),
        "input_frames": num_frames,
    }


def compute_vad_metrics(predictions, targets, threshold=0.5):
    """Compute all VAD metrics in one call."""
    frame_metrics = compute_frame_metrics(predictions, targets, threshold)
    der_metrics = compute_detection_error_rate(predictions, targets, threshold)
    return {**frame_metrics, **der_metrics}


def print_metrics_table(metrics_dict, model_name="Model"):
    """Pretty-print metrics."""
    print(f"\n{'-' * 45}")
    print(f"  {model_name}")
    print(f"{'-' * 45}")
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            print(f"  {k:<20s} {v:.4f}")
        else:
            print(f"  {k:<20s} {v}")
    print(f"{'-' * 45}")
