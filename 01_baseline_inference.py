"""
01_baseline_inference.py
Load pretrained SpeechBrain CRDNN-VAD (teacher model) and evaluate on LibriParty test set.
This establishes the teacher baseline for Knowledge Distillation.

Usage:
    python 01_baseline_inference.py --audio_dir data/LibriParty/dataset/test
    python 01_baseline_inference.py --audio_file path/to/single/file.wav
"""

import argparse
import os
import glob
import time
import json

import torch
import torchaudio
import numpy as np
from speechbrain.inference.VAD import VAD

from utils.metrics import compute_vad_metrics, count_parameters, print_metrics_table


def load_teacher(device="cpu"):
    """Load the pretrained CRDNN-VAD model from HuggingFace."""
    print("Loading pretrained CRDNN teacher from HuggingFace...")
    from speechbrain.utils.fetching import LocalStrategy
    vad = VAD.from_hparams(
        source="speechbrain/vad-crdnn-libriparty",
        savedir="pretrained_models/vad-crdnn-libriparty",
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )
    print("Teacher model loaded successfully.")
    return vad


def get_frame_level_predictions(vad_model, audio_file):
    """
    Get frame-level speech probabilities from the teacher model.
    Returns:
        probs: Tensor of shape (num_frames,) with speech posteriors [0, 1]
    """
    # Step 1: Get frame-level posteriors
    prob_chunks = vad_model.get_speech_prob_file(audio_file)
    return prob_chunks.squeeze()


def get_boundaries_from_model(vad_model, audio_file, threshold=0.5):
    """
    Full inference pipeline: posteriors -> threshold -> post-processing -> boundaries.
    Returns boundaries as a list of (start_sec, end_sec) tuples.
    """
    prob_chunks = vad_model.get_speech_prob_file(audio_file)
    prob_th = vad_model.apply_threshold(prob_chunks).float()
    boundaries = vad_model.get_boundaries(prob_th)

    # Post-processing
    boundaries = vad_model.merge_close_segments(boundaries, close_th=0.250)
    boundaries = vad_model.remove_short_segments(boundaries, len_th=0.250)

    return boundaries


def evaluate_on_directory(vad_model, audio_dir, device="cpu"):
    """Evaluate the teacher on all wav files in a directory."""
    wav_files = sorted(glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))

    if not wav_files:
        print(f"No .wav files found in {audio_dir}")
        print("Hint: Check that LibriParty is extracted correctly.")
        print("Expected structure: data/LibriParty/dataset/test/*.wav")
        return

    print(f"\nFound {len(wav_files)} audio files in {audio_dir}")
    print("=" * 60)

    total_audio_duration = 0.0
    total_inference_time = 0.0
    all_results = []

    for i, wav_file in enumerate(wav_files):
        # Load audio to get duration
        waveform, sr = torchaudio.load(wav_file)
        audio_duration = waveform.shape[1] / sr
        total_audio_duration += audio_duration

        # Inference with timing
        start_time = time.time()
        probs = get_frame_level_predictions(vad_model, wav_file)
        boundaries = get_boundaries_from_model(vad_model, wav_file)
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        # Speech ratio
        speech_ratio = probs.mean().item()

        result = {
            "file": os.path.basename(wav_file),
            "duration_s": round(audio_duration, 2),
            "inference_time_s": round(inference_time, 4),
            "rtf": round(inference_time / audio_duration, 4),  # Real-Time Factor
            "speech_ratio": round(speech_ratio, 3),
            "num_segments": boundaries.shape[0] if boundaries.dim() > 0 else 0,
        }
        all_results.append(result)

        if i < 5 or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(wav_files)}] {result['file']} | "
                  f"dur={result['duration_s']:.1f}s | "
                  f"RTF={result['rtf']:.3f} | "
                  f"speech={result['speech_ratio']:.1%} | "
                  f"segments={result['num_segments']}")

    # Summary
    avg_rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0

    print("\n" + "=" * 60)
    print("TEACHER BASELINE SUMMARY (CRDNN)")
    print("=" * 60)
    print(f"  Files evaluated:      {len(wav_files)}")
    print(f"  Total audio:          {total_audio_duration:.1f}s ({total_audio_duration/60:.1f} min)")
    print(f"  Total inference time: {total_inference_time:.2f}s")
    print(f"  Average RTF:          {avg_rtf:.4f} (< 1.0 means real-time)")
    print(f"  Avg speech ratio:     {np.mean([r['speech_ratio'] for r in all_results]):.1%}")
    print(f"  Device:               {device}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/teacher_baseline.json", "w") as f:
        json.dump({
            "model": "CRDNN (speechbrain/vad-crdnn-libriparty)",
            "device": device,
            "total_audio_s": round(total_audio_duration, 2),
            "total_inference_s": round(total_inference_time, 4),
            "avg_rtf": round(avg_rtf, 4),
            "files": all_results,
        }, f, indent=2)
    print(f"\nResults saved to results/teacher_baseline.json")


def demo_single_file(vad_model, audio_file):
    """Demo inference on a single audio file with visualization."""
    print(f"\nProcessing: {audio_file}")
    print("-" * 40)

    # Get predictions
    probs = get_frame_level_predictions(vad_model, audio_file)
    boundaries = get_boundaries_from_model(vad_model, audio_file)

    # Display boundaries
    print("\nDetected speech segments:")
    if boundaries.dim() > 0 and boundaries.shape[0] > 0:
        for idx in range(boundaries.shape[0]):
            start = boundaries[idx, 0].item()
            end = boundaries[idx, 1].item()
            print(f"  segment_{idx+1:03d}  {start:.2f}s - {end:.2f}s  "
                  f"(duration: {end - start:.2f}s)")
    else:
        print("  No speech segments detected.")

    # Frame-level stats
    print(f"\nFrame-level stats:")
    print(f"  Total frames:    {probs.shape[0]}")
    print(f"  Speech frames:   {(probs > 0.5).sum().item()} ({(probs > 0.5).float().mean().item():.1%})")
    print(f"  Mean posterior:  {probs.mean().item():.4f}")

    return probs, boundaries


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained CRDNN-VAD teacher")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Directory containing test wav files")
    parser.add_argument("--audio_file", type=str, default=None,
                        help="Single audio file for demo")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    vad_model = load_teacher(device=args.device)

    if args.audio_file:
        demo_single_file(vad_model, args.audio_file)
    elif args.audio_dir:
        evaluate_on_directory(vad_model, args.audio_dir, device=args.device)
    else:
        print("Provide --audio_dir or --audio_file")
        print("\nQuick test with pretrained model's example:")
        # SpeechBrain downloads an example file with the model
        example = "pretrained_models/vad-crdnn-libriparty/example_vad.wav"
        if os.path.exists(example):
            demo_single_file(vad_model, example)
        else:
            print("Run with --audio_dir data/LibriParty/dataset/test")


if __name__ == "__main__":
    main()
