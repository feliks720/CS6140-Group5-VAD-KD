"""
02_energy_vad_baseline.py
Energy-based Voice Activity Detection baseline.
This is the "algorithmic method" required for Part 1 (individual grade).

Implements a simple energy threshold VAD:
  1. Compute short-time energy (STE) of the signal
  2. Apply threshold to classify frames as speech/non-speech
  3. Post-process (merge short gaps, remove short segments)

Usage:
    python 02_energy_vad_baseline.py --audio_dir data/LibriParty/dataset/test
    python 02_energy_vad_baseline.py --audio_file example.wav --plot
"""

import argparse
import os
import glob
import json
import time

import torch
import torchaudio
import numpy as np

from utils.metrics import compute_frame_metrics


class EnergyVAD:
    """
    Energy-based Voice Activity Detection.

    Algorithm:
      1. Compute frame-level short-time energy (STE)
      2. Normalize energy to [0, 1]
      3. Apply activation/deactivation thresholds (hysteresis)
      4. Post-process segments (merge close, remove short)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        activation_th: float = 0.3,
        deactivation_th: float = 0.15,
        min_speech_s: float = 0.25,
        min_silence_s: float = 0.25,
    ):
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.frame_shift = int(sample_rate * frame_shift_ms / 1000)
        self.activation_th = activation_th
        self.deactivation_th = deactivation_th
        self.min_speech_frames = int(min_speech_s * 1000 / frame_shift_ms)
        self.min_silence_frames = int(min_silence_s * 1000 / frame_shift_ms)

    def compute_energy(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute frame-level short-time energy."""
        # waveform: (1, num_samples) or (num_samples,)
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        # Frame the signal
        num_frames = (waveform.shape[0] - self.frame_length) // self.frame_shift + 1
        energy = torch.zeros(num_frames)

        for i in range(num_frames):
            start = i * self.frame_shift
            frame = waveform[start : start + self.frame_length]
            energy[i] = (frame ** 2).mean()

        # Normalize to [0, 1] using log energy
        log_energy = torch.log1p(energy)
        if log_energy.max() > log_energy.min():
            log_energy = (log_energy - log_energy.min()) / (log_energy.max() - log_energy.min())

        return log_energy

    def apply_hysteresis(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Apply hysteresis thresholding.
        Speech starts when energy > activation_th,
        speech ends when energy < deactivation_th.
        """
        decisions = torch.zeros_like(energy)
        is_speech = False

        for i in range(len(energy)):
            if is_speech:
                if energy[i] < self.deactivation_th:
                    is_speech = False
                else:
                    decisions[i] = 1.0
            else:
                if energy[i] > self.activation_th:
                    is_speech = True
                    decisions[i] = 1.0

        return decisions

    def post_process(self, decisions: torch.Tensor) -> torch.Tensor:
        """Merge short silence gaps and remove short speech segments."""
        decisions = decisions.clone()

        # Merge short silence gaps (fill in small non-speech holes)
        in_speech = False
        silence_count = 0
        speech_start = 0

        for i in range(len(decisions)):
            if decisions[i] == 1.0:
                if not in_speech:
                    if silence_count > 0 and silence_count < self.min_silence_frames:
                        # Fill the gap
                        decisions[speech_start:i] = 1.0
                    speech_start = i
                in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    in_speech = False
                    silence_count = 0
                silence_count += 1

        # Remove short speech segments
        in_speech = False
        seg_start = 0

        for i in range(len(decisions)):
            if decisions[i] == 1.0 and not in_speech:
                seg_start = i
                in_speech = True
            elif decisions[i] == 0.0 and in_speech:
                seg_len = i - seg_start
                if seg_len < self.min_speech_frames:
                    decisions[seg_start:i] = 0.0
                in_speech = False

        # Handle trailing segment
        if in_speech:
            seg_len = len(decisions) - seg_start
            if seg_len < self.min_speech_frames:
                decisions[seg_start:] = 0.0

        return decisions

    def __call__(self, audio_file: str) -> dict:
        """
        Full energy VAD pipeline.
        Returns dict with 'energy', 'decisions', 'decisions_postprocessed'.
        """
        waveform, sr = torchaudio.load(audio_file)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        energy = self.compute_energy(waveform)
        decisions = self.apply_hysteresis(energy)
        decisions_pp = self.post_process(decisions)

        return {
            "energy": energy,
            "decisions_raw": decisions,
            "decisions": decisions_pp,
            "num_frames": len(energy),
            "speech_ratio": decisions_pp.mean().item(),
        }


def evaluate_directory(audio_dir, energy_vad):
    """Evaluate energy VAD on all wav files in a directory."""
    wav_files = sorted(glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))

    if not wav_files:
        print(f"No .wav files found in {audio_dir}")
        return

    print(f"\nFound {len(wav_files)} audio files")
    print("=" * 60)

    total_audio_dur = 0.0
    total_infer_time = 0.0
    results = []

    for i, wav_file in enumerate(wav_files):
        waveform, sr = torchaudio.load(wav_file)
        audio_dur = waveform.shape[1] / sr
        total_audio_dur += audio_dur

        start_time = time.time()
        output = energy_vad(wav_file)
        infer_time = time.time() - start_time
        total_infer_time += infer_time

        result = {
            "file": os.path.basename(wav_file),
            "duration_s": round(audio_dur, 2),
            "inference_time_s": round(infer_time, 4),
            "rtf": round(infer_time / audio_dur, 4),
            "speech_ratio": round(output["speech_ratio"], 3),
        }
        results.append(result)

        if i < 5 or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(wav_files)}] {result['file']} | "
                  f"dur={result['duration_s']:.1f}s | "
                  f"RTF={result['rtf']:.4f} | "
                  f"speech={result['speech_ratio']:.1%}")

    avg_rtf = total_infer_time / total_audio_dur if total_audio_dur > 0 else 0

    print("\n" + "=" * 60)
    print("ENERGY VAD BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Files evaluated:      {len(wav_files)}")
    print(f"  Total audio:          {total_audio_dur:.1f}s ({total_audio_dur/60:.1f} min)")
    print(f"  Total inference time: {total_infer_time:.2f}s")
    print(f"  Average RTF:          {avg_rtf:.6f}")
    print(f"  Avg speech ratio:     {np.mean([r['speech_ratio'] for r in results]):.1%}")
    print(f"  Parameters:           0 (algorithm-based, no learned params)")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/energy_vad_baseline.json", "w") as f:
        json.dump({
            "model": "Energy VAD (hysteresis thresholding)",
            "params": {
                "activation_th": energy_vad.activation_th,
                "deactivation_th": energy_vad.deactivation_th,
                "frame_length_ms": energy_vad.frame_length * 1000 / energy_vad.sample_rate,
                "frame_shift_ms": energy_vad.frame_shift * 1000 / energy_vad.sample_rate,
            },
            "avg_rtf": round(avg_rtf, 6),
            "files": results,
        }, f, indent=2)
    print(f"\nResults saved to results/energy_vad_baseline.json")


def demo_with_plot(audio_file, energy_vad):
    """Run energy VAD on a single file and plot the results."""
    import matplotlib.pyplot as plt

    output = energy_vad(audio_file)
    waveform, sr = torchaudio.load(audio_file)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        sr = 16000

    waveform = waveform.squeeze().numpy()
    time_axis = np.arange(len(waveform)) / sr

    energy = output["energy"].numpy()
    decisions = output["decisions"].numpy()
    frame_time = np.arange(len(energy)) * (energy_vad.frame_shift / sr)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Waveform
    axes[0].plot(time_axis, waveform, linewidth=0.3, color="steelblue")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Energy VAD - {os.path.basename(audio_file)}")

    # Energy
    axes[1].plot(frame_time, energy, color="darkorange", linewidth=0.8)
    axes[1].axhline(y=energy_vad.activation_th, color="red", linestyle="--",
                     linewidth=0.8, label=f"activation={energy_vad.activation_th}")
    axes[1].axhline(y=energy_vad.deactivation_th, color="green", linestyle="--",
                     linewidth=0.8, label=f"deactivation={energy_vad.deactivation_th}")
    axes[1].set_ylabel("Normalized Energy")
    axes[1].legend(fontsize=8)

    # VAD decisions
    axes[2].fill_between(frame_time, decisions, alpha=0.4, color="green", label="Speech")
    axes[2].set_ylabel("VAD Decision")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plot_path = "results/energy_vad_demo.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Energy-based VAD Baseline")
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--audio_file", type=str, default=None)
    parser.add_argument("--plot", action="store_true", help="Plot results for single file")
    parser.add_argument("--activation_th", type=float, default=0.3)
    parser.add_argument("--deactivation_th", type=float, default=0.15)
    args = parser.parse_args()

    energy_vad = EnergyVAD(
        activation_th=args.activation_th,
        deactivation_th=args.deactivation_th,
    )

    if args.audio_file:
        if args.plot:
            demo_with_plot(args.audio_file, energy_vad)
        else:
            output = energy_vad(args.audio_file)
            print(f"Speech ratio: {output['speech_ratio']:.1%}")
            print(f"Frames: {output['num_frames']}")
    elif args.audio_dir:
        evaluate_directory(args.audio_dir, energy_vad)
    else:
        print("Provide --audio_dir or --audio_file")


if __name__ == "__main__":
    main()
