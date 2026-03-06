"""
utils/dataset.py
Dataset utilities for loading LibriParty VAD data.

LibriParty provides audio files with corresponding annotation files
that mark speech/non-speech regions. This module creates PyTorch
datasets that extract FBANK features and frame-level labels.
"""

import os
import glob
import json

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


class LibriPartyVADDataset(Dataset):
    """
    PyTorch Dataset for LibriParty VAD.

    Each sample returns:
        features: (time, n_mels) FBANK features
        labels:   (time,) binary labels (1=speech, 0=non-speech)
    """

    def __init__(
        self,
        data_dir,
        split="train",
        n_mels=40,
        sample_rate=16000,
        max_duration_s=30.0,
        frame_shift_ms=10.0,
    ):
        """
        Args:
            data_dir: Path to LibriParty/dataset/
            split: 'train', 'dev', or 'test'
            n_mels: Number of mel filterbank channels
            sample_rate: Expected sample rate
            max_duration_s: Maximum clip duration (for memory)
            frame_shift_ms: Frame shift in ms (for label alignment)
        """
        self.data_dir = os.path.join(data_dir, split)
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_s * sample_rate)
        self.frame_shift = int(sample_rate * frame_shift_ms / 1000)

        # Find all wav files
        self.wav_files = sorted(glob.glob(
            os.path.join(self.data_dir, "**", "*.wav"), recursive=True
        ))

        if len(self.wav_files) == 0:
            raise FileNotFoundError(
                f"No wav files found in {self.data_dir}. "
                f"Make sure LibriParty is extracted correctly."
            )

        # FBANK extractor
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,       # 25ms at 16kHz
            hop_length=160,  # 10ms at 16kHz
            n_mels=n_mels,
        )

        print(f"LibriPartyVADDataset: {split} split, {len(self.wav_files)} files")

    def __len__(self):
        return len(self.wav_files)

    def _load_labels(self, wav_path):
        """
        Load ground-truth labels for a wav file.
        LibriParty stores annotations as JSON files alongside the wav files.
        Returns frame-level binary labels.
        """
        # Try to find annotation file
        # LibriParty annotation format: same name with .json extension
        json_path = wav_path.replace(".wav", ".json")

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                annotation = json.load(f)
            return annotation

        # Alternative: look for .rttm or .lab files
        rttm_path = wav_path.replace(".wav", ".rttm")
        if os.path.exists(rttm_path):
            return self._parse_rttm(rttm_path)

        return None

    def _annotations_to_frame_labels(self, annotations, num_frames, total_samples):
        """
        Convert time-based annotations to frame-level binary labels.
        This needs to be adapted to LibriParty's specific annotation format.
        """
        labels = torch.zeros(num_frames)

        if annotations is None:
            # If no annotations found, return all zeros
            # (you'll need to adapt this to your annotation format)
            return labels

        # If annotations is a list of segments: [{"start": float, "end": float}, ...]
        if isinstance(annotations, list):
            for seg in annotations:
                start_frame = int(seg.get("start", 0) * self.sample_rate / self.frame_shift)
                end_frame = int(seg.get("end", 0) * self.sample_rate / self.frame_shift)
                start_frame = max(0, min(start_frame, num_frames))
                end_frame = max(0, min(end_frame, num_frames))
                labels[start_frame:end_frame] = 1.0

        return labels

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]

        # Load audio
        waveform, sr = torchaudio.load(wav_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Truncate if too long
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]

        # Compute FBANK features
        mel_spec = self.fbank(waveform)  # (1, n_mels, time)
        features = torch.log1p(mel_spec).squeeze(0).transpose(0, 1)  # (time, n_mels)
        num_frames = features.shape[0]

        # Load and align labels
        annotations = self._load_labels(wav_path)
        labels = self._annotations_to_frame_labels(annotations, num_frames, waveform.shape[1])

        return features, labels


def collate_vad(batch):
    """
    Collate function for variable-length VAD sequences.
    Pads to the longest sequence in the batch.

    Returns:
        features: (batch, max_time, n_mels)
        labels:   (batch, max_time)
        lengths:  (batch,) original lengths
    """
    features, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in features])

    max_len = lengths.max().item()
    n_mels = features[0].shape[1]

    padded_features = torch.zeros(len(features), max_len, n_mels)
    padded_labels = torch.zeros(len(features), max_len)

    for i, (feat, lab) in enumerate(zip(features, labels)):
        t = feat.shape[0]
        padded_features[i, :t, :] = feat
        padded_labels[i, :t] = lab

    return padded_features, padded_labels, lengths


def create_dataloader(data_dir, split, batch_size=16, num_workers=4, **kwargs):
    """Create a DataLoader for LibriParty VAD."""
    dataset = LibriPartyVADDataset(data_dir, split=split, **kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_vad,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    return loader
