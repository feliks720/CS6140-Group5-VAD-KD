"""
models/students.py
Student architectures for Knowledge Distillation.

Three candidates:
  1. TinyCNN     - Lightweight 1D convolutions, no recurrence
  2. MLPVAD      - Pure feed-forward on FBANK features
  3. TinyTransformer - Small self-attention model

All models expect input shape: (batch, time, n_mels) where n_mels=40
All models output shape: (batch, time, 1) with sigmoid activation
"""

import torch
import torch.nn as nn
import math


class TinyCNN(nn.Module):
    """
    Lightweight 1D CNN for VAD.
    Architecture: 3 x (Conv1D -> BatchNorm -> ReLU -> Dropout) -> Global context -> FC -> Sigmoid

    Captures local spectro-temporal patterns without any recurrence.
    """

    def __init__(self, n_mels=40, hidden_channels=32, dropout=0.1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Layer 1: capture local spectral patterns
            nn.Conv1d(n_mels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2: wider receptive field
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3: further abstraction
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_layer = nn.Sequential(
            nn.Conv1d(hidden_channels, 1, kernel_size=1),  # pointwise conv
        )

    def forward(self, x):
        """
        Args:
            x: (batch, time, n_mels)
        Returns:
            logits: (batch, time, 1) -- raw logits (no sigmoid)
        """
        # (batch, time, n_mels) -> (batch, n_mels, time) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.output_layer(x)
        # (batch, 1, time) -> (batch, time, 1)
        x = x.transpose(1, 2)
        return x


class MLPVAD(nn.Module):
    """
    Pure feed-forward MLP for VAD.
    Processes each frame independently (+ small context window via concatenation).

    Uses a context window of `context` frames on each side, concatenated
    into a single vector before the MLP.
    """

    def __init__(self, n_mels=40, context=5, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.context = context
        input_dim = n_mels * (2 * context + 1)  # concat left + center + right

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, time, n_mels)
        Returns:
            logits: (batch, time, 1)
        """
        batch, time_steps, n_mels = x.shape

        # Pad for context window
        x_padded = torch.nn.functional.pad(x, (0, 0, self.context, self.context), mode="constant", value=0)

        # Build context frames: (batch, time, n_mels * (2*context+1))
        frames = []
        for t in range(time_steps):
            window = x_padded[:, t : t + 2 * self.context + 1, :]  # (batch, window, n_mels)
            frames.append(window.reshape(batch, -1))  # (batch, window * n_mels)

        x_context = torch.stack(frames, dim=1)  # (batch, time, input_dim)

        logits = self.mlp(x_context)  # (batch, time, 1)
        return logits


class TinyTransformer(nn.Module):
    """
    Small Transformer for VAD.
    Uses self-attention to capture temporal dependencies without recurrence.
    Parallelizable and efficient for moderate sequence lengths.
    """

    def __init__(self, n_mels=40, d_model=64, nhead=2, num_layers=2,
                 dim_feedforward=128, dropout=0.1, max_len=5000):
        super().__init__()

        self.input_proj = nn.Linear(n_mels, d_model)

        # Sinusoidal positional encoding
        self.pos_encoding = self._create_pos_encoding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, 1)

    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # (1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, time, n_mels)
        Returns:
            logits: (batch, time, 1)
        """
        x = self.input_proj(x)  # (batch, time, d_model)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)  # (batch, time, d_model)
        logits = self.output_layer(x)  # (batch, time, 1)
        return logits


# ======================== Factory ========================

STUDENT_REGISTRY = {
    "tiny_cnn": TinyCNN,
    "mlp": MLPVAD,
    "tiny_transformer": TinyTransformer,
}


def build_student(name: str, **kwargs) -> nn.Module:
    """
    Build a student model by name.

    Args:
        name: One of 'tiny_cnn', 'mlp', 'tiny_transformer'
        **kwargs: Override default hyperparameters

    Returns:
        nn.Module
    """
    if name not in STUDENT_REGISTRY:
        raise ValueError(f"Unknown student '{name}'. Choose from {list(STUDENT_REGISTRY.keys())}")

    return STUDENT_REGISTRY[name](**kwargs)


def print_model_summary(model: nn.Module, name: str = ""):
    """Print parameter count and model structure summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 50}")
    print(f"Model: {name or model.__class__.__name__}")
    print(f"{'=' * 50}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Size (approx):        {total_params * 4 / 1024:.1f} KB (FP32)")
    print(f"  Size (approx):        {total_params * 2 / 1024:.1f} KB (FP16)")
    print(f"{'=' * 50}")

    return total_params


if __name__ == "__main__":
    """Quick test: build each student and verify shapes."""
    batch, time_steps, n_mels = 4, 100, 40
    x = torch.randn(batch, time_steps, n_mels)

    for name in STUDENT_REGISTRY:
        model = build_student(name)
        params = print_model_summary(model, name)

        out = model(x)
        print(f"  Input:  {tuple(x.shape)}")
        print(f"  Output: {tuple(out.shape)}")
        assert out.shape == (batch, time_steps, 1), f"Shape mismatch for {name}!"
        print(f"  Shape check: PASSED\n")
