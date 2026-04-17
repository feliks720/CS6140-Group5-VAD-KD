"""
06_visualize.py
Generate publication-quality figures for the VAD-KD experiment report.

Figures:
  1. Training curves (F1 & loss vs epoch) for all architectures
  2. Architecture comparison bar chart (eval GT metrics)
  3. Temperature sweep line plot (TinyCNN)
  4. Alpha sweep comparison (TinyCNN vs MLP)
  5. Precision-Recall trade-off scatter
  6. Model efficiency plot (F1 vs model size)

Usage:
    python 06_visualize.py                  # Generate all figures
    python 06_visualize.py --output_dir figs  # Custom output directory
"""

import argparse
import os
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "tiny_cnn": "#2196F3",
    "mlp": "#FF9800",
    "tiny_transformer": "#4CAF50",
    "teacher": "#E91E63",
    "energy_vad": "#9E9E9E",
}

LABELS = {
    "tiny_cnn": "TinyCNN (15K)",
    "mlp": "MLP (82K)",
    "tiny_transformer": "Transformer (390K)",
    "teacher": "Teacher CRDNN (110K)",
}


# ── Helpers ────────────────────────────────────────────────────────

def load_tb_scalars(tb_dir, tag):
    """Load scalar values from TensorBoard event files."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def load_comprehensive_eval(results_dir):
    path = os.path.join(results_dir, "comprehensive_eval.json")
    with open(path) as f:
        return json.load(f)


def load_teacher_eval(results_dir):
    path = os.path.join(results_dir, "teacher_gt_eval.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_fig(fig, output_dir, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 1: Training Curves ─────────────────────────────────────

def plot_training_curves(results_dir, output_dir):
    """F1 and loss vs epoch for each architecture (T=4, alpha=0.7, pseudo labels)."""
    models = {
        "tiny_cnn_T4.0_a0.7": ("TinyCNN", COLORS["tiny_cnn"]),
        "mlp_T4.0_a0.7": ("MLP", COLORS["mlp"]),
        "tiny_transformer_T4.0_a0.7": ("Transformer", COLORS["tiny_transformer"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, (label, color) in models.items():
        tb_dir = os.path.join(results_dir, model_name, "tensorboard")
        if not os.path.isdir(tb_dir):
            continue

        # F1
        steps, values = load_tb_scalars(tb_dir, "val/f1")
        if steps:
            # Deduplicate: keep last value per step
            step_val = {}
            for s, v in zip(steps, values):
                step_val[s] = v
            epochs = sorted(step_val.keys())
            f1s = [step_val[e] for e in epochs]
            axes[0].plot(epochs, f1s, "-o", color=color, label=label,
                         markersize=4, linewidth=1.5)

        # Loss
        steps, values = load_tb_scalars(tb_dir, "val/loss")
        if steps:
            step_val = {}
            for s, v in zip(steps, values):
                step_val[s] = v
            epochs = sorted(step_val.keys())
            losses = [step_val[e] for e in epochs]
            axes[1].plot(epochs, losses, "-o", color=color, label=label,
                         markersize=4, linewidth=1.5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation F1")
    axes[0].set_title("Validation F1 vs Epoch")
    axes[0].legend()
    axes[0].set_ylim(0.3, 0.85)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Validation Loss vs Epoch")
    axes[1].legend()

    fig.suptitle("Training Curves (T=4, α=0.7, pseudo labels)", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir, "fig1_training_curves.png")


# ── Figure 2: Architecture Comparison Bar Chart ───────────────────

def plot_architecture_comparison(results_dir, output_dir):
    """Bar chart comparing all architectures on eval set with GT labels."""
    data = load_comprehensive_eval(results_dir)
    teacher = load_teacher_eval(results_dir)

    models_to_plot = [
        ("Teacher", teacher["eval"] if teacher else {}, COLORS["teacher"]),
        ("TinyTransformer", data.get("tiny_transformer_T4.0_a0.7", {}).get("splits", {}).get("eval", {}).get("gt", {}), COLORS["tiny_transformer"]),
        ("MLP", data.get("mlp_T4.0_a0.7", {}).get("splits", {}).get("eval", {}).get("gt", {}), COLORS["mlp"]),
        ("TinyCNN", data.get("tiny_cnn_T4.0_a0.7", {}).get("splits", {}).get("eval", {}).get("gt", {}), COLORS["tiny_cnn"]),
    ]

    metrics = ["f1", "precision", "recall", "accuracy"]
    metric_labels = ["F1", "Precision", "Recall", "Accuracy"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.arange(len(models_to_plot)) - (len(models_to_plot) - 1) / 2

    for i, (name, m, color) in enumerate(models_to_plot):
        vals = [m.get(k, 0) for k in metrics]
        bars = ax.bar(x + offsets[i] * width, vals, width, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0.6, 1.02)
    ax.set_title("Architecture Comparison on Eval Set (GT Labels, T=4, α=0.7)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, output_dir, "fig2_architecture_comparison.png")


# ── Figure 3: Temperature Sweep ───────────────────────────────────

def plot_temperature_sweep(results_dir, output_dir):
    """Line plot of F1, precision, recall vs temperature for TinyCNN."""
    data = load_comprehensive_eval(results_dir)
    temps = [1.0, 2.0, 4.0, 8.0]
    model_keys = [f"tiny_cnn_T{t}_a0.7" for t in temps]

    f1_dev, f1_eval = [], []
    prec_eval, rec_eval = [], []

    for key in model_keys:
        m = data.get(key, {})
        dev_gt = m.get("splits", {}).get("dev", {}).get("gt", {})
        eval_gt = m.get("splits", {}).get("eval", {}).get("gt", {})
        f1_dev.append(dev_gt.get("f1", 0))
        f1_eval.append(eval_gt.get("f1", 0))
        prec_eval.append(eval_gt.get("precision", 0))
        rec_eval.append(eval_gt.get("recall", 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temps, f1_eval, "-s", color="#2196F3", label="F1 (eval)", linewidth=2, markersize=8)
    ax.plot(temps, f1_dev, "--s", color="#2196F3", alpha=0.5, label="F1 (dev)", linewidth=1.5, markersize=6)
    ax.plot(temps, prec_eval, "-^", color="#4CAF50", label="Precision (eval)", linewidth=1.5, markersize=7)
    ax.plot(temps, rec_eval, "-v", color="#FF5722", label="Recall (eval)", linewidth=1.5, markersize=7)

    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Score")
    ax.set_title("Temperature Sweep — TinyCNN (α=0.7, GT Labels)")
    ax.set_xticks(temps)
    ax.set_xticklabels([f"T={t:.0f}" for t in temps])
    ax.set_ylim(0.65, 1.0)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, output_dir, "fig3_temperature_sweep.png")


# ── Figure 4: Alpha Sweep ─────────────────────────────────────────

def plot_alpha_sweep(results_dir, output_dir):
    """Alpha sweep comparison for TinyCNN and MLP."""
    data = load_comprehensive_eval(results_dir)
    alphas = [0.3, 0.5, 0.7, 0.9]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (student, color, label) in zip(axes, [
        ("tiny_cnn", COLORS["tiny_cnn"], "TinyCNN"),
        ("mlp", COLORS["mlp"], "MLP"),
    ]):
        f1_dev, f1_eval, der_eval = [], [], []
        for a in alphas:
            key = f"{student}_T4.0_a{a}_gt"
            m = data.get(key, {})
            dev_gt = m.get("splits", {}).get("dev", {}).get("gt", {})
            eval_gt = m.get("splits", {}).get("eval", {}).get("gt", {})
            f1_dev.append(dev_gt.get("f1", 0))
            f1_eval.append(eval_gt.get("f1", 0))
            der_eval.append(eval_gt.get("der", 0))

        ax2 = ax.twinx()

        l1, = ax.plot(alphas, f1_eval, "-o", color=color, label="F1 (eval)", linewidth=2, markersize=8)
        l2, = ax.plot(alphas, f1_dev, "--o", color=color, alpha=0.5, label="F1 (dev)", linewidth=1.5, markersize=6)
        l3, = ax2.plot(alphas, der_eval, "-D", color="#E91E63", label="DER (eval)", linewidth=1.5, markersize=7)

        ax.set_xlabel("Alpha (α)")
        ax.set_ylabel("F1 Score", color=color)
        ax2.set_ylabel("DER", color="#E91E63")
        ax.set_title(f"Alpha Sweep — {label} (T=4, GT Labels)")
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"α={a}" for a in alphas])

        lines = [l1, l2, l3]
        labels_legend = [l.get_label() for l in lines]
        ax.legend(lines, labels_legend, loc="center left")

    fig.tight_layout()
    save_fig(fig, output_dir, "fig4_alpha_sweep.png")


# ── Figure 5: Precision-Recall Scatter ────────────────────────────

def plot_precision_recall(results_dir, output_dir):
    """Precision vs Recall scatter for all models on eval set."""
    data = load_comprehensive_eval(results_dir)
    teacher = load_teacher_eval(results_dir)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Teacher point
    if teacher and "eval" in teacher:
        t = teacher["eval"]
        ax.scatter(t["recall"], t["precision"], s=200, c=COLORS["teacher"],
                   marker="*", zorder=5, label="Teacher CRDNN")
        ax.annotate("Teacher", (t["recall"], t["precision"]),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)

    # Student points
    plotted_types = set()
    for model_name, model_data in data.items():
        eval_gt = model_data.get("splits", {}).get("eval", {}).get("gt", {})
        if not eval_gt:
            continue

        student_type = model_data.get("student_type", "")
        color = COLORS.get(student_type, "#666")

        # Label only first of each type
        label = LABELS.get(student_type, student_type) if student_type not in plotted_types else None
        plotted_types.add(student_type)

        marker = {"tiny_cnn": "o", "mlp": "s", "tiny_transformer": "D"}.get(student_type, "o")

        ax.scatter(eval_gt["recall"], eval_gt["precision"], s=80, c=color,
                   marker=marker, alpha=0.8, label=label, zorder=3)

        # Annotate with short name
        short = model_name.replace("tiny_cnn_", "CNN ").replace("mlp_", "MLP ").replace("tiny_transformer_", "Trans ")
        short = short.replace("_gt", " GT").replace("_a", " α")
        ax.annotate(short, (eval_gt["recall"], eval_gt["precision"]),
                    textcoords="offset points", xytext=(5, -10), fontsize=6.5, alpha=0.7)

    # F1 iso-lines
    for f1_val in [0.75, 0.80, 0.85, 0.90, 0.95]:
        r_range = np.linspace(0.5, 1.0, 200)
        p_range = f1_val * r_range / (2 * r_range - f1_val)
        mask = (p_range > 0.5) & (p_range <= 1.0)
        ax.plot(r_range[mask], p_range[mask], "--", color="#ccc", linewidth=0.8)
        # Label the iso-line
        idx = np.argmin(np.abs(r_range[mask] - 0.95))
        if idx < len(r_range[mask]):
            ax.text(r_range[mask][idx], p_range[mask][idx], f"F1={f1_val}",
                    fontsize=7, color="#999", ha="center")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall (Eval Set, GT Labels)")
    ax.set_xlim(0.65, 1.0)
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc="lower left")
    fig.tight_layout()
    save_fig(fig, output_dir, "fig5_precision_recall.png")


# ── Figure 6: Efficiency Plot (F1 vs Model Size) ─────────────────

def plot_efficiency(results_dir, output_dir):
    """F1 vs model size bubble chart."""
    data = load_comprehensive_eval(results_dir)
    teacher = load_teacher_eval(results_dir)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Teacher
    if teacher and "eval" in teacher:
        ax.scatter(0.43, teacher["eval"]["f1"], s=300, c=COLORS["teacher"],
                   marker="*", zorder=5, label="Teacher CRDNN (0.43 MB)")

    # Best model per architecture
    best = {}
    for model_name, model_data in data.items():
        student_type = model_data.get("student_type", "")
        eval_gt = model_data.get("splits", {}).get("eval", {}).get("gt", {})
        if not eval_gt:
            continue
        f1 = eval_gt.get("f1", 0)
        if student_type not in best or f1 > best[student_type][1]:
            best[student_type] = (model_name, f1, model_data["model_size_mb"],
                                  model_data["total_params"], eval_gt)

    for student_type, (name, f1, size_mb, params, metrics) in best.items():
        color = COLORS.get(student_type, "#666")
        bubble_size = max(params / 500, 30)
        ax.scatter(size_mb, f1, s=bubble_size, c=color, alpha=0.8, zorder=3,
                   label=f"{LABELS.get(student_type, student_type)}\nF1={f1:.3f}, {size_mb:.3f}MB")

    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("F1 Score (Eval, GT)")
    ax.set_title("Model Efficiency: F1 vs Size")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0.75, 1.0)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save_fig(fig, output_dir, "fig6_efficiency.png")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate VAD-KD experiment figures")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating figures in {args.output_dir}/\n")

    print("[1/6] Training curves...")
    plot_training_curves(args.results_dir, args.output_dir)

    print("[2/6] Architecture comparison...")
    plot_architecture_comparison(args.results_dir, args.output_dir)

    print("[3/6] Temperature sweep...")
    plot_temperature_sweep(args.results_dir, args.output_dir)

    print("[4/6] Alpha sweep...")
    plot_alpha_sweep(args.results_dir, args.output_dir)

    print("[5/6] Precision-Recall scatter...")
    plot_precision_recall(args.results_dir, args.output_dir)

    print("[6/6] Efficiency plot...")
    plot_efficiency(args.results_dir, args.output_dir)

    print(f"\nAll 6 figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
