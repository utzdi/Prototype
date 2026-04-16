"""
Generates bar charts for thesis section 6.5.1 (Qualitätsmetriken – Rohwerte).
Saves all charts as PNG files in evaluation_results/charts/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = "evaluation_results/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Claude Sonnet 4.6": "#E8855E",
    "GPT-5.2":           "#6BAED6",
    "Gemini 3.1 Pro":    "#74C476",
    "LLaVA (local)":     "#BDBDBD",
    "Qwen3-VL 32b":      "#9E9AC8",
}
MODELS = list(MODEL_COLORS.keys())
COLORS = [MODEL_COLORS[m] for m in MODELS]

CATEGORY_LABELS = {
    1: "Basisvergleich",
    2: "Räumliche\nVerschiebung",
    3: "Semantisches\nVerständnis",
    4: "Visuelle\nBeeinträchtigung",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── 1. Overall Accuracy ───────────────────────────────────────────────────────
def chart_accuracy_overall():
    data = {
        "Claude Sonnet 4.6": 0.9556,
        "GPT-5.2":           0.9074,
        "Gemini 3.1 Pro":    0.9630,
        "LLaVA (local)":     0.2852,
        "Qwen3-VL 32b":      0.9222,
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(MODELS, [data[m] for m in MODELS],
                  color=COLORS, width=0.55, zorder=2)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Accuracy")
    ax.set_title("Gesamtgenauigkeit (Accuracy) je Modell")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axhline(1.0, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.grid(axis="y", color="#eeeeee", zorder=0)
    for bar, (m, v) in zip(bars, data.items()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save(fig, "1_accuracy_overall.png")


# ── 2. Accuracy by Category ───────────────────────────────────────────────────
def chart_accuracy_by_category():
    cat_data = {
        "Claude Sonnet 4.6": [0.9778, 0.9667, 0.9429, 0.9333],
        "GPT-5.2":           [1.0,    0.9333, 0.8476, 0.9000],
        "Gemini 3.1 Pro":    [1.0,    0.9667, 0.9429, 0.9667],
        "LLaVA (local)":     [0.2000, 0.3111, 0.3524, 0.1000],
        "Qwen3-VL 32b":      [0.9333, 0.9000, 0.9143, 1.0000],
    }
    n_cats = 4
    n_models = len(MODELS)
    x = np.arange(n_cats)
    width = 0.15

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(MODELS):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, cat_data[m], width=width,
                      color=MODEL_COLORS[m], label=m, zorder=2)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Genauigkeit je Modell und Kategorie")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in range(1, 5)])
    ax.grid(axis="y", color="#eeeeee", zorder=0)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save(fig, "2_accuracy_by_category.png")


# ── 3. Precision / Recall / F1 ───────────────────────────────────────────────
def chart_precision_recall_f1():
    prf = {
        "Claude Sonnet 4.6": (0.9975, 0.9616, 0.9792),
        "GPT-5.2":           (0.9896, 0.9161, 0.9514),
        "Gemini 3.1 Pro":    (1.0000, 0.9736, 0.9866),
        "LLaVA (local)":     (0.7747, 0.6019, 0.6775),
        "Qwen3-VL 32b":      (0.9757, 0.9616, 0.9686),
    }
    metrics = ["Precision", "Recall", "F1-Score"]
    metric_colors = ["#6BAED6", "#74C476", "#E8855E"]
    n_models = len(MODELS)
    x = np.arange(n_models)
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, (metric, color) in enumerate(zip(metrics, metric_colors)):
        offset = (j - 1) * width
        values = [prf[m][j] for m in MODELS]
        bars = ax.bar(x + offset, values, width=width,
                      color=color, label=metric, zorder=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall und F1-Score je Modell")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.grid(axis="y", color="#eeeeee", zorder=0)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save(fig, "3_precision_recall_f1.png")


# ── 4. Hallucination Rates ────────────────────────────────────────────────────
def chart_hallucination():
    hall = {
        "Claude Sonnet 4.6": (0.0019, 0.0037, 0.0000),
        "GPT-5.2":           (0.0074, 0.0148, 0.0037),
        "Gemini 3.1 Pro":    (0.0000, 0.0000, 0.0000),
        "LLaVA (local)":     (0.1352, 0.2704, 0.0444),
        "Qwen3-VL 32b":      (0.0185, 0.0370, 0.0000),
    }
    labels = [
        "Breit – Anteil halluzinierter\nEinzelscreenshots (von 540)",
        "Paarweise – mind. 1 halluzinierter\nScreenshot im Paar",
        "Strikt – Element war in keinem\nScreenshot vorhanden (GT: neither)",
    ]
    hall_colors = ["#FDAE6B", "#E6550D", "#A63603"]
    n_models = len(MODELS)
    x = np.arange(n_models)
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (label, color) in enumerate(zip(labels, hall_colors)):
        offset = (j - 1) * width
        values = [hall[m][j] for m in MODELS]
        bars = ax.bar(x + offset, values, width=width,
                      color=color, label=label, zorder=2)
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.1%}", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylim(0, 0.35)
    ax.set_ylabel("Halluzinationsrate")
    ax.set_title("Halluzinationsraten je Modell")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.grid(axis="y", color="#eeeeee", zorder=0)
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    save(fig, "4_hallucination_rates.png")


# ── 5. Stability Overall ──────────────────────────────────────────────────────
def chart_stability_overall():
    orig = {
        "Claude Sonnet 4.6": 0.9556,
        "GPT-5.2":           0.9444,
        "Gemini 3.1 Pro":    0.9667,
        "LLaVA (local)":     0.7111,
        "Qwen3-VL 32b":      0.9333,
    }
    corr = {
        "Claude Sonnet 4.6": 0.9889,
        "GPT-5.2":           0.9778,
        "Gemini 3.1 Pro":    0.9667,
        "LLaVA (local)":     0.7111,
        "Qwen3-VL 32b":      0.9778,
    }
    x = np.arange(len(MODELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - width / 2, [orig[m] for m in MODELS],
                   width=width, color=COLORS, label="Original", zorder=2)
    bars2 = ax.bar(x + width / 2, [corr[m] for m in MODELS],
                   width=width, color=COLORS, alpha=0.45, label="Korrigiert",
                   zorder=2, edgecolor=COLORS, linewidth=1.2)

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Stabilitätsrate")
    ax.set_title("Konsistenz (Stability) je Modell – Original vs. Korrigiert")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.grid(axis="y", color="#eeeeee", zorder=0)

    for bars, data_dict in [(bars1, orig), (bars2, corr)]:
        for bar, m in zip(bars, MODELS):
            v = data_dict[m]
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=8)

    # Custom legend
    orig_patch = mpatches.Patch(facecolor="#888888", label="Original")
    corr_patch = mpatches.Patch(facecolor="#888888", alpha=0.4, label="Korrigiert",
                                edgecolor="#888888", linewidth=1.2)
    ax.legend(handles=[orig_patch, corr_patch], fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save(fig, "5_stability_overall.png")


# ── 6. Stability by Category (heatmap-style grouped bar) ─────────────────────
def chart_stability_by_category():
    cat_stab = {
        "Claude Sonnet 4.6": [0.9333, 1.0000, 0.9429, 0.9000],
        "GPT-5.2":           [1.0000, 1.0000, 0.8571, 1.0000],
        "Gemini 3.1 Pro":    [1.0000, 1.0000, 0.9429, 0.9000],
        "LLaVA (local)":     [1.0000, 0.5000, 0.6857, 1.0000],
        "Qwen3-VL 32b":      [0.8667, 1.0000, 0.8857, 1.0000],
    }
    n_cats = 4
    n_models = len(MODELS)
    x = np.arange(n_cats)
    width = 0.15

    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, m in enumerate(MODELS):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, cat_stab[m], width=width,
               color=MODEL_COLORS[m], label=m, zorder=2)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Stabilitätsrate")
    ax.set_title("Konsistenz (Stability) je Modell und Kategorie")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in range(1, 5)])
    ax.grid(axis="y", color="#eeeeee", zorder=0)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save(fig, "6_stability_by_category.png")


# ── 7. Accuracy Heatmap (Modell × Kategorie) ─────────────────────────────────
def chart_accuracy_heatmap():
    import matplotlib.colors as mcolors

    data = {
        "Claude Sonnet 4.6": [0.9778, 0.9667, 0.9429, 0.9333],
        "GPT-5.2":           [1.0000, 0.9333, 0.8476, 0.9000],
        "Gemini 3.1 Pro":    [1.0000, 0.9667, 0.9429, 0.9667],
        "LLaVA (local)":     [0.2000, 0.3111, 0.3524, 0.1000],
        "Qwen3-VL 32b":      [0.9333, 0.9000, 0.9143, 1.0000],
    }
    cat_labels = [
        "Basisvergleich",
        "Räumliche\nVerschiebung",
        "Semantisches\nVerständnis",
        "Visuelle\nBeeinträchtigung",
    ]

    matrix = np.array([data[m] for m in MODELS])

    fig, ax = plt.subplots(figsize=(7, 4))

    # Two-segment colormap: red→white for low values, white→green for high values
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rw_g",
        [(0.0, "#D73027"), (0.5, "#FFFFBF"), (0.75, "#A8D98A"), (1.0, "#1A7D2E")],
    )

    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Axis labels
    ax.set_xticks(range(4))
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=9)
    ax.set_title("Accuracy – Heatmap Modell × Kategorie", pad=12)

    # Annotate cells
    for i in range(len(MODELS)):
        for j in range(4):
            v = matrix[i, j]
            text_color = "white" if v < 0.45 else "black"
            ax.text(j, i, f"{v:.1%}", ha="center", va="center",
                    fontsize=9.5, color=text_color, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    )
    cbar.set_label("Accuracy", fontsize=9)

    fig.tight_layout()
    save(fig, "7_accuracy_heatmap.png")


# ── 8. Performance: Tokens/Sek & Latenz ──────────────────────────────────────
def chart_performance():
    perf = {
        #                         tok/sek   latenz_ms
        "Claude Sonnet 4.6": (468.4,  9553),
        "GPT-5.2":           (658.4,  6042),
        "Gemini 3.1 Pro":    (501.0,  6616),
        "LLaVA (local)":     (115.3, 15311),
        "Qwen3-VL 32b":      (96.9,  32945),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Tokens / Sekunde ──────────────────────────────────────────────────────
    tok_vals = [perf[m][0] for m in MODELS]
    bars = ax1.barh(MODELS, tok_vals, color=COLORS, zorder=2)
    for bar, v in zip(bars, tok_vals):
        ax1.text(v + 8, bar.get_y() + bar.get_height() / 2,
                 f"{v:.0f}", va="center", fontsize=9)
    ax1.set_xlabel("Tokens / Sekunde")
    ax1.set_title("Durchsatz (Tokens/Sek)")
    ax1.set_xlim(0, max(tok_vals) * 1.22)
    ax1.invert_yaxis()
    ax1.grid(axis="x", color="#eeeeee", zorder=0)

    # ── Durchschnittliche Latenz ──────────────────────────────────────────────
    lat_vals = [perf[m][1] / 1000 for m in MODELS]   # → Sekunden
    bars2 = ax2.barh(MODELS, lat_vals, color=COLORS, zorder=2)
    for bar, v in zip(bars2, lat_vals):
        ax2.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f} s", va="center", fontsize=9)
    ax2.set_xlabel("Ø Latenz pro Anfrage (s)")
    ax2.set_title("Durchschnittliche Latenz")
    ax2.set_xlim(0, max(lat_vals) * 1.22)
    ax2.invert_yaxis()
    ax2.grid(axis="x", color="#eeeeee", zorder=0)

    fig.suptitle("Performance-Vergleich der Modelle", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "8_performance.png")


if __name__ == "__main__":
    print("Generating charts for section 6.5.1 …")
    chart_accuracy_overall()
    chart_accuracy_by_category()
    chart_precision_recall_f1()
    chart_hallucination()
    chart_stability_overall()
    chart_stability_by_category()
    chart_accuracy_heatmap()
    chart_performance()
    print("Done. All charts saved to", OUTPUT_DIR)
