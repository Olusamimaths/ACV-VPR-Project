#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.project_eda import compute_image_stats, plot_campus_label_breakdown, plot_day_night_stats, plot_place_coverage, stats_summary
from datasets.load_dataset import CampusDataset, GardensPointDataset
from evaluation.run_output import DEFAULT_OUTPUT_ROOT, ExperimentRunOutput


def plot_image_size_distributions(
    gp_db: dict[str, np.ndarray],
    gp_q: dict[str, np.ndarray],
    campus_db: dict[str, np.ndarray],
    campus_q: dict[str, np.ndarray],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(gp_db["widths"], gp_db["heights"], label="GP day", alpha=0.6, color="#4575b4")
    axes[0].scatter(gp_q["widths"], gp_q["heights"], label="GP night", alpha=0.6, color="#74add1")
    axes[0].scatter(campus_db["widths"], campus_db["heights"], label="Campus day", alpha=0.6, color="#d73027")
    axes[0].scatter(campus_q["widths"], campus_q["heights"], label="Campus night", alpha=0.6, color="#f46d43")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")
    axes[0].set_title("Image resolution clusters")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    labels = ["GP day", "GP night", "Campus day", "Campus night"]
    aspect_values = [
        gp_db["widths"] / gp_db["heights"],
        gp_q["widths"] / gp_q["heights"],
        campus_db["widths"] / campus_db["heights"],
        campus_q["widths"] / campus_q["heights"],
    ]
    axes[1].boxplot(aspect_values, tick_labels=labels, patch_artist=True)
    axes[1].set_title("Aspect ratio distribution")
    axes[1].set_ylabel("Width / Height")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    return fig


def make_montage(images: list[np.ndarray], indices: list[int], title: str) -> plt.Figure:
    cols = len(indices)
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.2))
    if cols == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        ax.imshow(images[idx])
        ax.set_title(f"{title} #{idx+1}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return fig


def save_fig(fig: plt.Figure, run_output: ExperimentRunOutput, stable_dir: Path, filename: str) -> None:
    run_path = Path(run_output.run_path(filename))
    fig.savefig(run_path, dpi=150, bbox_inches="tight")
    fig.savefig(stable_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_profile_summary(
    gp_db_summary: dict[str, float],
    gp_q_summary: dict[str, float],
    campus_db_summary: dict[str, float],
    campus_q_summary: dict[str, float],
    campus_label_counts: Counter,
    place_counts: dict[str, Counter],
) -> str:
    top_place_id, top_place_count = place_counts["night"].most_common(1)[0]
    top_place_name = place_counts["names"][top_place_id]
    return (
        "Dataset Profiling EDA Summary\n"
        + "=" * 70
        + "\n\n"
        + "1. Are the datasets balanced at the split level?\n"
        + "   GardensPoint is balanced at 200 day references and 200 night queries.\n"
        + "   Campus is smaller and less balanced at 50 day references and 64 night queries.\n\n"
        + "2. Are image sizes consistent?\n"
        + f"   GardensPoint day mean size: {gp_db_summary['width_mean']:.0f}x{gp_db_summary['height_mean']:.0f}\n"
        + f"   GardensPoint night mean size: {gp_q_summary['width_mean']:.0f}x{gp_q_summary['height_mean']:.0f}\n"
        + f"   Campus day mean size: {campus_db_summary['width_mean']:.0f}x{campus_db_summary['height_mean']:.0f}\n"
        + f"   Campus night mean size: {campus_q_summary['width_mean']:.0f}x{campus_q_summary['height_mean']:.0f}\n"
        + "   GardensPoint uses different day and night resolutions, while campus is size-consistent.\n\n"
        + "3. How different are the day and night image distributions?\n"
        + f"   GardensPoint brightness means: day {gp_db_summary['brightness_mean']:.1f}, night {gp_q_summary['brightness_mean']:.1f}\n"
        + f"   Campus brightness means: day {campus_db_summary['brightness_mean']:.1f}, night {campus_q_summary['brightness_mean']:.1f}\n"
        + "   GardensPoint night images appear exposure-compensated or over-brightened, while campus night images are genuinely darker.\n\n"
        + "4. How challenging is the strict campus query composition before any model is used?\n"
        + f"   Exact filename matches: {campus_label_counts.get('Exact filename match', 0)}\n"
        + f"   Marked no-match queries: {campus_label_counts.get('Marked no-match', 0)}\n"
        + f"   PXL unmatched queries: {campus_label_counts.get('PXL unmatched', 0)}\n"
        + "   This means the strict campus protocol is almost half hard no-match or non-1-to-1 queries.\n\n"
        + "5. Is campus place coverage balanced?\n"
        + f"   Unique place IDs: {len(place_counts['names'])}\n"
        + f"   Largest night place bucket: {top_place_name} ({top_place_count} images)\n"
        + "   Place coverage is uneven, so some scenes are much more represented than others.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile raw dataset distributions before experimentation.")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gardens_dir", type=str, default="images/GardensPoint/")
    parser.add_argument("--campus_dir", type=str, default="custom_dataset/")
    parser.add_argument("--campus_place_dir", type=str, default="custom_dataset_place_level/")
    args = parser.parse_args()

    run_output = ExperimentRunOutput.create(
        args.output_root,
        run_slug="dataset_profile_eda",
        category="eda-data-profiling",
    )
    stable_dir = Path(args.output_root) / "eda" / "data_profiling"
    stable_dir.mkdir(parents=True, exist_ok=True)
    print(f"===== Saving dataset profiling EDA outputs to {run_output.run_dir}")

    print("===== Load GardensPoint and campus datasets")
    gp_db_images, gp_q_images, _, _ = GardensPointDataset(destination=args.gardens_dir).load()
    campus_db_images, campus_q_images, _, _ = CampusDataset(destination=args.campus_dir).load()

    print("===== Compute raw image statistics")
    gp_db_stats = compute_image_stats(gp_db_images)
    gp_q_stats = compute_image_stats(gp_q_images)
    campus_db_stats = compute_image_stats(campus_db_images)
    campus_q_stats = compute_image_stats(campus_q_images)

    save_fig(
        plot_day_night_stats(gp_db_stats, gp_q_stats, "GardensPoint raw day/night distributions"),
        run_output,
        stable_dir,
        "gardenspoint_raw_distributions.png",
    )
    save_fig(
        plot_day_night_stats(campus_db_stats, campus_q_stats, "Campus raw day/night distributions"),
        run_output,
        stable_dir,
        "campus_raw_distributions.png",
    )
    save_fig(
        plot_image_size_distributions(gp_db_stats, gp_q_stats, campus_db_stats, campus_q_stats),
        run_output,
        stable_dir,
        "image_size_distributions.png",
    )

    label_fig, campus_label_counts = plot_campus_label_breakdown(Path(args.campus_dir) / "night_images")
    save_fig(label_fig, run_output, stable_dir, "campus_strict_query_composition.png")

    place_fig, place_counts = plot_place_coverage(Path(args.campus_place_dir))
    save_fig(place_fig, run_output, stable_dir, "campus_place_coverage.png")

    save_fig(
        make_montage(gp_db_images, [0, 50, 100, 150], "GP day"),
        run_output,
        stable_dir,
        "gardenspoint_day_montage.png",
    )
    save_fig(
        make_montage(gp_q_images, [0, 50, 100, 150], "GP night"),
        run_output,
        stable_dir,
        "gardenspoint_night_montage.png",
    )
    save_fig(
        make_montage(campus_db_images, [0, 9, 20, 40], "Campus day"),
        run_output,
        stable_dir,
        "campus_day_montage.png",
    )
    save_fig(
        make_montage(campus_q_images, [0, 9, 20, 40], "Campus night"),
        run_output,
        stable_dir,
        "campus_night_montage.png",
    )

    summary_text = build_profile_summary(
        stats_summary(gp_db_stats),
        stats_summary(gp_q_stats),
        stats_summary(campus_db_stats),
        stats_summary(campus_q_stats),
        campus_label_counts,
        place_counts,
    )
    Path(run_output.run_path("dataset_profile_summary.txt")).write_text(summary_text, encoding="utf-8")
    (stable_dir / "dataset_profile_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print("===== Dataset profiling EDA complete")


if __name__ == "__main__":
    main()
