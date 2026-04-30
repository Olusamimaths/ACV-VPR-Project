#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.load_dataset import CampusDataset, GardensPointDataset
from evaluation.run_output import DEFAULT_OUTPUT_ROOT, ExperimentRunOutput


RESULTS_FILENAME = "results.txt"


@dataclass
class MetricRecord:
    descriptor: str
    auc: float
    r100p: float
    r1: float
    r5: float
    r10: float
    tp: int
    fp: int
    run_path: Path


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    rgb = image[..., :3].astype(np.float32)
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


def compute_image_stats(images: list[np.ndarray]) -> dict[str, np.ndarray]:
    brightness = []
    contrast = []
    sharpness = []
    heights = []
    widths = []
    for image in images:
        gray = rgb_to_gray(image)
        brightness.append(float(gray.mean()))
        contrast.append(float(gray.std()))
        sharpness.append(float(np.var(laplace(gray))))
        heights.append(int(image.shape[0]))
        widths.append(int(image.shape[1]))
    return {
        "brightness": np.array(brightness, dtype=np.float32),
        "contrast": np.array(contrast, dtype=np.float32),
        "sharpness": np.array(sharpness, dtype=np.float32),
        "heights": np.array(heights, dtype=np.int32),
        "widths": np.array(widths, dtype=np.int32),
    }


def stats_summary(stats: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "brightness_mean": float(np.mean(stats["brightness"])),
        "contrast_mean": float(np.mean(stats["contrast"])),
        "sharpness_mean": float(np.mean(stats["sharpness"])),
        "height_mean": float(np.mean(stats["heights"])),
        "width_mean": float(np.mean(stats["widths"])),
    }


def plot_day_night_stats(
    db_stats: dict[str, np.ndarray],
    q_stats: dict[str, np.ndarray],
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("brightness", "Brightness (mean grayscale)"),
        ("contrast", "Contrast (std grayscale)"),
        ("sharpness", "Sharpness (Laplacian variance)"),
    ]
    colors = {"Database / day": "#4575b4", "Query / night": "#d73027"}

    for ax, (key, label) in zip(axes.flat[:3], metrics):
        ax.hist(
            db_stats[key],
            bins=20,
            alpha=0.7,
            color=colors["Database / day"],
            label="Database / day",
        )
        ax.hist(
            q_stats[key],
            bins=20,
            alpha=0.6,
            color=colors["Query / night"],
            label="Query / night",
        )
        ax.set_title(label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    box_ax = axes.flat[3]
    box_data = [
        db_stats["brightness"],
        q_stats["brightness"],
        db_stats["contrast"],
        q_stats["contrast"],
        db_stats["sharpness"],
        q_stats["sharpness"],
    ]
    box_labels = [
        "Day\nbright",
        "Night\nbright",
        "Day\ncontrast",
        "Night\ncontrast",
        "Day\nsharp",
        "Night\nsharp",
    ]
    box_ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    box_ax.set_title("Summary comparison")
    box_ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def classify_campus_night_image(name: str) -> str:
    if name.startswith("PXL_"):
        return "PXL unmatched"
    if "-npm" in name.lower() or name.startswith("npm"):
        return "Marked no-match"
    return "Exact filename match"


def plot_campus_label_breakdown(night_dir: Path) -> tuple[plt.Figure, Counter]:
    counts = Counter()
    for image_path in sorted(night_dir.glob("*.jpg")):
        counts[classify_campus_night_image(image_path.stem)] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    colors = ["#74add1", "#fdae61", "#d73027"]

    axes[0].bar(labels, values, color=colors[: len(labels)])
    axes[0].set_title("Campus strict query label breakdown")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].pie(values, labels=labels, autopct="%1.0f%%", startangle=90, colors=colors[: len(labels)])
    axes[1].set_title("Query composition")

    fig.tight_layout()
    return fig, counts


def plot_place_coverage(dataset_dir: Path) -> tuple[plt.Figure, dict[str, object]]:
    day_counts = Counter()
    night_counts = Counter()
    place_names = {}
    flow_steps = {}

    with (dataset_dir / "day_place_index.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            place_id = row["place_id"]
            day_counts[place_id] += 1
            place_names[place_id] = row["place_name"]
            flow_steps[place_id] = int(row["flow_step"])

    with (dataset_dir / "night_rematches.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            place_id = row["rematched_place_id"]
            night_counts[place_id] += 1
            place_names[place_id] = row["place_name"]
            flow_steps[place_id] = int(row["flow_step"])

    ordered_places = sorted(place_names.keys(), key=lambda pid: flow_steps[pid])
    x = np.arange(len(ordered_places))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, [day_counts[p] for p in ordered_places], width, label="Day references", color="#4575b4")
    ax.bar(x + width / 2, [night_counts[p] for p in ordered_places], width, label="Night queries", color="#d73027")
    ax.set_xticks(x)
    ax.set_xticklabels([place_names[p] for p in ordered_places], rotation=0, fontsize=9)
    ax.set_ylabel("Image count")
    ax.set_title("Campus place-level coverage")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig, {"day": day_counts, "night": night_counts, "names": place_names, "ordered_places": ordered_places}


def parse_result_file(path: Path) -> MetricRecord | None:
    text = path.read_text(encoding="utf-8")
    descriptor_match = re.search(r"Descriptor:\s*(.+)", text)
    auc_match = re.search(r"AUC:\s*([0-9.]+)", text)
    r100p_match = re.search(r"R@100P:\s*([0-9.]+)", text)
    r1_match = re.search(r"R@1:\s*([0-9.]+)", text)
    r5_match = re.search(r"R@5:\s*([0-9.]+)", text)
    r10_match = re.search(r"R@10:\s*([0-9.]+)", text)
    tp_match = re.search(r"True positives \(thresholded\):\s*(\d+)", text)
    fp_match = re.search(r"False positives \(thresholded\):\s*(\d+)", text)

    if not all([descriptor_match, auc_match, r100p_match, r1_match, r5_match, r10_match, tp_match, fp_match]):
        return None

    return MetricRecord(
        descriptor=descriptor_match.group(1).strip(),
        auc=float(auc_match.group(1)),
        r100p=float(r100p_match.group(1)),
        r1=float(r1_match.group(1)),
        r5=float(r5_match.group(1)),
        r10=float(r10_match.group(1)),
        tp=int(tp_match.group(1)),
        fp=int(fp_match.group(1)),
        run_path=path,
    )


def load_experiment_results(results_root: Path) -> dict[str, dict[str, MetricRecord]]:
    grouped: dict[str, dict[str, MetricRecord]] = defaultdict(dict)
    for result_path in sorted(results_root.glob("runs/*/*/results.txt")):
        record = parse_result_file(result_path)
        if record is None:
            continue
        category = result_path.parents[1].name
        grouped[category][record.descriptor] = record
    return grouped


def plot_descriptor_comparison(grouped_results: dict[str, dict[str, MetricRecord]]) -> plt.Figure:
    gardens = grouped_results.get("benchmarks", {})
    campus = grouped_results.get("campus_strict", {})
    shared_descriptors = [
        descriptor
        for descriptor in [
            "CosPlace",
            "EigenPlaces",
            "NetVLAD",
            "PatchNetVLAD",
            "HDC-DELF",
            "SALAD",
            "SelaVPR++",
            "VPRTempo",
        ]
        if descriptor in gardens or descriptor in campus
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(shared_descriptors))
    width = 0.36

    gardens_auc = [gardens[d].auc if d in gardens else np.nan for d in shared_descriptors]
    gardens_r1 = [gardens[d].r1 if d in gardens else np.nan for d in shared_descriptors]
    campus_auc = [campus[d].auc if d in campus else np.nan for d in shared_descriptors]
    campus_r1 = [campus[d].r1 if d in campus else np.nan for d in shared_descriptors]

    axes[0].bar(x - width / 2, gardens_auc, width, label="GardensPoint AUC", color="#4575b4")
    axes[0].bar(x + width / 2, campus_auc, width, label="Campus strict AUC", color="#d73027")
    axes[0].set_title("AUC comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(shared_descriptors, rotation=35, ha="right")
    axes[0].grid(alpha=0.2, axis="y")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - width / 2, gardens_r1, width, label="GardensPoint R@1", color="#74add1")
    axes[1].bar(x + width / 2, campus_r1, width, label="Campus strict R@1", color="#f46d43")
    axes[1].set_title("R@1 comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(shared_descriptors, rotation=35, ha="right")
    axes[1].grid(alpha=0.2, axis="y")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig


def find_best(record_map: dict[str, MetricRecord], key: str) -> tuple[str, float]:
    best = max(record_map.values(), key=lambda record: getattr(record, key))
    return best.descriptor, getattr(best, key)


def find_max_place(counter: Counter, place_names: dict[str, str]) -> tuple[str, int]:
    place_id, count = counter.most_common(1)[0]
    return place_names[place_id], count


def build_summary_text(
    gp_db_summary: dict[str, float],
    gp_q_summary: dict[str, float],
    campus_db_summary: dict[str, float],
    campus_q_summary: dict[str, float],
    campus_label_counts: Counter,
    place_counts: dict[str, Counter],
    grouped_results: dict[str, dict[str, MetricRecord]],
) -> str:
    gardens = grouped_results.get("benchmarks", {})
    campus = grouped_results.get("campus_strict", {})

    gp_brightness_drop = gp_db_summary["brightness_mean"] - gp_q_summary["brightness_mean"]
    campus_brightness_drop = campus_db_summary["brightness_mean"] - campus_q_summary["brightness_mean"]

    place_id, place_count = place_counts["night"].most_common(1)[0]
    place_name = place_counts["names"][place_id]
    best_gp_auc_name, best_gp_auc = find_best(gardens, "auc")
    best_gp_r1_name, best_gp_r1 = find_best(gardens, "r1")
    best_campus_auc_name, best_campus_auc = find_best(campus, "auc")
    best_campus_r1_name, best_campus_r1 = find_best(campus, "r1")

    return (
        "Project EDA Summary\n"
        + "=" * 70
        + "\n\n"
        + "1. How strong is the day-night appearance gap?\n"
        + f"   GardensPoint day mean brightness: {gp_db_summary['brightness_mean']:.1f}\n"
        + f"   GardensPoint night mean brightness: {gp_q_summary['brightness_mean']:.1f}\n"
        + f"   GardensPoint brightness change (day - night): {gp_brightness_drop:.1f}\n"
        + f"   Campus day mean brightness: {campus_db_summary['brightness_mean']:.1f}\n"
        + f"   Campus night mean brightness: {campus_q_summary['brightness_mean']:.1f}\n"
        + f"   Campus brightness change (day - night): {campus_brightness_drop:.1f}\n"
        + "   GardensPoint night images appear exposure-compensated or over-brightened relative to the day split,\n"
        + "   so their higher mean intensity should not be interpreted as an easier natural night condition.\n\n"
        + "2. How hard is the strict campus label setup?\n"
        + f"   Exact filename match queries: {campus_label_counts.get('Exact filename match', 0)}\n"
        + f"   Marked no-match queries: {campus_label_counts.get('Marked no-match', 0)}\n"
        + f"   PXL unmatched queries: {campus_label_counts.get('PXL unmatched', 0)}\n\n"
        + "3. Is the campus place-level coverage balanced?\n"
        + f"   Largest night place bucket: {place_name} ({place_count} night images)\n"
        + f"   Number of unique place IDs: {len(place_counts['names'])}\n\n"
        + "4. Which descriptors currently look strongest?\n"
        + f"   Best GardensPoint AUC: {best_gp_auc_name} ({best_gp_auc:.3f})\n"
        + f"   Best GardensPoint R@1: {best_gp_r1_name} ({best_gp_r1:.3f})\n"
        + f"   Best strict campus AUC: {best_campus_auc_name} ({best_campus_auc:.3f})\n"
        + f"   Best strict campus R@1: {best_campus_r1_name} ({best_campus_r1:.3f})\n\n"
        + "5. Main implication\n"
        + "   The campus dataset is not only darker at night, but also structurally harder because\n"
        + "   many queries are treated as no-match cases. This supports the project focus on strong\n"
        + "   descriptors first, followed by better rejection and temporal reasoning.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis for the VPR project.")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Base directory for saved outputs.")
    parser.add_argument("--gardens_dir", type=str, default="images/GardensPoint/", help="GardensPoint dataset directory.")
    parser.add_argument("--campus_dir", type=str, default="custom_dataset/", help="Strict campus dataset directory.")
    parser.add_argument(
        "--campus_place_dir",
        type=str,
        default="custom_dataset_place_level/",
        help="Place-level campus dataset directory.",
    )
    args = parser.parse_args()

    run_output = ExperimentRunOutput.create(
        args.output_root,
        run_slug="project_eda",
        category="eda-experiment-diagnostics",
    )
    print(f"===== Saving EDA outputs to {run_output.run_dir}")
    stable_dir = Path(args.output_root) / "eda" / "experiment_diagnostics"
    stable_dir.mkdir(parents=True, exist_ok=True)

    print("===== Load GardensPoint dataset")
    gardens_db, gardens_q, _, _ = GardensPointDataset(destination=args.gardens_dir).load()
    print("===== Load strict campus dataset")
    campus_db, campus_q, _, _ = CampusDataset(destination=args.campus_dir).load()

    print("===== Compute image statistics")
    gp_db_stats = compute_image_stats(gardens_db)
    gp_q_stats = compute_image_stats(gardens_q)
    campus_db_stats = compute_image_stats(campus_db)
    campus_q_stats = compute_image_stats(campus_q)

    gp_fig = plot_day_night_stats(gp_db_stats, gp_q_stats, "GardensPoint day vs night statistics")
    gp_run_path = Path(run_output.run_path("gardenspoint_day_night_stats.png"))
    gp_fig.savefig(gp_run_path, dpi=150)
    gp_fig.savefig(stable_dir / "gardenspoint_day_night_stats.png", dpi=150)
    plt.close(gp_fig)

    campus_fig = plot_day_night_stats(campus_db_stats, campus_q_stats, "Campus day vs night statistics")
    campus_run_path = Path(run_output.run_path("campus_day_night_stats.png"))
    campus_fig.savefig(campus_run_path, dpi=150)
    campus_fig.savefig(stable_dir / "campus_day_night_stats.png", dpi=150)
    plt.close(campus_fig)

    label_fig, campus_label_counts = plot_campus_label_breakdown(Path(args.campus_dir) / "night_images")
    label_run_path = Path(run_output.run_path("campus_label_breakdown.png"))
    label_fig.savefig(label_run_path, dpi=150)
    label_fig.savefig(stable_dir / "campus_label_breakdown.png", dpi=150)
    plt.close(label_fig)

    place_fig, place_counts = plot_place_coverage(Path(args.campus_place_dir))
    place_run_path = Path(run_output.run_path("campus_place_coverage.png"))
    place_fig.savefig(place_run_path, dpi=150)
    place_fig.savefig(stable_dir / "campus_place_coverage.png", dpi=150)
    plt.close(place_fig)

    print("===== Parse experiment result files")
    grouped_results = load_experiment_results(Path(args.output_root))
    comparison_fig = plot_descriptor_comparison(grouped_results)
    comparison_run_path = Path(run_output.run_path("descriptor_comparison.png"))
    comparison_fig.savefig(comparison_run_path, dpi=150)
    comparison_fig.savefig(stable_dir / "descriptor_comparison.png", dpi=150)
    plt.close(comparison_fig)

    summary_text = build_summary_text(
        stats_summary(gp_db_stats),
        stats_summary(gp_q_stats),
        stats_summary(campus_db_stats),
        stats_summary(campus_q_stats),
        campus_label_counts,
        place_counts,
        grouped_results,
    )
    Path(run_output.run_path("eda_summary.txt")).write_text(summary_text, encoding="utf-8")
    (stable_dir / "eda_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print("===== EDA complete")


if __name__ == "__main__":
    main()
