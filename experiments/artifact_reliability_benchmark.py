#!/usr/bin/env python3
"""CPU-only reliability audit for archived generative-model outputs.

The repository does not bundle the datasets and checkpoints required to train
its models end to end. This benchmark therefore makes a narrower, defensible
claim: it evaluates coverage, conditional consistency, structural adherence,
artifact spill, and control-trajectory smoothness directly from the versioned
outputs already present in the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/generative-artifact-matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import Image, ImageSequence
from scipy.ndimage import binary_dilation, sobel
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DIGIT_FAMILIES = {
    "DCGAN": "gan_baselines/dcgan/DCGAN_Results",
    "WGAN": "gan_baselines/wgan/images",
    "WGAN-GP": "gan_baselines/wgan_gp/images",
    "Conditional WGAN-GP": "gan_baselines/conditional_gan/images",
}

FAMILY_COLORS = {
    "DCGAN": "#64748B",
    "WGAN": "#2563EB",
    "WGAN-GP": "#8B5CF6",
    "Conditional WGAN-GP": "#F97316",
}


@dataclass(frozen=True)
class DigitCheckpointDiagnostic:
    family: str
    checkpoint: str
    cluster_seed: int
    effective_modes: float
    normalized_cluster_entropy: float
    nearest_neighbor_distance: float


@dataclass(frozen=True)
class ConditionalDiagnostic:
    checkpoint: str
    leave_one_checkpoint_out_accuracy: float


@dataclass(frozen=True)
class Pix2PixDiagnostic:
    sample: str
    edge_recall: float
    edge_precision: float
    edge_f1: float
    background_chroma_spill: float
    background_mean_chroma: float


@dataclass(frozen=True)
class TrajectoryDiagnostic:
    control: str
    sample: str
    frames: int
    mean_step_rms: float
    step_coefficient_of_variation: float
    temporal_roughness: float
    path_linearity: float
    consecutive_edge_correlation: float
    endpoint_edge_correlation: float


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=float) / 255.0


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=float) / 255.0


def normalize_digit(tile: np.ndarray, size: int = 16) -> np.ndarray:
    tile = np.asarray(tile, dtype=float)
    tile -= tile.min()
    tile /= max(float(tile.max()), 1e-12)
    mask = tile > 0.08
    if mask.any():
        rows, columns = np.where(mask)
        tile = tile[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    height, width = tile.shape
    inner = int(size * 0.72)
    scale = min(inner / height, inner / width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = np.asarray(
        Image.fromarray(np.uint8(tile * 255.0)).resize(
            (resized_width, resized_height), Image.Resampling.LANCZOS
        ),
        dtype=float,
    ) / 255.0
    canvas = np.zeros((size, size), dtype=float)
    y = (size - resized_height) // 2
    x = (size - resized_width) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def extract_digit_grid(path: Path, rows: int = 5, columns: int = 8) -> np.ndarray:
    image = read_gray(path)
    y_edges = np.linspace(0, image.shape[0], rows + 1, dtype=int)
    x_edges = np.linspace(0, image.shape[1], columns + 1, dtype=int)
    tiles = [
        normalize_digit(image[y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]])
        for row in range(rows)
        for column in range(columns)
    ]
    result = np.asarray(tiles)
    if result.shape != (rows * columns, 16, 16):
        raise ValueError(f"Unexpected digit-grid shape for {path}: {result.shape}")
    return result


def effective_modes(labels: np.ndarray, modes: int) -> tuple[float, float]:
    counts = np.bincount(labels, minlength=modes).astype(float)
    probabilities = counts / counts.sum()
    positive = probabilities > 0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    return math.exp(entropy), entropy / math.log(modes)


def digit_benchmark(
    root: Path,
    cluster_seeds: Sequence[int],
) -> tuple[
    list[DigitCheckpointDiagnostic],
    list[ConditionalDiagnostic],
    dict[str, dict],
    dict[str, Path],
]:
    grids: dict[str, list[tuple[Path, np.ndarray]]] = {}
    for family, relative in DIGIT_FAMILIES.items():
        paths = sorted((root / relative).glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No archived grids found for {family}")
        grids[family] = [(path, extract_digit_grid(path)) for path in paths]

    flattened = []
    index: dict[str, list[tuple[str, slice]]] = {}
    offset = 0
    for family, items in grids.items():
        index[family] = []
        for path, grid in items:
            features = grid.reshape(len(grid), -1)
            flattened.append(features)
            index[family].append((path.stem, slice(offset, offset + len(features))))
            offset += len(features)
    all_features = np.concatenate(flattened, axis=0)
    standardized = StandardScaler().fit_transform(all_features)
    embedding = PCA(n_components=32, random_state=0).fit_transform(standardized)

    nearest_by_checkpoint = {}
    for family, checkpoints in index.items():
        for checkpoint, section in checkpoints:
            distances = pairwise_distances(embedding[section])
            np.fill_diagonal(distances, np.inf)
            nearest_by_checkpoint[(family, checkpoint)] = float(np.mean(distances.min(axis=1)))

    rows: list[DigitCheckpointDiagnostic] = []
    for seed in cluster_seeds:
        assignments = KMeans(n_clusters=10, n_init=20, random_state=seed).fit_predict(embedding)
        for family, checkpoints in index.items():
            for checkpoint, section in checkpoints:
                modes, entropy = effective_modes(assignments[section], 10)
                rows.append(
                    DigitCheckpointDiagnostic(
                        family=family,
                        checkpoint=checkpoint,
                        cluster_seed=seed,
                        effective_modes=modes,
                        normalized_cluster_entropy=entropy,
                        nearest_neighbor_distance=nearest_by_checkpoint[(family, checkpoint)],
                    )
                )

    conditional_items = grids["Conditional WGAN-GP"]
    expected = np.arange(40, dtype=int) % 10
    conditional_rows = []
    for held_out, (path, test_grid) in enumerate(conditional_items):
        train_grids = [grid for index_, (_, grid) in enumerate(conditional_items) if index_ != held_out]
        train_x = np.concatenate(train_grids).reshape(-1, 16 * 16)
        train_y = np.tile(expected, len(train_grids))
        classifier = make_pipeline(
            StandardScaler(),
            SVC(C=10.0, gamma="scale"),
        )
        classifier.fit(train_x, train_y)
        predicted = classifier.predict(test_grid.reshape(-1, 16 * 16))
        conditional_rows.append(
            ConditionalDiagnostic(
                checkpoint=path.stem,
                leave_one_checkpoint_out_accuracy=float(np.mean(predicted == expected)),
            )
        )

    family_summary = {}
    for family in DIGIT_FAMILIES:
        selected = [item for item in rows if item.family == family]
        family_summary[family] = {
            "archived_checkpoints": len(grids[family]),
            "samples_per_checkpoint": 40,
            "effective_modes_mean": float(np.mean([item.effective_modes for item in selected])),
            "effective_modes_std": float(np.std([item.effective_modes for item in selected])),
            "nearest_neighbor_distance_mean": float(np.mean([item.nearest_neighbor_distance for item in selected])),
            "nearest_neighbor_distance_std": float(np.std([item.nearest_neighbor_distance for item in selected])),
        }
    consistency = np.asarray(
        [item.leave_one_checkpoint_out_accuracy for item in conditional_rows]
    )
    family_summary["Conditional WGAN-GP"]["leave_one_checkpoint_out_consistency_mean"] = float(consistency.mean())
    family_summary["Conditional WGAN-GP"]["leave_one_checkpoint_out_consistency_std"] = float(consistency.std())
    family_summary["Conditional WGAN-GP"]["leave_one_checkpoint_out_consistency_min"] = float(consistency.min())

    representatives = {family: items[-1][0] for family, items in grids.items()}
    return rows, conditional_rows, family_summary, representatives


def luminance(image: np.ndarray) -> np.ndarray:
    return image @ np.asarray([0.2126, 0.7152, 0.0722])


def pix2pix_pair_metric(source_path: Path, output_path: Path) -> Pix2PixDiagnostic:
    source = read_rgb(source_path)
    output = read_rgb(output_path)
    source_gray = source.mean(axis=2)
    output_gray = luminance(output)
    source_ink = source_gray < 0.75
    gradient = np.hypot(sobel(output_gray, axis=0), sobel(output_gray, axis=1))
    output_edge = gradient > np.quantile(gradient, 0.78)
    edge_recall = float(
        np.sum(binary_dilation(output_edge, iterations=2) & source_ink)
        / max(int(source_ink.sum()), 1)
    )
    edge_precision = float(
        np.sum(binary_dilation(source_ink, iterations=2) & output_edge)
        / max(int(output_edge.sum()), 1)
    )
    edge_f1 = 2.0 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-12)
    background = source_gray > 0.97
    chroma = output.max(axis=2) - output.min(axis=2)
    background_chroma_spill = float(np.mean(chroma[background] > 0.25))
    background_mean_chroma = float(np.mean(chroma[background]))
    return Pix2PixDiagnostic(
        sample=source_path.stem.replace("image_or", ""),
        edge_recall=edge_recall,
        edge_precision=edge_precision,
        edge_f1=edge_f1,
        background_chroma_spill=background_chroma_spill,
        background_mean_chroma=background_mean_chroma,
    )


def pix2pix_benchmark(root: Path) -> tuple[list[Pix2PixDiagnostic], dict]:
    directory = root / "image_translation/pix2pix/1"
    rows = []
    for source in sorted(directory.glob("image_or*.png")):
        sample = source.stem.replace("image_or", "")
        output = directory / f"image_{sample}.png"
        if output.is_file():
            rows.append(pix2pix_pair_metric(source, output))
    if not rows:
        raise FileNotFoundError("No paired Pix2Pix artifacts were found")
    summary = {
        "pairs": len(rows),
        "median_edge_recall": float(np.median([item.edge_recall for item in rows])),
        "median_edge_precision": float(np.median([item.edge_precision for item in rows])),
        "median_edge_f1": float(np.median([item.edge_f1 for item in rows])),
        "median_background_chroma_spill": float(np.median([item.background_chroma_spill for item in rows])),
        "worst_background_chroma_spill": float(max(item.background_chroma_spill for item in rows)),
        "worst_spill_sample": max(rows, key=lambda item: item.background_chroma_spill).sample,
        "lowest_edge_f1": float(min(item.edge_f1 for item in rows)),
        "lowest_edge_f1_sample": min(rows, key=lambda item: item.edge_f1).sample,
    }
    return rows, summary


def load_gif(path: Path, size: int = 64) -> np.ndarray:
    frames = []
    with Image.open(path) as image:
        for frame in ImageSequence.Iterator(image):
            resized = frame.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            frames.append(np.asarray(resized, dtype=float) / 255.0)
    if len(frames) < 3:
        raise ValueError(f"Expected at least three frames in {path}")
    return np.stack(frames)


def trajectory_metric(path: Path, control: str) -> TrajectoryDiagnostic:
    frames = load_gif(path)
    first_difference = np.sqrt(np.mean(np.diff(frames, axis=0) ** 2, axis=(1, 2, 3)))
    second_difference = np.sqrt(np.mean(np.diff(frames, n=2, axis=0) ** 2, axis=(1, 2, 3)))
    endpoint = float(np.sqrt(np.mean((frames[-1] - frames[0]) ** 2)))
    edges = []
    for frame in frames:
        gray = luminance(frame)
        edges.append(np.hypot(sobel(gray, axis=0), sobel(gray, axis=1)).reshape(-1))
    consecutive = [
        float(np.corrcoef(edges[index], edges[index + 1])[0, 1])
        for index in range(len(edges) - 1)
    ]
    return TrajectoryDiagnostic(
        control=control,
        sample=path.stem,
        frames=len(frames),
        mean_step_rms=float(first_difference.mean()),
        step_coefficient_of_variation=float(first_difference.std() / first_difference.mean()),
        temporal_roughness=float(second_difference.mean() / first_difference.mean()),
        path_linearity=float(endpoint / first_difference.sum()),
        consecutive_edge_correlation=float(np.median(consecutive)),
        endpoint_edge_correlation=float(np.corrcoef(edges[0], edges[-1])[0, 1]),
    )


def trajectory_benchmark(root: Path) -> tuple[list[TrajectoryDiagnostic], dict, dict[str, Path]]:
    rows = []
    paths_by_control = {}
    for control in ("Pose", "Illumination"):
        paths = sorted((root / "stylegan" / control).glob("*.gif"))
        paths_by_control[control] = paths
        rows.extend(trajectory_metric(path, control) for path in paths)
    summary = {}
    representatives = {}
    for control, paths in paths_by_control.items():
        selected = [item for item in rows if item.control == control]
        summary[control] = {
            "trajectories": len(selected),
            "frames_per_trajectory": sorted({item.frames for item in selected}),
            "median_temporal_roughness": float(np.median([item.temporal_roughness for item in selected])),
            "median_path_linearity": float(np.median([item.path_linearity for item in selected])),
            "median_consecutive_edge_correlation": float(np.median([item.consecutive_edge_correlation for item in selected])),
            "median_step_coefficient_of_variation": float(np.median([item.step_coefficient_of_variation for item in selected])),
        }
        median = summary[control]["median_temporal_roughness"]
        representative = min(selected, key=lambda item: abs(item.temporal_roughness - median))
        representatives[control] = root / "stylegan" / control / f"{representative.sample}.gif"
    return rows, summary, representatives


def diffusion_inventory(root: Path) -> dict:
    paths = sorted((root / "diffusion/Results").glob("*.png"))
    hashes = []
    for path in paths:
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return {
        "archived_images": len(paths),
        "resolution": [64, 64],
        "exact_duplicate_files": len(hashes) - len(set(hashes)),
        "evaluation_status": "artifact inventory only; no real-data reference or checkpoint is bundled",
    }


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_digit_evidence(
    representatives: dict[str, Path],
    family_summary: dict[str, dict],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(15, 4.5), constrained_layout=True)
    for axis, family in zip(axes, DIGIT_FAMILIES):
        axis.imshow(read_gray(representatives[family]), cmap="gray", vmin=0, vmax=1)
        summary = family_summary[family]
        axis.set_title(
            f"{family}\n{summary['effective_modes_mean']:.2f} ± {summary['effective_modes_std']:.2f} effective clusters",
            fontsize=10,
            fontweight="bold",
        )
        axis.axis("off")
    consistency = family_summary["Conditional WGAN-GP"]["leave_one_checkpoint_out_consistency_mean"]
    figure.suptitle(
        f"Archived MNIST grids: support diagnostics and conditional consistency ({consistency * 100:.1f}%)",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def sample_pair_paths(root: Path, sample: str) -> tuple[Path, Path]:
    directory = root / "image_translation/pix2pix/1"
    return directory / f"image_or{sample}.png", directory / f"image_{sample}.png"


def plot_pix2pix_failures(
    root: Path,
    rows: list[Pix2PixDiagnostic],
    output_path: Path,
) -> None:
    median_f1 = float(np.median([item.edge_f1 for item in rows]))
    median_item = min(rows, key=lambda item: abs(item.edge_f1 - median_f1))
    low_edge = min(rows, key=lambda item: item.edge_f1)
    high_spill = max(rows, key=lambda item: item.background_chroma_spill)
    selections = [
        (median_item, "median adherence"),
        (low_edge, "lowest edge F1"),
        (high_spill, "highest background spill"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(8, 11), constrained_layout=True)
    for row_index, (item, label) in enumerate(selections):
        source, output = sample_pair_paths(root, item.sample)
        axes[row_index, 0].imshow(read_rgb(source))
        axes[row_index, 1].imshow(read_rgb(output))
        axes[row_index, 0].set_title(f"{label}: condition {item.sample}", fontsize=10, fontweight="bold")
        axes[row_index, 1].set_title(
            f"F1 {item.edge_f1:.3f} · spill {item.background_chroma_spill * 100:.1f}%",
            fontsize=10,
            fontweight="bold",
        )
        axes[row_index, 0].axis("off")
        axes[row_index, 1].axis("off")
    figure.suptitle("Pix2Pix archive: structural adherence and visible failure cases", fontsize=15, fontweight="bold")
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def selected_gif_frames(path: Path) -> list[np.ndarray]:
    frames = load_gif(path, size=128)
    return [frames[0], frames[len(frames) // 2], frames[-1]]


def plot_control_trajectories(
    representatives: dict[str, Path],
    summary: dict,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for row, control in enumerate(("Illumination", "Pose")):
        frames = selected_gif_frames(representatives[control])
        for column, frame in enumerate(frames):
            axes[row, column].imshow(frame)
            axes[row, column].axis("off")
            axes[row, column].set_title(("start", "middle", "end")[column], fontsize=9)
        axis = axes[row, 3]
        values = [
            summary[control]["median_temporal_roughness"],
            summary[control]["median_path_linearity"],
            summary[control]["median_consecutive_edge_correlation"],
        ]
        labels = ["roughness ↓", "linearity ↑", "edge continuity ↑"]
        axis.barh(np.arange(3), values, color=("#2563EB" if control == "Illumination" else "#F97316"))
        axis.set_yticks(np.arange(3), labels)
        axis.set_xlim(0, 1.05)
        axis.set_title(control, fontweight="bold")
        axis.grid(axis="x", alpha=0.15)
        for index, value in enumerate(values):
            axis.text(value + 0.02, index, f"{value:.3f}", va="center", fontsize=9)
    figure.suptitle("Archived StyleGAN controls: illumination is smoother than pose", fontsize=15, fontweight="bold")
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_portfolio_summary(
    root: Path,
    digit_representatives: dict[str, Path],
    digit_summary: dict[str, dict],
    pix2pix_rows: list[Pix2PixDiagnostic],
    pix2pix_summary: dict,
    trajectory_representatives: dict[str, Path],
    trajectory_summary: dict,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(16, 9), facecolor="#F6F8FC")
    grid = figure.add_gridspec(2, 3, height_ratios=[0.16, 0.84], wspace=0.20)
    title = figure.add_subplot(grid[0, :])
    title.axis("off")
    title.text(0.0, 0.82, "Generative reliability is multi-axis", fontsize=25, fontweight="bold", color="#172033")
    title.text(
        0.0,
        0.28,
        "Artifact-backed evaluation · coverage · conditional consistency · failure cases · control smoothness",
        fontsize=12,
        color="#5D677A",
    )

    digit_axis = figure.add_subplot(grid[1, 0])
    digit_axis.imshow(read_gray(digit_representatives["Conditional WGAN-GP"]), cmap="gray", vmin=0, vmax=1)
    consistency = digit_summary["Conditional WGAN-GP"]["leave_one_checkpoint_out_consistency_mean"]
    consistency_std = digit_summary["Conditional WGAN-GP"]["leave_one_checkpoint_out_consistency_std"]
    digit_axis.set_title("A. Conditional consistency", loc="left", fontsize=14, fontweight="bold")
    digit_axis.text(
        0.0,
        -0.10,
        f"{consistency * 100:.1f}% ± {consistency_std * 100:.1f}% leave-one-checkpoint-out\n"
        "Condition separation—not semantic digit accuracy.",
        transform=digit_axis.transAxes,
        fontsize=9.8,
        color="#344054",
        va="top",
    )
    digit_axis.axis("off")

    pix_axis_grid = grid[1, 1].subgridspec(3, 2, height_ratios=[0.12, 0.44, 0.44], hspace=0.06, wspace=0.04)
    pix_title = figure.add_subplot(pix_axis_grid[0, :])
    pix_title.axis("off")
    pix_title.text(0.0, 0.5, "B. Adherence includes failure analysis", fontsize=14, fontweight="bold", va="center")
    median_f1 = float(np.median([item.edge_f1 for item in pix2pix_rows]))
    median_item = min(pix2pix_rows, key=lambda item: abs(item.edge_f1 - median_f1))
    spill_item = max(pix2pix_rows, key=lambda item: item.background_chroma_spill)
    for row_index, item in enumerate((median_item, spill_item), start=1):
        source, output = sample_pair_paths(root, item.sample)
        for column, image in enumerate((read_rgb(source), read_rgb(output))):
            axis = figure.add_subplot(pix_axis_grid[row_index, column])
            axis.imshow(image)
            axis.axis("off")
            if row_index == 1:
                axis.set_title(("condition", "generated")[column], fontsize=9, fontweight="bold")
    figure.text(
        0.355,
        0.055,
        f"median edge recall {pix2pix_summary['median_edge_recall'] * 100:.1f}% · "
        f"median spill {pix2pix_summary['median_background_chroma_spill'] * 100:.1f}% · "
        f"worst spill {pix2pix_summary['worst_background_chroma_spill'] * 100:.1f}%",
        fontsize=9.8,
        color="#344054",
    )

    trajectory_grid = grid[1, 2].subgridspec(3, 3, height_ratios=[0.12, 0.44, 0.44], hspace=0.06, wspace=0.03)
    trajectory_title = figure.add_subplot(trajectory_grid[0, :])
    trajectory_title.axis("off")
    trajectory_title.text(0.0, 0.5, "C. Control axes have different stability", fontsize=14, fontweight="bold", va="center")
    for row_index, control in enumerate(("Illumination", "Pose"), start=1):
        frames = selected_gif_frames(trajectory_representatives[control])
        for column, frame in enumerate(frames):
            axis = figure.add_subplot(trajectory_grid[row_index, column])
            axis.imshow(frame)
            axis.axis("off")
            if column == 0:
                axis.set_title(control.lower(), fontsize=9, fontweight="bold")
    figure.text(
        0.685,
        0.055,
        f"roughness: illumination {trajectory_summary['Illumination']['median_temporal_roughness']:.3f} · "
        f"pose {trajectory_summary['Pose']['median_temporal_roughness']:.3f}\n"
        f"edge continuity: {trajectory_summary['Illumination']['median_consecutive_edge_correlation']:.3f} · "
        f"{trajectory_summary['Pose']['median_consecutive_edge_correlation']:.3f}",
        fontsize=9.8,
        color="#344054",
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-seeds", type=parse_ints, default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=Path("results/artifact_reliability"))
    args = parser.parse_args()
    if not args.cluster_seeds:
        parser.error("At least one cluster seed is required")

    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    digit_rows, conditional_rows, digit_summary, digit_representatives = digit_benchmark(
        root, args.cluster_seeds
    )
    pix2pix_rows, pix2pix_summary = pix2pix_benchmark(root)
    trajectory_rows, trajectory_summary, trajectory_representatives = trajectory_benchmark(root)
    diffusion_summary = diffusion_inventory(root)

    digit_dicts = [asdict(item) for item in digit_rows]
    conditional_dicts = [asdict(item) for item in conditional_rows]
    pix2pix_dicts = [asdict(item) for item in pix2pix_rows]
    trajectory_dicts = [asdict(item) for item in trajectory_rows]
    write_csv(output_dir / "digit_checkpoint_diagnostics.csv", digit_dicts, list(digit_dicts[0]))
    write_csv(output_dir / "conditional_consistency.csv", conditional_dicts, list(conditional_dicts[0]))
    write_csv(output_dir / "pix2pix_diagnostics.csv", pix2pix_dicts, list(pix2pix_dicts[0]))
    write_csv(output_dir / "control_trajectory_diagnostics.csv", trajectory_dicts, list(trajectory_dicts[0]))

    plot_digit_evidence(digit_representatives, digit_summary, output_dir / "digit_evidence.png")
    plot_pix2pix_failures(root, pix2pix_rows, output_dir / "pix2pix_failure_cases.png")
    plot_control_trajectories(
        trajectory_representatives, trajectory_summary, output_dir / "control_trajectory_evidence.png"
    )
    plot_portfolio_summary(
        root,
        digit_representatives,
        digit_summary,
        pix2pix_rows,
        pix2pix_summary,
        trajectory_representatives,
        trajectory_summary,
        output_dir / "generative_systems_portfolio_summary.png",
    )

    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(root),
        "command": " ".join([sys.executable, *sys.argv]),
        "configuration": {"cluster_seeds": args.cluster_seeds},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn_version,
            "matplotlib": matplotlib.__version__,
        },
        "digit_artifacts": digit_summary,
        "pix2pix_artifacts": pix2pix_summary,
        "stylegan_control_artifacts": trajectory_summary,
        "diffusion_artifacts": diffusion_summary,
        "scope": {
            "current_validation": [
                "archived MNIST grid coverage and conditional consistency",
                "archived Pix2Pix structural adherence and background spill",
                "archived StyleGAN control-trajectory smoothness",
                "diffusion artifact inventory and exact-duplicate check",
            ],
            "historical_only": [
                "end-to-end training and fidelity for GAN, diffusion, translation, and super-resolution models",
                "progressive GAN and coupled-GAN training",
                "CycleGAN quantitative evaluation",
            ],
            "limitations": [
                "Archived checkpoint windows differ across GAN families, so metrics are not an architecture leaderboard.",
                "Conditional consistency measures separability of requested conditions, not semantic digit correctness.",
                "Pix2Pix has conditions and generated outputs but no ground-truth targets in the archive.",
                "Pixel/edge trajectory metrics are proxies and do not establish identity preservation.",
                "No FID, KID, precision/recall, or task accuracy is claimed without real reference data and checkpoints.",
            ],
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "conditional_consistency": {
            key: digit_summary["Conditional WGAN-GP"][key]
            for key in (
                "leave_one_checkpoint_out_consistency_mean",
                "leave_one_checkpoint_out_consistency_std",
                "leave_one_checkpoint_out_consistency_min",
            )
        },
        "pix2pix": pix2pix_summary,
        "stylegan_controls": trajectory_summary,
        "diffusion": diffusion_summary,
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
