#!/usr/bin/env python3
"""
Generate golden reference metrics for INT8 TFLite inference.

Given a manifest of audio clips and a TFLite model, this script runs the
RealtimeTFLiteDetector and writes per-window scores to CSV files alongside
summary statistics. These CSVs can be checked into version control and used as
the baseline when validating Android (or other) inference implementations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_realtime_tflite import RealtimeTFLiteDetector


@dataclass
class ClipConfig:
    """Configuration for a single validation clip."""

    path: Path
    label: str | None = None
    threshold: float | None = None

    @property
    def id(self) -> str:
        """A filesystem-friendly identifier for output file names."""
        stem = self.path.stem
        if self.label:
            return f"{stem}_{self.label}".replace(" ", "_")
        return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the INT8 TFLite model on a manifest of audio clips and "
        "record per-window metrics for validation."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a JSON manifest describing the validation clips.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the quantized TFLite model to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Default anomaly threshold to apply when computing smoothed scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the CSV files and summary JSON will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSVs instead of skipping them.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> List[ClipConfig]:
    """Load clip definitions from JSON manifest.

    The manifest may either be a simple list of file paths, or a dict containing
    a ``clips`` list with objects of the form:

    ``{ "path": "validation/clip.wav", "label": "normal", "threshold": 0.25 }``
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "clips" in data:
        clip_entries = data["clips"]
    elif isinstance(data, list):
        clip_entries = data
    else:
        raise ValueError(
            "Manifest must be either a list of clip paths or a dict with a 'clips' key."
        )

    manifest_root = path.parent
    clips: List[ClipConfig] = []
    for entry in clip_entries:
        if isinstance(entry, str):
            clip_path = (manifest_root / entry).resolve()
            clips.append(ClipConfig(path=clip_path))
        elif isinstance(entry, dict):
            clip_path_raw = entry.get("path")
            if not clip_path_raw:
                raise ValueError(f"Clip entry missing 'path': {entry}")
            clip_path = (manifest_root / clip_path_raw).resolve()
            clips.append(
                ClipConfig(
                    path=clip_path,
                    label=entry.get("label"),
                    threshold=entry.get("threshold", None),
                )
            )
        else:
            raise ValueError(f"Unsupported clip entry in manifest: {entry}")

    return clips


def ensure_clips_exist(clips: Sequence[ClipConfig]) -> None:
    missing = [clip.path for clip in clips if not clip.path.exists()]
    if missing:
        missing_str = "\n  ".join(str(m) for m in missing)
        raise FileNotFoundError(f"Missing audio files:\n  {missing_str}")


def compute_summary(rows: Sequence[Dict[str, float]], threshold: float) -> Dict[str, float]:
    if not rows:
        return {
            "windows": 0,
            "raw_min": math.nan,
            "raw_mean": math.nan,
            "raw_max": math.nan,
            "smoothed_min": math.nan,
            "smoothed_mean": math.nan,
            "smoothed_max": math.nan,
            "smoothed_peak_time": math.nan,
            "smoothed_peak_window": -1,
            "threshold": threshold,
            "anomaly_windows": 0,
        }

    raw_scores = [row["raw_score"] for row in rows]
    smoothed_scores = [row["smoothed_score"] for row in rows]

    peak_index = max(range(len(rows)), key=lambda idx: smoothed_scores[idx])
    peak_time = rows[peak_index]["time_sec"]

    anomaly_windows = sum(score > threshold for score in smoothed_scores)

    summary = {
        "windows": int(len(rows)),
        "raw_min": float(min(raw_scores)),
        "raw_mean": float(statistics.fmean(raw_scores)),
        "raw_max": float(max(raw_scores)),
        "smoothed_min": float(min(smoothed_scores)),
        "smoothed_mean": float(statistics.fmean(smoothed_scores)),
        "smoothed_max": float(max(smoothed_scores)),
        "smoothed_peak_time": float(peak_time),
        "smoothed_peak_window": int(rows[peak_index]["window_index"]),
        "threshold": float(threshold),
        "anomaly_windows": int(anomaly_windows),
    }
    return summary


def write_csv_with_comments(
    csv_path: Path, rows: Sequence[Dict[str, float]], summary: Dict[str, float]
) -> None:
    fieldnames = [
        "window_index",
        "time_sec",
        "timestamp_wall",
        "raw_score",
        "smoothed_score",
        "threshold",
        "is_anomaly",
    ]
    with csv_path.open("w", newline="") as handle:
        for key, value in summary.items():
            handle.write(f"# {key}: {value}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_clip(
    detector: RealtimeTFLiteDetector,
    clip: ClipConfig,
    default_threshold: float,
    output_dir: Path,
    overwrite: bool,
) -> Dict[str, float]:
    threshold = clip.threshold if clip.threshold is not None else default_threshold
    detector.threshold = threshold  # ensure threshold matches clip

    output_csv = output_dir / f"{clip.id}.csv"
    if output_csv.exists() and not overwrite:
        print(f"Skipping {clip.path.name} (exists). Use --overwrite to regenerate.")
        return {"clip": str(clip.path), "skipped": True}

    print(f"\n→ Processing {clip.path.name} (threshold={threshold})")
    start_time = time.perf_counter()
    results = detector.process_file_streaming(clip.path, visualize=False, progress=False)
    elapsed = time.perf_counter() - start_time

    rows: List[Dict[str, float]] = []
    for entry in results:
        rows.append(
            {
                "window_index": entry.get("window_index", len(rows)),
                "time_sec": float(entry["time_seconds"]),
                "timestamp_wall": float(entry["timestamp"]),
                "raw_score": float(entry["raw_score"]),
                "smoothed_score": float(entry["smoothed_score"]),
                "threshold": float(threshold),
                "is_anomaly": int(bool(entry["is_anomaly"])),
            }
        )

    summary = compute_summary(rows, threshold)
    summary.update(
        {
            "clip": str(clip.path),
            "label": clip.label or "",
            "runtime_seconds": round(elapsed, 6),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_with_comments(output_csv, rows, summary)
    print(f"   Wrote {output_csv.name} ({len(rows)} windows, {elapsed:.3f}s runtime)")

    return summary


def main() -> None:
    args = parse_args()
    clips = load_manifest(args.manifest)
    ensure_clips_exist(clips)

    detector = RealtimeTFLiteDetector(model_path=args.model, threshold=args.threshold)

    summaries: List[Dict[str, float]] = []
    for clip in clips:
        summary = process_clip(
            detector=detector,
            clip=clip,
            default_threshold=args.threshold,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        summaries.append(summary)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"\nSaved aggregated summary → {summary_path}")
    print("Validation baseline generation complete.")


if __name__ == "__main__":
    main()
