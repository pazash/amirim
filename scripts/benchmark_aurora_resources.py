"""
Benchmark Aurora prediction resource usage across different bounding-box scopes.

Measures:
  • Peak CPU RAM (RSS) during each rollout
  • Peak GPU memory allocated / reserved (if CUDA available)
  • Wall-clock time per rollout
  • GPU utilisation % (nvidia-smi snapshot, if available)

Results are saved as:
  • benchmark_aurora_resources.csv  – one row per (scope, lead_time)
  • benchmark_aurora_resources.json – same data, machine-readable
"""

import os
import sys
import gc
import json
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import torch
import numpy as np
import pandas as pd
import psutil

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from aurora import AuroraV1p5

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Same prediction scopes from compare_global_vs_med_aurora.py
# ---------------------------------------------------------------------------
PREDICTION_SCOPES = {
    "Original": {          # 176×112 points
        "lon_min": 18.0, "lon_max": 61.75,
        "lat_max": 53.25, "lat_min": 25.5,
    },
    "Enlarged_Small": {    # 224×176 points
        "lon_min": 12.0, "lon_max": 67.75,
        "lat_max": 61.25, "lat_min": 17.5,
    },
    "Enlarged_Large": {    # 320×240 points
        "lon_min": 0.0, "lon_max": 79.75,
        "lat_max": 69.25, "lat_min": 9.5,
    },
    "Global": None,
}

LEAD_TIMES_H = [6, 24, 72, 120]

# Use a single representative init time — resource usage depends on grid size,
# not the particular weather state.
BENCHMARK_INIT_DT = datetime(2021, 6, 15, 12)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_process_rss_mb() -> float:
    """Return current-process Resident Set Size in MiB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _get_gpu_stats() -> dict:
    """Return current GPU memory stats from PyTorch (MiB)."""
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_mem_allocated_MiB": torch.cuda.memory_allocated() / (1024 ** 2),
        "gpu_mem_reserved_MiB": torch.cuda.memory_reserved() / (1024 ** 2),
        "gpu_max_mem_allocated_MiB": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "gpu_max_mem_reserved_MiB": torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


def _nvidia_smi_gpu_util() -> dict | None:
    """Query nvidia-smi for utilisation % and total / used / free memory."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
        )
        parts = out.decode().strip().split(",")
        return {
            "nvidia_smi_gpu_util_pct": float(parts[0]),
            "nvidia_smi_mem_total_MiB": float(parts[1]),
            "nvidia_smi_mem_used_MiB": float(parts[2]),
            "nvidia_smi_mem_free_MiB": float(parts[3]),
        }
    except Exception:
        return None


def _clear_caches():
    """Force-free as much memory as possible between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def benchmark():
    output_dir = Path("benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path("weather_data")
    data_pipeline = AuroraDataLoader(cache_dir=cache_dir)

    logger.info("Loading Aurora model …")
    aurora_model = AuroraV1p5()
    aurora_model.load_checkpoint(
        "microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main"
    )
    forecaster = AuroraForecaster(model=aurora_model)

    results: list[dict] = []

    for scope_name, bbox in PREDICTION_SCOPES.items():
        logger.info(f"{'=' * 60}")
        logger.info(f"SCOPE: {scope_name}")
        logger.info(f"{'=' * 60}")

        # ---- Pre-load the input batch (shared across lead times) ----
        _clear_caches()
        rss_before_data = _get_process_rss_mb()

        logger.info("Loading input batch …")
        input_batch, _ = data_pipeline.get_batches(
            BENCHMARK_INIT_DT, history_steps=1, forecast_steps=0, bbox=bbox
        )

        # Compute grid dimensions for reporting
        lat_size = len(input_batch.metadata.lat)
        lon_size = len(input_batch.metadata.lon)
        grid_points = lat_size * lon_size

        rss_after_data = _get_process_rss_mb()
        data_ram_delta = rss_after_data - rss_before_data
        logger.info(
            f"  Grid: {lon_size}×{lat_size} = {grid_points:,} points  |  "
            f"Data loading RAM delta: {data_ram_delta:+.1f} MiB"
        )

        # ---- For each lead time, run the needed rollout steps ----
        for lt_h in LEAD_TIMES_H:
            steps_needed = lt_h // 6
            logger.info(f"  Lead time {lt_h}h  ({steps_needed} AR steps) …")

            _clear_caches()
            rss_pre = _get_process_rss_mb()
            gpu_pre = _get_gpu_stats()
            nvsmi_pre = _nvidia_smi_gpu_util()

            t_start = time.perf_counter()
            try:
                predictions = forecaster.predict_rollout(
                    initial_batch=input_batch,
                    steps=steps_needed,
                    fine_lead_times=[6.0],
                )
                success = True
            except Exception as e:
                logger.error(f"    FAILED: {e}")
                success = False
                predictions = []

            t_end = time.perf_counter()
            wall_sec = t_end - t_start

            rss_post = _get_process_rss_mb()
            rss_peak = max(rss_pre, rss_post)
            gpu_post = _get_gpu_stats()
            nvsmi_post = _nvidia_smi_gpu_util()

            row = {
                "scope": scope_name,
                "bbox": str(bbox) if bbox else "Global (full globe)",
                "grid_lon": lon_size,
                "grid_lat": lat_size,
                "grid_points": grid_points,
                "lead_time_h": lt_h,
                "rollout_steps": steps_needed,
                "wall_clock_sec": round(wall_sec, 2),
                "success": success,
                # CPU RAM
                "ram_before_MiB": round(rss_pre, 1),
                "ram_after_MiB": round(rss_post, 1),
                "ram_delta_MiB": round(rss_post - rss_pre, 1),
                "ram_peak_estimate_MiB": round(rss_peak, 1),
            }

            # GPU memory (PyTorch)
            if gpu_post:
                row.update({
                    "gpu_peak_alloc_MiB": round(gpu_post.get("gpu_max_mem_allocated_MiB", 0), 1),
                    "gpu_peak_reserved_MiB": round(gpu_post.get("gpu_max_mem_reserved_MiB", 0), 1),
                    "gpu_alloc_before_MiB": round(gpu_pre.get("gpu_mem_allocated_MiB", 0), 1),
                    "gpu_alloc_after_MiB": round(gpu_post.get("gpu_mem_allocated_MiB", 0), 1),
                })

            # nvidia-smi snapshot
            if nvsmi_post:
                row.update({
                    "nvsmi_gpu_util_pct": nvsmi_post["nvidia_smi_gpu_util_pct"],
                    "nvsmi_mem_used_MiB": nvsmi_post["nvidia_smi_mem_used_MiB"],
                    "nvsmi_mem_total_MiB": nvsmi_post["nvidia_smi_mem_total_MiB"],
                })

            results.append(row)

            logger.info(
                f"    ✓ {wall_sec:.1f}s  |  RAM Δ {row['ram_delta_MiB']:+.0f} MiB  "
                f"| GPU peak alloc {row.get('gpu_peak_alloc_MiB', 'N/A')} MiB"
            )

            # Free predictions
            del predictions
            _clear_caches()

        # Free the input batch before the next scope
        del input_batch
        _clear_caches()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)

    csv_path = output_dir / "benchmark_aurora_resources.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV  → {csv_path}")

    json_path = output_dir / "benchmark_aurora_resources.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON → {json_path}")

    # ------------------------------------------------------------------
    # Print a human-readable summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RESOURCE BENCHMARK SUMMARY")
    print("=" * 90)

    summary_cols = [
        "scope", "grid_lon", "grid_lat", "lead_time_h",
        "wall_clock_sec", "ram_delta_MiB",
    ]
    if "gpu_peak_alloc_MiB" in df.columns:
        summary_cols.append("gpu_peak_alloc_MiB")
    if "gpu_peak_reserved_MiB" in df.columns:
        summary_cols.append("gpu_peak_reserved_MiB")

    print(df[summary_cols].to_string(index=False))
    print("=" * 90)

    # ------------------------------------------------------------------
    # Print efficiency comparison relative to Global
    # ------------------------------------------------------------------
    global_rows = df[df["scope"] == "Global"]
    if not global_rows.empty:
        print("\nEFFICIENCY vs GLOBAL")
        print("-" * 70)
        for lt_h in LEAD_TIMES_H:
            g = global_rows[global_rows["lead_time_h"] == lt_h]
            if g.empty:
                continue
            g_time = g["wall_clock_sec"].values[0]
            g_gpu = g.get("gpu_peak_alloc_MiB", pd.Series([None])).values[0]

            for scope_name in PREDICTION_SCOPES:
                if scope_name == "Global":
                    continue
                s = df[(df["scope"] == scope_name) & (df["lead_time_h"] == lt_h)]
                if s.empty:
                    continue
                s_time = s["wall_clock_sec"].values[0]
                speedup = g_time / s_time if s_time > 0 else float("inf")
                msg = f"  {scope_name:20s} @ {lt_h:3d}h: {speedup:.2f}× faster"
                if g_gpu is not None and "gpu_peak_alloc_MiB" in s.columns:
                    s_gpu = s["gpu_peak_alloc_MiB"].values[0]
                    gpu_ratio = g_gpu / s_gpu if s_gpu > 0 else float("inf")
                    msg += f"  |  {gpu_ratio:.2f}× less GPU mem"
                print(msg)
        print("-" * 70)


if __name__ == "__main__":
    benchmark()
