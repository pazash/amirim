"""
Compute forecast skill scores for the Aurora context-comparison experiment.

Baselines computed:
  1. Persistence  — use ERA5 at init time as the forecast for all lead times
                    ("tomorrow = today")
  2. Climatology  — per-grid-point mean temperature across all 10 events at
                    the same lead time (a "typical temperature for this region"
                    approximation from the available sample)

Metrics produced:
  • RMSE of each baseline
  • Skill Score (SS) = 1 − (MSE_model / MSE_baseline)
      SS > 0 → model beats baseline
      SS = 0 → same as baseline
      SS < 0 → worse than baseline

Usage:
    python scripts/compute_forecast_skill.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataloader import AuroraDataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Same bounding box used for evaluation in compare_global_vs_med_aurora.py
# ---------------------------------------------------------------------------
ORIGINAL_MED_BBOX = {
    "lon_min": 18.0, "lon_max": 61.75,
    "lat_max": 53.25, "lat_min": 25.5,
}

LEAD_TIMES_H = [6, 24, 72, 120]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.nanmean(diff ** 2)))


def main():
    csv_path = Path("aurora_context_comparison_rmse.csv")
    if not csv_path.exists():
        logger.error(f"{csv_path} not found. Run compare_global_vs_med_aurora.py --evaluate first.")
        return

    aurora_df = pd.read_csv(csv_path)
    aurora_df["init_time"] = pd.to_datetime(aurora_df["init_time"])
    aurora_df["valid_time"] = pd.to_datetime(aurora_df["valid_time"])

    init_times = sorted(aurora_df["init_time"].unique())
    logger.info(f"Found {len(init_times)} init times in RMSE CSV")

    cache_dir = Path("weather_data")
    data_pipeline = AuroraDataLoader(cache_dir=cache_dir)

    # -----------------------------------------------------------------------
    # 1. Compute persistence RMSE for each (init_time, lead_time)
    # -----------------------------------------------------------------------
    persistence_rows = []
    ground_truth_fields = {}  # (init_time, lt_h) → 2D array for climatology later

    for init_dt in init_times:
        init_dt = pd.Timestamp(init_dt).to_pydatetime()
        logger.info(f"Processing {init_dt} …")

        # ERA5 at init time (for persistence)
        init_ds = data_pipeline._fetch_era5_combined([init_dt], bbox=ORIGINAL_MED_BBOX)
        init_t2m = init_ds["t2m"].sel(time=init_dt.strftime("%Y-%m-%dT%H:00:00")).values

        # ERA5 at each valid time (ground truth)
        target_dts = [init_dt + timedelta(hours=lt) for lt in LEAD_TIMES_H]
        target_ds = data_pipeline._fetch_era5_combined(target_dts, bbox=ORIGINAL_MED_BBOX)

        for lt_h in LEAD_TIMES_H:
            valid_dt = init_dt + timedelta(hours=lt_h)
            valid_str = valid_dt.strftime("%Y-%m-%dT%H:00:00")

            try:
                tgt_t2m = target_ds["t2m"].sel(time=valid_str).values
            except KeyError:
                logger.warning(f"  Missing ground truth for {valid_str}, skipping.")
                continue

            # Store for climatology computation
            ground_truth_fields[(init_dt, lt_h)] = tgt_t2m

            # Persistence RMSE
            persist_rmse = rmse(init_t2m, tgt_t2m)
            persistence_rows.append({
                "init_time": init_dt,
                "lead_time_hours": lt_h,
                "persistence_rmse": persist_rmse,
            })

            logger.info(f"  Lt {lt_h:3d}h  |  Persistence RMSE = {persist_rmse:.3f} K")

    persist_df = pd.DataFrame(persistence_rows)

    # -----------------------------------------------------------------------
    # 2. Compute climatology RMSE
    #    Climatology = per-grid-point mean of ground truth across all events
    #    at the same lead time. This is a leave-one-out-like approach: for each
    #    event we use the mean of ALL events (including itself, which is a
    #    slight optimism but acceptable with 10 events).
    # -----------------------------------------------------------------------
    clim_rows = []

    for lt_h in LEAD_TIMES_H:
        # Collect all ground truth fields for this lead time
        fields = []
        init_dts_for_lt = []
        for (itime, lt), arr in ground_truth_fields.items():
            if lt == lt_h:
                fields.append(arr)
                init_dts_for_lt.append(itime)

        if not fields:
            continue

        clim_mean = np.nanmean(np.stack(fields, axis=0), axis=0)

        for idx, init_dt in enumerate(init_dts_for_lt):
            tgt = ground_truth_fields[(init_dt, lt_h)]
            clim_rmse_val = rmse(clim_mean, tgt)
            clim_rows.append({
                "init_time": init_dt,
                "lead_time_hours": lt_h,
                "climatology_rmse": clim_rmse_val,
            })

    clim_df = pd.DataFrame(clim_rows)

    # -----------------------------------------------------------------------
    # 3. Merge everything and compute skill scores
    # -----------------------------------------------------------------------
    # Merge persistence and climatology baselines
    baselines = persist_df.merge(clim_df, on=["init_time", "lead_time_hours"], how="outer")

    # Merge with Aurora results
    aurora_df_renamed = aurora_df.rename(columns={"value": "aurora_rmse"})
    merged = aurora_df_renamed.merge(
        baselines,
        on=["init_time", "lead_time_hours"],
        how="left",
    )

    # Skill scores: SS = 1 - (MSE_model / MSE_baseline)
    merged["ss_vs_persistence"] = 1 - (merged["aurora_rmse"] ** 2 / merged["persistence_rmse"] ** 2)
    merged["ss_vs_climatology"] = 1 - (merged["aurora_rmse"] ** 2 / merged["climatology_rmse"] ** 2)

    # -----------------------------------------------------------------------
    # 4. Save full results
    # -----------------------------------------------------------------------
    out_cols = [
        "init_time", "valid_time", "lead_time_hours", "model",
        "aurora_rmse", "persistence_rmse", "climatology_rmse",
        "ss_vs_persistence", "ss_vs_climatology",
    ]
    merged[out_cols].to_csv("aurora_forecast_skill_scores.csv", index=False)
    logger.info("Saved → aurora_forecast_skill_scores.csv")

    # -----------------------------------------------------------------------
    # 5. Print summary tables
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("AVERAGE RMSE BY SCOPE & LEAD TIME")
    print("=" * 100)

    summary = merged.groupby(["model", "lead_time_hours"]).agg(
        aurora_rmse=("aurora_rmse", "mean"),
        persistence_rmse=("persistence_rmse", "mean"),
        climatology_rmse=("climatology_rmse", "mean"),
    ).round(3)
    print(summary.to_string())

    print("\n" + "=" * 100)
    print("AVERAGE SKILL SCORES  (>0 = model beats baseline)")
    print("=" * 100)

    skill_summary = merged.groupby(["model", "lead_time_hours"]).agg(
        ss_vs_persistence=("ss_vs_persistence", "mean"),
        ss_vs_climatology=("ss_vs_climatology", "mean"),
    ).round(3)
    print(skill_summary.to_string())

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("""
  Skill Score (SS) = 1 − (MSE_model / MSE_baseline)

    SS > 0   → Model is BETTER than the baseline  (useful forecast)
    SS = 0   → Model equals the baseline           (no skill)
    SS < 0   → Model is WORSE than the baseline    (worse than naive guess)
    SS = 1   → Perfect forecast

  Persistence baseline: "tomorrow = today" (ERA5 at init time)
  Climatology baseline: mean temperature over all 10 events for that lead time
    """)

    # -----------------------------------------------------------------------
    # 6. Quick verdict per scope
    # -----------------------------------------------------------------------
    print("=" * 100)
    print("VERDICT PER SCOPE")
    print("=" * 100)

    for model_name in sorted(merged["model"].unique()):
        sub = merged[merged["model"] == model_name]
        avg_ss_persist = sub["ss_vs_persistence"].mean()
        avg_ss_clim = sub["ss_vs_climatology"].mean()
        beats_persist = (sub["ss_vs_persistence"] > 0).sum()
        total = len(sub)
        beats_clim = (sub["ss_vs_climatology"] > 0).sum()
        print(
            f"  {model_name:25s} | "
            f"Beats persistence {beats_persist}/{total} cases (avg SS={avg_ss_persist:+.3f}) | "
            f"Beats climatology {beats_clim}/{total} cases (avg SS={avg_ss_clim:+.3f})"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
