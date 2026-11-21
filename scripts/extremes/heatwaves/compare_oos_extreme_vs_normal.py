"""
Out-of-sample (2022) Extreme vs. Normal weather forecast degradation analysis
and visualization suite for Aurora 0.25° (Pretrained & Finetuned) vs. ECMWF HRES
and ERA5 Persistence.

Methodology
-----------
1. Loads ``oos_heatwave_evaluations.csv`` (12 EWB 2022 heatwave cases) and
   ``oos_normal_evaluations.csv`` (12 paired 2022 same-season normal windows
   with identical bounding boxes and durations).
2. Restricts both datasets to strictly out-of-sample rows (``is_out_of_sample == True``)
   and lead times ``<= 240h``, aligning all active models onto the common
   ``(case_id_number, init_time, valid_time)`` intersection within each regime.
3. Pairs Extreme and Normal steps 1-to-1 on their unique composite key
   ``(case_id_number, model_group, metric, start_lead_h, lead_time)`` where
   ``start_lead_h in {24, 72, 120}`` is the initialization offset before window
   onset. This avoids any Cartesian product across overlapping lead times from
   different onset buckets.
4. Evaluates physically meaningful cross-regime metrics:
   - **RootMeanSquaredError (RMSE)**: Absolute values, Extreme/Normal Ratio,
     Percentage Degradation ``((Extreme - Normal) / Normal) * 100``, and
     Excess Degradation relative to ECMWF HRES.
   - **RMSE Skill Scores**: Skill Score vs. ERA5 Persistence and vs. ECMWF HRES
     (``SS = 1 - MSE_model / MSE_baseline``) in Normal vs. Extreme regimes.
   - **Peak_Amplitude_Error (PAE) & Conditional_Bias_Extremes (CBE)**: Uses
     normal-weather PAE and CBE as a spatial-smoothing and topographic control
     to isolate net extreme amplitude collapse (``Delta_PAE = PAE_ext - PAE_norm``,
     ``Delta_CBE = CBE_ext - CBE_norm``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

FOCUSED_METRICS: tuple[str, ...] = (
    "RootMeanSquaredError",
    "Peak_Amplitude_Error",
    "Conditional_Bias_Extremes",
)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "aurora_pretrained": "Aurora 0.25 Pretrained",
    "aurora_finetuned": "Aurora 0.25 Finetuned",
    "HRES": "ECMWF HRES",
    "Persistence": "ERA5 Persistence",
}

MODEL_ORDER: list[str] = [
    "Aurora 0.25 Pretrained",
    "Aurora 0.25 Finetuned",
    "ECMWF HRES",
    "ERA5 Persistence",
]

MODEL_PALETTE: dict[str, str] = {
    "Aurora 0.25 Pretrained": "#1f77b4",
    "Aurora 0.25 Finetuned": "#2ca02c",
    "ECMWF HRES": "#d62728",
    "ERA5 Persistence": "#7f7f7f",
}

LEAD_BINS: list[int] = [0, 72, 144, 241]
LEAD_BIN_LABELS: list[str] = ["0-3 Days", "3-6 Days", "6-10 Days"]


def _clean_regime_df(
    csv_path: Path,
    regime_name: str,
    max_lead_hours: int = 240,
) -> pd.DataFrame:
    """
    Load a single regime evaluation CSV, filter to <= *max_lead_hours* and
    out-of-sample rows, enforce common (case_id_number, init_time, valid_time)
    targets across all present models, and derive ``start_lead_h``.
    """
    df = pd.read_csv(csv_path)
    df = df[df["lead_time"] <= max_lead_hours].copy()
    df = df.dropna(subset=["value"])
    df = df[df["metric"].isin(FOCUSED_METRICS)].copy()

    if "model_family" in df.columns:
        df["model_group"] = df["model_family"].map(
            lambda x: MODEL_DISPLAY_NAMES.get(str(x), str(x))
        )
    else:
        df["model_group"] = df["forecast_source"].astype(str)

    if "is_out_of_sample" in df.columns:
        aurora_mask = df["model_group"].str.contains("aurora", case=False, na=False)
        if bool((df.loc[aurora_mask, "is_out_of_sample"] == True).any()):  # noqa: E712
            df = df[df["is_out_of_sample"] == True].copy()  # noqa: E712

    active_groups = [g for g in MODEL_ORDER if g in set(df["model_group"].unique())]
    if not active_groups:
        raise ValueError(f"No valid model groups found in {csv_path}.")

    # Standardize timestamp strings before intersecting targets across models
    df["init_dt"] = pd.to_datetime(df["init_time"], format="ISO8601")
    df["valid_dt"] = pd.to_datetime(df["valid_time"], format="ISO8601")
    df["init_time"] = df["init_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["valid_time"] = df["valid_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    key_cols = ["case_id_number", "init_time", "valid_time"]
    common_targets: pd.DataFrame | None = None
    for group in active_groups:
        group_keys = df[df["model_group"] == group][key_cols].drop_duplicates()
        common_targets = (
            group_keys
            if common_targets is None
            else pd.merge(common_targets, group_keys, on=key_cols, how="inner")
        )

    if common_targets is None or common_targets.empty:
        raise ValueError(f"No common evaluation targets shared across models in {csv_path}.")

    df = pd.merge(df, common_targets, on=key_cols, how="inner")

    # Derive exact onset start_lead_h per (case_id_number, init_time)
    case_starts = df.groupby("case_id_number")["valid_dt"].transform("min")
    df["start_lead_h"] = (
        (case_starts - df["init_dt"]).dt.total_seconds() / 3600.0
    ).round().astype(int)
    df["event_offset_h"] = (
        (df["valid_dt"] - case_starts).dt.total_seconds() / 3600.0
    ).round().astype(int)
    df["regime"] = regime_name
    df["lead_time_bucket"] = pd.cut(
        df["lead_time"], bins=LEAD_BINS, labels=LEAD_BIN_LABELS, right=False
    )

    logger.info(
        f"[{regime_name}] Loaded {len(common_targets)} common (case, init, valid) targets "
        f"across {df['case_id_number'].nunique()} cases for models={active_groups}."
    )
    return df


def load_paired_data(
    extreme_csv: Path,
    normal_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Extreme and Normal DataFrames and construct a strictly 1-to-1 paired
    DataFrame keyed on ``(case_id_number, model_group, metric, start_lead_h, lead_time)``.

    Returns:
        (combined_long_df, paired_wide_df)
    """
    df_ext = _clean_regime_df(extreme_csv, regime_name="Extreme (Heatwave)")
    df_norm = _clean_regime_df(normal_csv, regime_name="Normal Weather")

    pair_keys = [
        "case_id_number",
        "model_group",
        "metric",
        "start_lead_h",
        "lead_time",
    ]

    # Deduplicate defensively in case an input CSV had duplicate evaluation runs
    df_ext_dedup = df_ext.drop_duplicates(subset=pair_keys).copy()
    df_norm_dedup = df_norm.drop_duplicates(subset=pair_keys).copy()

    paired = pd.merge(
        df_ext_dedup[
            pair_keys + ["value", "init_time", "valid_time", "lead_time_bucket"]
        ],
        df_norm_dedup[pair_keys + ["value", "init_time", "valid_time"]],
        on=pair_keys,
        how="inner",
        suffixes=("_extreme", "_normal"),
    )
    if paired.empty:
        raise ValueError(
            "No matching (case_id_number, model_group, metric, start_lead_h, lead_time) "
            "steps found between extreme and normal CSVs."
        )

    paired["diff_extreme_minus_normal"] = (
        paired["value_extreme"] - paired["value_normal"]
    )
    paired["ratio_extreme_over_normal"] = np.where(
        np.abs(paired["value_normal"]) > 1e-8,
        paired["value_extreme"] / paired["value_normal"],
        np.nan,
    )
    paired["pct_degradation"] = np.where(
        np.abs(paired["value_normal"]) > 1e-8,
        ((paired["value_extreme"] - paired["value_normal"]) / paired["value_normal"])
        * 100.0,
        np.nan,
    )

    # Filter long DataFrames to the exact paired key intersection
    matched_keys = paired[pair_keys].drop_duplicates()
    combined_long = pd.concat(
        [
            pd.merge(df_ext_dedup, matched_keys, on=pair_keys, how="inner"),
            pd.merge(df_norm_dedup, matched_keys, on=pair_keys, how="inner"),
        ],
        ignore_index=True,
    )

    logger.info(
        f"Constructed {len(paired)} strictly paired 1-to-1 evaluation rows across "
        f"{paired['case_id_number'].nunique()} matched 2022 cases."
    )
    return combined_long, paired


def compute_summary_tables(
    combined_long: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Compute and save CSV summary tables for RMSE degradation, Skill Scores vs.
    Persistence and HRES, smoothing-controlled PAE/CBE tail bias, and per-case
    paired statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. RMSE Degradation by (model_group, lead_time)
    rmse_paired = paired[paired["metric"] == "RootMeanSquaredError"].copy()
    rmse_by_lt = (
        rmse_paired.groupby(["model_group", "lead_time"], as_index=False)
        .agg(
            n_samples=("value_extreme", "count"),
            rmse_normal=("value_normal", "mean"),
            rmse_extreme=("value_extreme", "mean"),
            rmse_diff=("diff_extreme_minus_normal", "mean"),
            paired_ratio_mean=("ratio_extreme_over_normal", "mean"),
        )
    )
    rmse_by_lt["rmse_ratio"] = (
        rmse_by_lt["rmse_extreme"] / rmse_by_lt["rmse_normal"]
    )
    rmse_by_lt["rmse_degradation_pct"] = (
        (rmse_by_lt["rmse_extreme"] - rmse_by_lt["rmse_normal"])
        / rmse_by_lt["rmse_normal"]
    ) * 100.0

    # Attach HRES reference degradation at each lead_time to compute excess degradation
    hres_ref = rmse_by_lt[rmse_by_lt["model_group"] == "ECMWF HRES"][
        ["lead_time", "rmse_normal", "rmse_extreme", "rmse_degradation_pct"]
    ].rename(
        columns={
            "rmse_normal": "hres_rmse_normal",
            "rmse_extreme": "hres_rmse_extreme",
            "rmse_degradation_pct": "hres_degradation_pct",
        }
    )
    rmse_by_lt = pd.merge(rmse_by_lt, hres_ref, on="lead_time", how="left")
    rmse_by_lt["gap_vs_hres_normal"] = (
        rmse_by_lt["rmse_normal"] - rmse_by_lt["hres_rmse_normal"]
    )
    rmse_by_lt["gap_vs_hres_extreme"] = (
        rmse_by_lt["rmse_extreme"] - rmse_by_lt["hres_rmse_extreme"]
    )
    rmse_by_lt["double_diff_vs_hres"] = (
        rmse_by_lt["gap_vs_hres_extreme"] - rmse_by_lt["gap_vs_hres_normal"]
    )
    rmse_by_lt["excess_degradation_vs_hres_pct"] = (
        rmse_by_lt["rmse_degradation_pct"] - rmse_by_lt["hres_degradation_pct"]
    )
    rmse_by_lt.to_csv(output_dir / "rmse_degradation_by_lead_time.csv", index=False)

    # 2. Skill Score Comparison vs. Persistence & HRES by (model_group, regime, lead_time)
    rmse_long = combined_long[
        combined_long["metric"] == "RootMeanSquaredError"
    ].copy()
    rmse_regime_lt = (
        rmse_long.groupby(["regime", "model_group", "lead_time"], as_index=False)
        .agg(mean_rmse=("value", "mean"), mean_mse=("value", lambda s: float(np.mean(s**2))))
    )

    persist_mse = rmse_regime_lt[
        rmse_regime_lt["model_group"] == "ERA5 Persistence"
    ][["regime", "lead_time", "mean_rmse", "mean_mse"]].rename(
        columns={"mean_rmse": "persist_rmse", "mean_mse": "persist_mse"}
    )
    hres_mse = rmse_regime_lt[rmse_regime_lt["model_group"] == "ECMWF HRES"][
        ["regime", "lead_time", "mean_rmse", "mean_mse"]
    ].rename(columns={"mean_rmse": "hres_rmse", "mean_mse": "hres_mse"})

    skill_df = pd.merge(
        rmse_regime_lt, persist_mse, on=["regime", "lead_time"], how="left"
    )
    skill_df = pd.merge(
        skill_df, hres_mse, on=["regime", "lead_time"], how="left"
    )
    skill_df["ss_vs_persistence"] = 1.0 - (
        skill_df["mean_mse"] / skill_df["persist_mse"]
    )
    skill_df["ss_vs_hres"] = 1.0 - (skill_df["mean_mse"] / skill_df["hres_mse"])
    skill_df.to_csv(output_dir / "skill_score_comparison.csv", index=False)

    # 3. Tail & Peak Bias Degradation (PAE and CBE, controlling for normal spatial smoothing)
    tail_paired = paired[
        paired["metric"].isin(["Peak_Amplitude_Error", "Conditional_Bias_Extremes"])
    ].copy()
    tail_by_lt = (
        tail_paired.groupby(["metric", "model_group", "lead_time"], as_index=False)
        .agg(
            n_samples=("value_extreme", "count"),
            bias_normal=("value_normal", "mean"),
            bias_extreme=("value_extreme", "mean"),
            net_extreme_shift=("diff_extreme_minus_normal", "mean"),
            net_extreme_shift_std=("diff_extreme_minus_normal", "std"),
        )
    )
    tail_by_lt.to_csv(output_dir / "tail_bias_degradation.csv", index=False)

    # 4. Per-Case Paired Summary (across the 12 2022 regions)
    case_summary = (
        paired.groupby(["case_id_number", "metric", "model_group"], as_index=False)
        .agg(
            n_steps=("value_extreme", "count"),
            mean_normal=("value_normal", "mean"),
            mean_extreme=("value_extreme", "mean"),
            mean_diff_ext_minus_norm=("diff_extreme_minus_normal", "mean"),
        )
    )
    case_summary["ratio_ext_over_norm"] = np.where(
        np.abs(case_summary["mean_normal"]) > 1e-8,
        case_summary["mean_extreme"] / case_summary["mean_normal"],
        np.nan,
    )
    case_summary.to_csv(output_dir / "paired_case_degradation.csv", index=False)

    # Print concise executive table to console
    overall_summary = (
        paired.groupby(["metric", "model_group"], as_index=False)
        .agg(
            n_pairs=("value_extreme", "count"),
            normal_mean=("value_normal", "mean"),
            extreme_mean=("value_extreme", "mean"),
            paired_diff=("diff_extreme_minus_normal", "mean"),
        )
    )
    overall_summary["degradation_pct"] = np.where(
        overall_summary["metric"] == "RootMeanSquaredError",
        (
            (overall_summary["extreme_mean"] - overall_summary["normal_mean"])
            / overall_summary["normal_mean"]
        )
        * 100.0,
        np.nan,
    )
    logger.info(
        "\n=== 2022 Out-of-Sample Extreme vs. Normal Paired Summary (<= 240h) ===\n"
        + overall_summary.to_string(index=False)
    )

    return {
        "rmse_by_lt": rmse_by_lt,
        "skill_df": skill_df,
        "tail_by_lt": tail_by_lt,
        "case_summary": case_summary,
        "overall_summary": overall_summary,
    }


def _active_models(
    df: pd.DataFrame, include_persistence: bool, col: str = "model_group"
) -> tuple[list[str], dict[str, str]]:
    present = set(df[col].unique())
    if not include_persistence:
        present.discard("ERA5 Persistence")
    order = [m for m in MODEL_ORDER if m in present]
    palette = {m: MODEL_PALETTE[m] for m in order if m in MODEL_PALETTE}
    return order, palette


def plot_rmse_normal_vs_extreme_curves(
    combined_long: pd.DataFrame,
    output_dir: Path,
    include_persistence: bool = False,
) -> None:
    """Plot 1: Overlaid Normal (dashed) vs. Extreme (solid) RMSE curves over lead time."""
    df_rmse = combined_long[
        combined_long["metric"] == "RootMeanSquaredError"
    ].copy()
    order, palette = _active_models(df_rmse, include_persistence)
    df_rmse = df_rmse[df_rmse["model_group"].isin(order)]

    plt.figure(figsize=(11, 6.5))
    sns.lineplot(
        data=df_rmse,
        x="lead_time",
        y="value",
        hue="model_group",
        hue_order=order,
        palette=palette,
        style="regime",
        style_order=["Extreme (Heatwave)", "Normal Weather"],
        markers=True,
        dashes=True,
        linewidth=2.2,
        err_style="band",
    )
    plt.title(
        "Out-of-Sample (2022) 2m Temperature RMSE: Extreme Heatwaves vs. Normal Weather",
        fontsize=13,
    )
    plt.xlabel("Lead Time (Hours)", fontsize=12)
    plt.ylabel("RMSE (K)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Model / Regime")
    plt.tight_layout()
    plt.savefig(output_dir / "rmse_normal_vs_extreme_curves.png", dpi=300)
    plt.close()


def plot_rmse_degradation_ratio_and_pct(
    rmse_by_lt: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
    include_persistence: bool = False,
) -> None:
    """
    Plot 2: 1x2 panel of (Left) RMSE Ratio (Extreme / Normal) vs. Lead Time and
    (Right) RMSE Percentage Degradation by 3-day Lead-Time Bucket.
    """
    order, palette = _active_models(rmse_by_lt, include_persistence)
    sub_lt = rmse_by_lt[rmse_by_lt["model_group"].isin(order)].copy()

    rmse_paired = paired[
        (paired["metric"] == "RootMeanSquaredError")
        & (paired["model_group"].isin(order))
    ].copy()
    bucket_agg = (
        rmse_paired.groupby(
            ["model_group", "lead_time_bucket"], observed=False, as_index=False
        )
        .agg(
            rmse_normal=("value_normal", "mean"),
            rmse_extreme=("value_extreme", "mean"),
        )
    )
    bucket_agg["degradation_pct"] = (
        (bucket_agg["rmse_extreme"] - bucket_agg["rmse_normal"])
        / bucket_agg["rmse_normal"]
    ) * 100.0

    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: RMSE Ratio over Lead Time
    sns.lineplot(
        ax=axes[0],
        data=sub_lt,
        x="lead_time",
        y="rmse_ratio",
        hue="model_group",
        hue_order=order,
        palette=palette,
        marker="o",
        linewidth=2.2,
    )
    axes[0].axhline(
        1.0,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="No Degradation (Ratio = 1.0)",
    )
    axes[0].set_title(
        "RMSE Degradation Ratio (Extreme / Normal) vs. Lead Time", fontsize=12.5
    )
    axes[0].set_xlabel("Lead Time (Hours)")
    axes[0].set_ylabel("RMSE Ratio (Extreme / Normal)")
    axes[0].legend(title="Model")

    # Right: Percentage Degradation by 3-Day Lead Time Bucket
    sns.barplot(
        ax=axes[1],
        data=bucket_agg,
        x="lead_time_bucket",
        y="degradation_pct",
        hue="model_group",
        hue_order=order,
        palette=palette,
        alpha=0.88,
    )
    axes[1].axhline(0.0, color="black", linestyle="-", linewidth=1.2)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[1].set_title(
        "RMSE Percentage Degradation by Lead-Time Window", fontsize=12.5
    )
    axes[1].set_xlabel("Lead Time Window")
    axes[1].set_ylabel("RMSE Degradation ((Extreme - Normal) / Normal)")
    axes[1].legend(title="Model")

    plt.tight_layout()
    plt.savefig(output_dir / "rmse_degradation_ratio_and_pct.png", dpi=300)
    plt.close()


def plot_aurora_vs_hres_gap_and_skill(
    rmse_by_lt: pd.DataFrame,
    skill_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot 3: 1x2 panel of:
      - Left: RMSE Difference (Aurora - ECMWF HRES) in Normal vs. Extreme Weather
              (Negative = Aurora outperforms HRES).
      - Right: Forecast Skill Score vs. ERA5 Persistence (1 - MSE_model / MSE_persist)
               in Normal vs. Extreme Weather.
    """
    aurora_order = [
        m
        for m in ("Aurora 0.25 Pretrained", "Aurora 0.25 Finetuned")
        if m in set(rmse_by_lt["model_group"].unique())
    ]
    aurora_palette = {m: MODEL_PALETTE[m] for m in aurora_order}

    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Aurora - HRES RMSE Gap in Normal vs Extreme
    if aurora_order:
        gap_rows = []
        for _, row in rmse_by_lt[
            rmse_by_lt["model_group"].isin(aurora_order)
        ].iterrows():
            gap_rows.append(
                {
                    "model_group": row["model_group"],
                    "lead_time": row["lead_time"],
                    "regime": "Extreme (Heatwave)",
                    "rmse_gap_vs_hres": row["gap_vs_hres_extreme"],
                }
            )
            gap_rows.append(
                {
                    "model_group": row["model_group"],
                    "lead_time": row["lead_time"],
                    "regime": "Normal Weather",
                    "rmse_gap_vs_hres": row["gap_vs_hres_normal"],
                }
            )
        gap_df = pd.DataFrame(gap_rows)
        sns.lineplot(
            ax=axes[0],
            data=gap_df,
            x="lead_time",
            y="rmse_gap_vs_hres",
            hue="model_group",
            hue_order=aurora_order,
            palette=aurora_palette,
            style="regime",
            style_order=["Extreme (Heatwave)", "Normal Weather"],
            markers=True,
            dashes=True,
            linewidth=2.2,
        )
    axes[0].axhline(
        0.0, color="black", linestyle="--", alpha=0.6, label="Parity with ECMWF HRES"
    )
    axes[0].set_title(
        "Aurora vs. ECMWF HRES RMSE Gap (Negative = Aurora Better)", fontsize=12.5
    )
    axes[0].set_xlabel("Lead Time (Hours)")
    axes[0].set_ylabel("RMSE Difference: Aurora - HRES (K)")
    axes[0].legend(title="Model / Regime")

    # Right: Skill Score vs. ERA5 Persistence
    skill_order, skill_palette = _active_models(
        skill_df, include_persistence=False
    )
    sub_skill = skill_df[skill_df["model_group"].isin(skill_order)].copy()
    sns.lineplot(
        ax=axes[1],
        data=sub_skill,
        x="lead_time",
        y="ss_vs_persistence",
        hue="model_group",
        hue_order=skill_order,
        palette=skill_palette,
        style="regime",
        style_order=["Extreme (Heatwave)", "Normal Weather"],
        markers=True,
        dashes=True,
        linewidth=2.2,
    )
    axes[1].axhline(
        0.0,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Zero Skill (Equals Persistence)",
    )
    axes[1].set_title(
        "Forecast Skill Score vs. ERA5 Persistence (1 - MSE / MSE_persist)",
        fontsize=12.5,
    )
    axes[1].set_xlabel("Lead Time (Hours)")
    axes[1].set_ylabel("Skill Score vs. Persistence")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Model / Regime")

    plt.tight_layout()
    plt.savefig(output_dir / "aurora_vs_hres_gap_and_skill.png", dpi=300)
    plt.close()


def plot_tail_bias_smoothing_controlled(
    combined_long: pd.DataFrame,
    tail_by_lt: pd.DataFrame,
    output_dir: Path,
    include_persistence: bool = False,
) -> None:
    """
    Plot 4: 2x2 panel for Peak Amplitude Error (PAE) and Conditional Bias of
    Extremes (CBE):
      - Top row: Raw Normal (dashed) vs. Extreme (solid) bias curves over lead time.
      - Bottom row: Smoothing-corrected Net Extreme Shift (Extreme - Normal),
        isolating extra peak/tail underestimation during heatwaves beyond normal
        spatial smoothing.
    """
    order, palette = _active_models(combined_long, include_persistence)
    metrics_info = [
        ("Peak_Amplitude_Error", "Peak Amplitude Error (PAE, K)"),
        ("Conditional_Bias_Extremes", "Top-10% Conditional Bias (CBE, K)"),
    ]

    _, axes = plt.subplots(2, 2, figsize=(16, 11))

    for col_idx, (m_key, m_label) in enumerate(metrics_info):
        sub_long = combined_long[
            (combined_long["metric"] == m_key)
            & (combined_long["model_group"].isin(order))
        ]
        sub_shift = tail_by_lt[
            (tail_by_lt["metric"] == m_key)
            & (tail_by_lt["model_group"].isin(order))
        ]

        # Top: Raw Normal vs Extreme
        sns.lineplot(
            ax=axes[0, col_idx],
            data=sub_long,
            x="lead_time",
            y="value",
            hue="model_group",
            hue_order=order,
            palette=palette,
            style="regime",
            style_order=["Extreme (Heatwave)", "Normal Weather"],
            markers=True,
            dashes=True,
            linewidth=2.0,
            err_style="band",
        )
        axes[0, col_idx].axhline(0.0, color="black", linestyle="--", alpha=0.5)
        axes[0, col_idx].set_title(
            f"Raw {m_label}: Extreme vs. Normal", fontsize=12
        )
        axes[0, col_idx].set_xlabel("Lead Time (Hours)")
        axes[0, col_idx].set_ylabel(m_label)
        axes[0, col_idx].legend(title="Model / Regime", fontsize=9)

        # Bottom: Net Extreme Shift (Extreme - Normal)
        sns.lineplot(
            ax=axes[1, col_idx],
            data=sub_shift,
            x="lead_time",
            y="net_extreme_shift",
            hue="model_group",
            hue_order=order,
            palette=palette,
            marker="o",
            linewidth=2.2,
        )
        axes[1, col_idx].axhline(
            0.0,
            color="black",
            linestyle="--",
            alpha=0.5,
            label="No Extra Extreme Bias (0.0 K)",
        )
        axes[1, col_idx].set_title(
            f"Smoothing-Controlled Net Extreme Shift: Δ{m_key.split('_')[0]} (Extreme − Normal)",
            fontsize=12,
        )
        axes[1, col_idx].set_xlabel("Lead Time (Hours)")
        axes[1, col_idx].set_ylabel("Net Extreme Bias Shift (K)")
        axes[1, col_idx].legend(title="Model", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "tail_bias_smoothing_controlled.png", dpi=300)
    plt.close()


def plot_paired_case_degradation_boxplots(
    paired: pd.DataFrame,
    output_dir: Path,
    include_persistence: bool = False,
) -> None:
    """
    Plot 5: 1x3 boxplots of paired step-by-step degradation (Extreme - Normal)
    across the 3-day lead-time buckets for RMSE, PAE, and CBE.
    """
    order, palette = _active_models(paired, include_persistence)
    sub_paired = paired[paired["model_group"].isin(order)].copy()

    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = {
        "RootMeanSquaredError": "ΔRMSE (Extreme − Normal, K)\n[> 0 means higher error in extreme]",
        "Peak_Amplitude_Error": "ΔPAE (Extreme − Normal, K)\n[< 0 means stronger peak underprediction]",
        "Conditional_Bias_Extremes": "ΔCBE (Extreme − Normal, K)\n[< 0 means stronger warm-tail underprediction]",
    }

    for idx, m_key in enumerate(FOCUSED_METRICS):
        m_df = sub_paired[sub_paired["metric"] == m_key]
        sns.boxplot(
            ax=axes[idx],
            data=m_df,
            x="lead_time_bucket",
            y="diff_extreme_minus_normal",
            hue="model_group",
            hue_order=order,
            palette=palette,
            showfliers=False,
        )
        axes[idx].axhline(0.0, color="black", linestyle="--", alpha=0.6)
        axes[idx].set_title(titles[m_key], fontsize=11.5)
        axes[idx].set_xlabel("Lead Time Window")
        axes[idx].set_ylabel("Paired Difference: Extreme − Normal (K)")
        if idx < 2:
            axes[idx].get_legend().remove()
        else:
            axes[idx].legend(
                bbox_to_anchor=(1.02, 1), loc="upper left", title="Model"
            )

    plt.tight_layout()
    plt.savefig(output_dir / "paired_case_degradation_boxplot.png", dpi=300)
    plt.close()


def generate_all_plots(
    combined_long: pd.DataFrame,
    paired: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    include_persistence: bool = False,
) -> None:
    """Render all 5 comparative chart sets into *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_rmse_normal_vs_extreme_curves(
        combined_long, output_dir, include_persistence=include_persistence
    )
    plot_rmse_degradation_ratio_and_pct(
        tables["rmse_by_lt"],
        paired,
        output_dir,
        include_persistence=include_persistence,
    )
    plot_aurora_vs_hres_gap_and_skill(
        tables["rmse_by_lt"], tables["skill_df"], output_dir
    )
    plot_tail_bias_smoothing_controlled(
        combined_long,
        tables["tail_by_lt"],
        output_dir,
        include_persistence=include_persistence,
    )
    plot_paired_case_degradation_boxplots(
        paired, output_dir, include_persistence=include_persistence
    )
    logger.info(f"Saved plot suite to {output_dir.resolve()}")


def generate_single_model_comparison_plots(
    paired: pd.DataFrame,
    aurora_model: str,
    output_dir: Path,
) -> None:
    """
    Generate clean, single-model ``compare_extreme_vs_normal.py``-style plots
    and CSVs comparing one Aurora variant (*aurora_model*) against ECMWF HRES.
    """
    present_models = set(paired["model_group"].unique())
    if aurora_model not in present_models:
        logger.warning(f"Skipping single-model plots for {aurora_model}: not in paired data.")
        return

    has_hres = "ECMWF HRES" in present_models
    order = [aurora_model, "ECMWF HRES"] if has_hres else [aurora_model]
    palette = {m: MODEL_PALETTE[m] for m in order if m in MODEL_PALETTE}

    sub = paired[paired["model_group"].isin(order)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. RMSE Aggregation
    rmse_df = (
        sub[sub["metric"] == "RootMeanSquaredError"]
        .groupby(["model_group", "lead_time"], as_index=False)
        .agg(
            value_extreme=("value_extreme", "mean"),
            value_normal=("value_normal", "mean"),
        )
    )
    if rmse_df.empty:
        logger.warning(f"No RMSE data for {aurora_model}; skipping single-model plots.")
        return

    rmse_df["rmse_ratio"] = rmse_df["value_extreme"] / rmse_df["value_normal"]
    rmse_df["rmse_degradation_pct"] = (
        (rmse_df["value_extreme"] - rmse_df["value_normal"])
        / rmse_df["value_normal"]
    ) * 100.0
    rmse_df.to_csv(output_dir / "rmse_comparison.csv", index=False)

    # 2. Peak Amplitude Error Aggregation
    peak_df = (
        sub[sub["metric"] == "Peak_Amplitude_Error"]
        .groupby(["model_group", "lead_time"], as_index=False)
        .agg(
            value_extreme=("value_extreme", "mean"),
            value_normal=("value_normal", "mean"),
        )
    )
    if not peak_df.empty:
        peak_df["peak_bias_diff"] = (
            peak_df["value_extreme"] - peak_df["value_normal"]
        )
        peak_df.to_csv(output_dir / "peak_error_comparison.csv", index=False)

    # Plot 1: RMSE Ratio (Extreme / Normal)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=rmse_df,
        x="lead_time",
        y="rmse_ratio",
        hue="model_group",
        hue_order=order,
        palette=palette,
        marker="o",
        linewidth=2,
    )
    plt.axhline(1.0, color="black", linestyle="--", alpha=0.6, label="Equal Performance")
    plt.title(
        f"RMSE Ratio (Extreme / Normal Weather) — {aurora_model} vs. HRES\n"
        "Higher ratio = model struggles more with extremes",
        fontsize=13,
    )
    plt.xlabel("Lead Time (Hours)", fontsize=12)
    plt.ylabel("RMSE Ratio", fontsize=12)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_dir / "rmse_ratio.png", dpi=300)
    plt.close()

    # Plot 2: Peak Amplitude Error (Extreme vs Normal)
    if not peak_df.empty:
        peak_melt = pd.melt(
            peak_df,
            id_vars=["model_group", "lead_time"],
            value_vars=["value_extreme", "value_normal"],
            var_name="regime",
            value_name="peak_error",
        )
        peak_melt["regime"] = peak_melt["regime"].replace(
            {
                "value_extreme": "Extreme (Heatwave)",
                "value_normal": "Normal Weather",
            }
        )
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=peak_melt,
            x="lead_time",
            y="peak_error",
            hue="model_group",
            hue_order=order,
            palette=palette,
            style="regime",
            style_order=["Extreme (Heatwave)", "Normal Weather"],
            markers=True,
            dashes=True,
            linewidth=2,
        )
        plt.axhline(0.0, color="black", linestyle="-", alpha=0.3)
        plt.title(
            f"Mean Peak Amplitude Error (Forecast - Target) — {aurora_model} vs. HRES\n"
            "Negative values indicate underprediction of peaks",
            fontsize=13,
        )
        plt.xlabel("Lead Time (Hours)", fontsize=12)
        plt.ylabel("Peak Error (K)", fontsize=12)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Model / Regime")
        plt.tight_layout()
        plt.savefig(output_dir / "peak_amplitude_error.png", dpi=300)
        plt.close()

    # Plot 3: Single Aurora Model RMSE (Normal vs Extreme)
    aurora_rmse = rmse_df[rmse_df["model_group"] == aurora_model]
    if not aurora_rmse.empty:
        aurora_melt = pd.melt(
            aurora_rmse,
            id_vars=["lead_time"],
            value_vars=["value_normal", "value_extreme"],
            var_name="regime",
            value_name="rmse",
        )
        aurora_melt["regime"] = aurora_melt["regime"].replace(
            {
                "value_normal": "Normal Weather",
                "value_extreme": "Extreme (Heatwave)",
            }
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=aurora_melt,
            x="lead_time",
            y="rmse",
            hue="regime",
            hue_order=["Normal Weather", "Extreme (Heatwave)"],
            marker="o",
            linewidth=2,
        )
        plt.title(f"{aurora_model} RMSE: Normal vs Extreme Weather", fontsize=14)
        plt.xlabel("Lead Time (Hours)", fontsize=12)
        plt.ylabel("RMSE (K)", fontsize=12)
        plt.legend(title="Regime")
        plt.tight_layout()
        plt.savefig(output_dir / "aurora_rmse_normal_vs_extreme.png", dpi=300)
        plt.close()

    # Plot 4: Aurora vs HRES RMSE Difference (Normal vs Extreme)
    if has_hres:
        ext_piv = rmse_df.pivot(
            index="lead_time", columns="model_group", values="value_extreme"
        ).reset_index()
        norm_piv = rmse_df.pivot(
            index="lead_time", columns="model_group", values="value_normal"
        ).reset_index()
        if aurora_model in ext_piv.columns and "ECMWF HRES" in ext_piv.columns:
            diff_df = pd.DataFrame(
                {
                    "lead_time": norm_piv["lead_time"],
                    "Normal Weather": norm_piv[aurora_model] - norm_piv["ECMWF HRES"],
                    "Extreme (Heatwave)": ext_piv[aurora_model] - ext_piv["ECMWF HRES"],
                }
            )
            diff_melt = pd.melt(
                diff_df,
                id_vars=["lead_time"],
                value_vars=["Normal Weather", "Extreme (Heatwave)"],
                var_name="regime",
                value_name="rmse_diff",
            )
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=diff_melt,
                x="lead_time",
                y="rmse_diff",
                hue="regime",
                hue_order=["Normal Weather", "Extreme (Heatwave)"],
                marker="o",
                linewidth=2,
            )
            plt.axhline(0.0, color="black", linestyle="--", alpha=0.6, label="Equal Performance")
            plt.title(
                f"{aurora_model} vs ECMWF HRES RMSE Difference\n"
                f"Negative values = {aurora_model} is better than HRES",
                fontsize=13,
            )
            plt.xlabel("Lead Time (Hours)", fontsize=12)
            plt.ylabel("RMSE Difference (Aurora - HRES, K)", fontsize=12)
            plt.legend(title="Regime")
            plt.tight_layout()
            plt.savefig(output_dir / "aurora_vs_hres_rmse_diff.png", dpi=300)
            plt.close()

    # Plot 5: RMSE Percentage Degradation Bar Chart
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        data=rmse_df,
        x="lead_time",
        y="rmse_degradation_pct",
        hue="model_group",
        hue_order=order,
        palette=palette,
        alpha=0.88,
    )
    plt.axhline(0.0, color="black", linestyle="-", linewidth=1.4)
    plt.title(
        f"RMSE Percentage Degradation (Extreme vs Normal Weather) — {aurora_model} vs. HRES\n"
        "How much higher is the error during heatwaves?",
        fontsize=13,
    )
    plt.xlabel("Lead Time (Hours)", fontsize=12)
    plt.ylabel("RMSE Degradation (%)", fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.tick_params(axis="x", labelsize=9)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_dir / "rmse_percentage_degradation.png", dpi=300)
    plt.close()

    logger.info(f"Saved single-model comparison plots for {aurora_model} to {output_dir.resolve()}")


def run_comparison(
    extreme_csv: str | Path = Path("oos_heatwave_evaluations.csv"),
    normal_csv: str | Path = Path("oos_normal_evaluations.csv"),
    output_dir: str | Path = Path("plots") / "oos_extreme_vs_normal",
) -> dict[str, pd.DataFrame]:
    """Run the full 2022 OOS Extreme vs. Normal degradation analysis and plot suite."""
    ext_path = Path(extreme_csv)
    norm_path = Path(normal_csv)
    out_path = Path(output_dir)

    if not ext_path.exists():
        raise FileNotFoundError(
            f"Extreme evaluation CSV not found at {ext_path.resolve()}. "
            "Run oos_heatwave_evaluator.py first."
        )
    if not norm_path.exists():
        raise FileNotFoundError(
            f"Normal evaluation CSV not found at {norm_path.resolve()}. "
            "Run oos_normal_weather_evaluator.py first."
        )

    combined_long, paired = load_paired_data(ext_path, norm_path)
    tables = compute_summary_tables(combined_long, paired, out_path)

    generate_all_plots(
        combined_long, paired, tables, out_path, include_persistence=False
    )
    generate_all_plots(
        combined_long,
        paired,
        tables,
        out_path / "with_persistence",
        include_persistence=True,
    )

    for aurora_label, folder_name in (
        ("Aurora 0.25 Finetuned", "aurora_finetuned"),
        ("Aurora 0.25 Pretrained", "aurora_pretrained"),
    ):
        generate_single_model_comparison_plots(
            paired, aurora_label, out_path / folder_name
        )

    return tables


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare 2022 Out-of-Sample Extreme Heatwaves vs. Paired 2022 Normal "
            "Weather Baselines for Aurora 0.25 (Pretrained & Finetuned) vs. HRES "
            "and ERA5 Persistence."
        )
    )
    parser.add_argument(
        "--extreme-csv",
        type=Path,
        default=Path("oos_heatwave_evaluations.csv"),
        help="Path to 2022 out-of-sample heatwave evaluation CSV.",
    )
    parser.add_argument(
        "--normal-csv",
        type=Path,
        default=Path("oos_normal_evaluations.csv"),
        help="Path to 2022 out-of-sample normal evaluation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots") / "oos_extreme_vs_normal",
        help="Output directory for summary CSVs and plots.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run_comparison(
            extreme_csv=args.extreme_csv,
            normal_csv=args.normal_csv,
            output_dir=args.output_dir,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))


if __name__ == "__main__":
    main()
