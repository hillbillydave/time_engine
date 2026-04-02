import os
import math
import time
import random
import numpy as np
import pandas as pd

from time_engine_v2 import TimeEngineV2, Metric


# ============================================================
# CONFIG
# ============================================================

BASE_CSV = "cms_d0_odim_processed.csv"

DATASET_CSVS = [
    BASE_CSV,
]

NOISE_LEVELS = [0.01, 0.02, 0.05]

METRIC_SWEEP_WEIGHTS = [
    (10, 5),
    (20, 10),
    (40, 20),
    (80, 40),
]

MC_NOISE_RUNS = 200
MC_NOISE_MAX = 0.10

MC_METRIC_RUNS = 100
AX_RANGE = (5.0, 80.0)
BX_RANGE = (2.5, 40.0)

NOISE_RESULTS_CSV = "noise_stress_results.csv"
METRIC_SWEEP_RESULTS_CSV = "metric_sweep_results.csv"
MC_NOISE_RESULTS_CSV = "mc_noise_stress_results.csv"
MC_METRIC_RESULTS_CSV = "mc_metric_stress_results.csv"
MERGED_DATASET_CSV = "cms_merged_datasets.csv"
MERGED_RUN_OUTPUT_CSV = "time_engine_v2_output_merged.csv"


# ============================================================
# UTILS
# ============================================================

def inject_noise_sigma(df, noise_level):
    df_noisy = df.copy()
    noise = np.random.normal(0, noise_level, size=len(df))
    df_noisy["sigma"] = df["sigma"] * (1.0 + noise)
    return df_noisy


def make_metric_O2(ax, bx):
    def metric_fn(X):
        return np.diag([
            ax / (X["x"]**2 + 1e-12),
            bx / (X["Q2"]**2 + 1e-12),
            1.0,
            0.1,
            0.1,
            0.1
        ])
    return Metric(f"O2_ax{ax:.2f}_bx{bx:.2f}", metric_fn)


def run_engine_on_csv(csv_path):
    """ FIXED: csv_path is now correct """
    t0 = time.time()
    engine = TimeEngineV2(processed_csv=csv_path)
    engine.run_all()
    runtime = time.time() - t0
    return engine.df, runtime


def summarize_run(df, runtime_sec, extra=None):
    last = df.iloc[-1]
    summary = {
        "final_t_O1": last["t_O1"],
        "final_t_O2": last["t_O2"],
        "final_t_O3": last["t_O3"],
        "final_dt_O2_O1": last["dt_O2_O1"],
        "final_dt_O3_O1": last["dt_O3_O1"],
        "final_dt_O3_O2": last["dt_O3_O2"],
        "num_horizon_O2_O1": int(df["horizon_dt_O2_O1"].sum()),
        "num_horizon_O3_O1": int(df["horizon_dt_O3_O1"].sum()),
        "num_horizon_O3_O2": int(df["horizon_dt_O3_O2"].sum()),
        "runtime_sec": runtime_sec,
    }
    if extra:
        summary.update(extra)
    return summary


# ============================================================
# PHASE 1: ORIGINAL NOISE TEST
# ============================================================

def run_noise_test_phase1():
    base_df = pd.read_csv(BASE_CSV)
    rows = []

    for nl in NOISE_LEVELS:
        print(f"[PHASE1:NOISE] Running noise level = {nl*100:.1f}%")
        df_noisy = inject_noise_sigma(base_df, nl)
        noisy_csv = f"cms_d0_noisy_{int(nl*100)}pct.csv"
        df_noisy.to_csv(noisy_csv, index=False)

        out_df, runtime = run_engine_on_csv(noisy_csv)
        summary = summarize_run(out_df, runtime, extra={"noise_level": nl})
        rows.append(summary)

    pd.DataFrame(rows).to_csv(NOISE_RESULTS_CSV, index=False)
    print(f"[PHASE1:NOISE] Saved {NOISE_RESULTS_CSV}")


# ============================================================
# PHASE 1: ORIGINAL METRIC SWEEP
# ============================================================

def run_metric_sweep_phase1():
    base_df = pd.read_csv(BASE_CSV)
    rows = []

    for (ax, bx) in METRIC_SWEEP_WEIGHTS:
        print(f"[PHASE1:SWEEP] Running O2 metric weights ax={ax}, bx={bx}")

        temp_csv = "cms_d0_metric_sweep_temp.csv"
        base_df.to_csv(temp_csv, index=False)

        engine = TimeEngineV2(processed_csv=temp_csv)
        engine.metrics["O2"] = make_metric_O2(ax, bx)

        t0 = time.time()
        engine.run_all()
        runtime = time.time() - t0
        out_df = engine.df

        summary = summarize_run(out_df, runtime, extra={"ax": ax, "bx": bx})
        rows.append(summary)

    pd.DataFrame(rows).to_csv(METRIC_SWEEP_RESULTS_CSV, index=False)
    print(f"[PHASE1:SWEEP] Saved {METRIC_SWEEP_RESULTS_CSV}")


# ============================================================
# PHASE 2: MONTE CARLO NOISE STORM
# ============================================================

def run_mc_noise_storm():
    base_df = pd.read_csv(BASE_CSV)
    rows = []

    for i in range(MC_NOISE_RUNS):
        nl = random.uniform(0.0, MC_NOISE_MAX)
        print(f"[PHASE2:MC_NOISE] Run {i+1}/{MC_NOISE_RUNS}, noise_level={nl:.4f}")

        df_noisy = inject_noise_sigma(base_df, nl)
        noisy_csv = f"cms_d0_mc_noisy_run_{i+1}.csv"
        df_noisy.to_csv(noisy_csv, index=False)

        out_df, runtime = run_engine_on_csv(noisy_csv)
        summary = summarize_run(out_df, runtime, extra={"noise_level": nl, "run_index": i+1})
        rows.append(summary)

    pd.DataFrame(rows).to_csv(MC_NOISE_RESULTS_CSV, index=False)
    print(f"[PHASE2:MC_NOISE] Saved {MC_NOISE_RESULTS_CSV}")


# ============================================================
# PHASE 2: RANDOM METRIC SWEEP
# ============================================================

def run_mc_metric_sweep():
    base_df = pd.read_csv(BASE_CSV)
    rows = []

    for i in range(MC_METRIC_RUNS):
        ax = random.uniform(*AX_RANGE)
        bx = random.uniform(*BX_RANGE)
        print(f"[PHASE2:MC_METRIC] Run {i+1}/{MC_METRIC_RUNS}, ax={ax:.3f}, bx={bx:.3f}")

        temp_csv = "cms_d0_mc_metric_temp.csv"
        base_df.to_csv(temp_csv, index=False)

        engine = TimeEngineV2(processed_csv=temp_csv)
        engine.metrics["O2"] = make_metric_O2(ax, bx)

        t0 = time.time()
        engine.run_all()
        runtime = time.time() - t0
        out_df = engine.df

        summary = summarize_run(out_df, runtime, extra={"ax": ax, "bx": bx, "run_index": i+1})
        rows.append(summary)

    pd.DataFrame(rows).to_csv(MC_METRIC_RESULTS_CSV, index=False)
    print(f"[PHASE2:MC_METRIC] Saved {MC_METRIC_RESULTS_CSV}")


# ============================================================
# PHASE 2: MULTI-DATASET MERGE
# ============================================================

def run_merged_dataset_test():
    print("[PHASE2:MERGE] Merging datasets:", DATASET_CSVS)

    dfs = [pd.read_csv(p) for p in DATASET_CSVS]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(MERGED_DATASET_CSV, index=False)

    print(f"[PHASE2:MERGE] Saved merged dataset to {MERGED_DATASET_CSV}")

    out_df, runtime = run_engine_on_csv(MERGED_DATASET_CSV)
    out_df.to_csv(MERGED_RUN_OUTPUT_CSV, index=False)

    print(f"[PHASE2:MERGE] Saved merged run output to {MERGED_RUN_OUTPUT_CSV}")
    print(f"[PHASE2:MERGE] Runtime: {runtime:.4f} s")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=== PHASE 1: Original Noise Test ===")
    run_noise_test_phase1()

    print("=== PHASE 1: Original Metric Sweep ===")
    run_metric_sweep_phase1()

    print("=== PHASE 2: Monte Carlo Noise Storm ===")
    run_mc_noise_storm()

    print("=== PHASE 2: Random Metric Sweep ===")
    run_mc_metric_sweep()

    print("=== PHASE 2: Multi-Dataset Merge Test ===")
    run_merged_dataset_test()

    print("=== DONE: Full Phase 2 Stress Suite Complete ===")
