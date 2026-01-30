# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Analyze substructure search benchmark results from CSV data.

Usage:
    python substruct_bench_analysis.py <csv_file> [--output-dir DIR] [--output-prefix PREFIX] [--sample-size N]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUMERIC_COLS = [
    "time_seconds",
    "num_matches",
    "target_num_atoms",
    "query_num_atoms",
    "query_recursion_count",
    "query_recursion_depth",
]

SCATTER_SAMPLE_SIZE = 50000


def load_numeric_data(csv_path: str) -> pd.DataFrame:
    """Load only the numeric columns from the CSV file."""
    print(f"Loading data from {csv_path}...")
    desired_cols = list(NUMERIC_COLS) + ["num_threads"]
    present_cols = list(pd.read_csv(csv_path, nrows=0).columns)
    usecols = [c for c in desired_cols if c in present_cols]

    dtypes = {
        "time_seconds": float,
        "num_matches": int,
        "target_num_atoms": int,
        "query_num_atoms": int,
        "query_recursion_count": int,
        "query_recursion_depth": int,
        "num_threads": int,
    }
    df = pd.read_csv(csv_path, usecols=usecols, dtype={k: v for k, v in dtypes.items() if k in usecols})
    print(f"Loaded {len(df):,} rows")
    return df


def subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Return a random subsample of the dataframe if larger than n."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def get_time_percentile_limits(df: pd.DataFrame, low: float = 0, high: float = 99) -> tuple:
    """Get percentile-based limits for time data in microseconds."""
    times_us = df["time_seconds"] * 1e6
    return times_us.quantile(low / 100), times_us.quantile(high / 100)


def plot_time_vs_target_size(df: pd.DataFrame, sample_size: int):
    """Scatter plot of search time vs target molecule size."""
    fig, ax = plt.subplots(figsize=(10, 7))
    df_sample = subsample(df, sample_size)
    scatter = ax.scatter(
        df_sample["target_num_atoms"],
        df_sample["time_seconds"] * 1e6,
        alpha=0.3,
        s=10,
        c=df_sample["query_num_atoms"],
        cmap="viridis",
    )
    ax.set_xlabel("Target Atoms")
    ax.set_ylabel("Time (µs)")
    _, y_max = get_time_percentile_limits(df, 0, 99)
    ax.set_ylim(0, y_max * 1.05)
    title = "Search Time vs Target Size (99th pctl)"
    if len(df) > sample_size:
        title += f" (n={sample_size:,})"
    ax.set_title(title)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Query Atoms")
    plt.tight_layout()
    return fig


def plot_time_vs_query_size(df: pd.DataFrame, sample_size: int):
    """Scatter plot of search time vs query pattern size."""
    fig, ax = plt.subplots(figsize=(10, 7))
    df_sample = subsample(df, sample_size)
    scatter = ax.scatter(
        df_sample["query_num_atoms"],
        df_sample["time_seconds"] * 1e6,
        alpha=0.3,
        s=10,
        c=df_sample["target_num_atoms"],
        cmap="plasma",
    )
    ax.set_xlabel("Query Atoms")
    ax.set_ylabel("Time (µs)")
    _, y_max = get_time_percentile_limits(df, 0, 99)
    ax.set_ylim(0, y_max * 1.05)
    title = "Search Time vs Query Size (99th pctl)"
    if len(df) > sample_size:
        title += f" (n={sample_size:,})"
    ax.set_title(title)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Target Atoms")
    plt.tight_layout()
    return fig


def plot_time_heatmap(df: pd.DataFrame, bin_size: int = 10):
    """2D heatmap of mean time by target/query size, binned."""
    fig, ax = plt.subplots(figsize=(10, 7))
    df_binned = df.copy()
    df_binned["target_bin"] = (df["target_num_atoms"] // bin_size) * bin_size
    df_binned["query_bin"] = (df["query_num_atoms"] // bin_size) * bin_size
    pivot = df_binned.pivot_table(
        values="time_seconds",
        index="query_bin",
        columns="target_bin",
        aggfunc="mean",
    )
    im = ax.imshow(
        pivot.values * 1e6,
        aspect="auto",
        origin="lower",
        cmap="hot",
        extent=[
            pivot.columns.min() - bin_size / 2,
            pivot.columns.max() + bin_size / 2,
            pivot.index.min() - bin_size / 2,
            pivot.index.max() + bin_size / 2,
        ],
    )
    ax.set_xlabel("Target Atoms")
    ax.set_ylabel("Query Atoms")
    ax.set_title(f"Mean Time (µs) by Size ({bin_size}-atom bins)")
    plt.colorbar(im, ax=ax, label="Mean Time (µs)")
    plt.tight_layout()
    return fig


def plot_time_by_matches(df: pd.DataFrame, sample_size: int):
    """Box plot of search time grouped by match count (subsampled per group)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    unique_matches = df["num_matches"].unique()
    per_group_sample = max(1000, sample_size // len(unique_matches))

    data = []
    positions = []
    for match_val in sorted(unique_matches):
        group = df[df["num_matches"] == match_val]["time_seconds"]
        if len(group) > per_group_sample:
            group = group.sample(n=per_group_sample, random_state=42)
        data.append(group.values * 1e6)
        positions.append(match_val)

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightcoral")
    ax.set_xlabel("Number of Matches")
    ax.set_ylabel("Time (µs)")
    ax.set_title("Search Time by Match Count (no outliers)")
    plt.tight_layout()
    return fig


def plot_time_vs_recursion(df: pd.DataFrame, sample_size: int):
    """Scatter plot of time vs recursion count, colored by depth."""
    fig, ax = plt.subplots(figsize=(10, 7))
    df_sample = subsample(df, sample_size)
    scatter = ax.scatter(
        df_sample["query_recursion_count"],
        df_sample["time_seconds"] * 1e6,
        alpha=0.4,
        s=15,
        c=df_sample["query_recursion_depth"],
        cmap="coolwarm",
    )
    ax.set_xlabel("Recursion Count")
    ax.set_ylabel("Time (µs)")
    _, y_max = get_time_percentile_limits(df, 0, 99)
    ax.set_ylim(0, y_max * 1.05)
    title = "Time vs Query Recursion (99th pctl)"
    if len(df) > sample_size:
        title += f" (n={sample_size:,})"
    ax.set_title(title)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Recursion Depth")
    plt.tight_layout()
    return fig


def plot_time_distribution(df: pd.DataFrame):
    """Histogram of search times."""
    fig, ax = plt.subplots(figsize=(10, 7))
    times_us = df["time_seconds"] * 1e6
    _, x_max = get_time_percentile_limits(df, 0, 99)
    times_clipped = times_us[times_us <= x_max]
    ax.hist(times_clipped, bins=100, color="teal", edgecolor="black", alpha=0.7)
    ax.axvline(times_us.median(), color="red", linestyle="--", label=f"Median: {times_us.median():.2f} µs")
    ax.axvline(times_us.mean(), color="orange", linestyle="--", label=f"Mean: {times_us.mean():.2f} µs")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Frequency")
    ax.set_title("Search Time Distribution (≤99th pctl)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_aggregated_by_target(df: pd.DataFrame):
    """Mean and std of time aggregated by target size."""
    fig, ax = plt.subplots(figsize=(10, 7))
    agg = df.groupby("target_num_atoms")["time_seconds"].agg(["mean", "std"])
    agg["mean"] *= 1e6
    agg["std"] *= 1e6
    ax.errorbar(
        agg.index,
        agg["mean"],
        yerr=agg["std"],
        fmt="o-",
        capsize=3,
        color="darkgreen",
        markersize=4,
    )
    ax.set_xlabel("Target Atoms")
    ax.set_ylabel("Time (µs)")
    ax.set_title("Mean Time by Target Size")
    plt.tight_layout()
    return fig


def plot_block_averages(df: pd.DataFrame):
    """Combined plot of mean time vs target, query, matches, and recursion."""
    print(f"  Unique recursion depths: {sorted(df['query_recursion_depth'].unique())}")
    print(f"  Unique recursion counts: {sorted(df['query_recursion_count'].unique())}")
    if "num_threads" in df.columns:
        print(f"  Unique num_threads: {sorted(df['num_threads'].unique())}")
        fig, axes = plt.subplots(1, 6, figsize=(29, 5))
    else:
        fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    agg_target = df.groupby("target_num_atoms")["time_seconds"].mean() * 1e6
    axes[0].plot(agg_target.index, agg_target.values, "o-", color="darkgreen", markersize=4)
    axes[0].set_xlabel("Target Atoms")
    axes[0].set_ylabel("Mean Time (µs)")
    axes[0].set_title("Time vs Target Size")

    agg_query = df.groupby("query_num_atoms")["time_seconds"].mean() * 1e6
    axes[1].plot(agg_query.index, agg_query.values, "o-", color="steelblue", markersize=4)
    axes[1].set_xlabel("Query Atoms")
    axes[1].set_title("Time vs Query Size")

    agg_matches = df.groupby("num_matches")["time_seconds"].mean() * 1e6
    axes[2].plot(agg_matches.index, agg_matches.values, "o-", color="coral", markersize=4)
    axes[2].set_xlabel("Number of Matches")
    axes[2].set_title("Time vs Match Count")

    agg_recursion = df.groupby("query_recursion_count")["time_seconds"].mean() * 1e6
    axes[3].plot(agg_recursion.index, agg_recursion.values, "o-", color="purple", markersize=4)
    axes[3].set_xlabel("Recursion Count")
    axes[3].set_title("Time vs Recursion Count")

    agg_depth = df.groupby("query_recursion_depth")["time_seconds"].mean() * 1e6
    axes[4].plot(agg_depth.index, agg_depth.values, "o-", color="goldenrod", markersize=4)
    axes[4].set_xlabel("Recursion Depth")
    axes[4].set_title("Time vs Recursion Depth")

    if "num_threads" in df.columns:
        agg_threads = df.groupby("num_threads")["time_seconds"].mean() * 1e6
        axes[5].plot(agg_threads.index, agg_threads.values, "o-", color="slategray", markersize=4)
        axes[5].set_xlabel("num_threads")
        axes[5].set_title("Time vs num_threads")

    fig.suptitle("Mean Search Time by Variable", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics to stdout."""
    print("\n" + "=" * 60)
    print("SUBSTRUCTURE SEARCH BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total searches: {len(df):,}")
    print(f"\nTime statistics (µs):")
    times_us = df["time_seconds"] * 1e6
    print(f"  Min:    {times_us.min():.3f}")
    print(f"  Max:    {times_us.max():.3f}")
    print(f"  Mean:   {times_us.mean():.3f}")
    print(f"  Median: {times_us.median():.3f}")
    print(f"  Std:    {times_us.std():.3f}")
    print(f"\nTarget molecule atoms:")
    print(f"  Range: {df['target_num_atoms'].min()} - {df['target_num_atoms'].max()}")
    print(f"  Mean:  {df['target_num_atoms'].mean():.1f}")
    print(f"\nQuery pattern atoms:")
    print(f"  Range: {df['query_num_atoms'].min()} - {df['query_num_atoms'].max()}")
    print(f"  Mean:  {df['query_num_atoms'].mean():.1f}")
    print(f"\nMatch statistics:")
    print(f"  Searches with matches: {(df['num_matches'] > 0).sum():,} ({100 * (df['num_matches'] > 0).mean():.1f}%)")
    print(f"  Max matches: {df['num_matches'].max()}")
    if (df['num_matches'] > 0).any():
        print(f"  Mean matches (when > 0): {df[df['num_matches'] > 0]['num_matches'].mean():.2f}")
    print(f"\nRecursion statistics:")
    print(f"  Recursion count range: {df['query_recursion_count'].min()} - {df['query_recursion_count'].max()}")
    print(f"  Recursion depth range: {df['query_recursion_depth'].min()} - {df['query_recursion_depth'].max()}")
    if "num_threads" in df.columns:
        print(f"\nnum_threads:")
        print(f"  Unique: {sorted(df['num_threads'].unique())}")
    print("=" * 60)


def save_and_show(fig, output_dir: str, prefix: str, name: str):
    """Save figure and display it."""
    path = os.path.join(output_dir, f"{prefix}_{name}.png")
    print(f"  Saving to {path}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze substructure search benchmark data")
    parser.add_argument("csv_file", help="Path to the benchmark CSV file")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output plot files (default: current directory)",
    )
    parser.add_argument(
        "--output-prefix",
        default="substruct_bench",
        help="Prefix for output plot files (default: substruct_bench)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SCATTER_SAMPLE_SIZE,
        help=f"Number of points to sample for scatter plots (default: {SCATTER_SAMPLE_SIZE})",
    )
    args = parser.parse_args()

    df = load_numeric_data(args.csv_file)

    print("\nComputing summary statistics...")
    print_summary_stats(df)

    print("\nGenerating plots...")

    print("Plotting block averages...")
    fig = plot_block_averages(df)
    save_and_show(fig, args.output_dir, args.output_prefix, "block_averages")

    # print("Plotting time vs target size...")
    # fig = plot_time_vs_target_size(df, args.sample_size)
    # save_and_show(fig, args.output_dir, args.output_prefix, "time_vs_target")

    # print("Plotting time vs query size...")
    # fig = plot_time_vs_query_size(df, args.sample_size)
    # save_and_show(fig, args.output_dir, args.output_prefix, "time_vs_query")

    print("Plotting time heatmap...")
    fig = plot_time_heatmap(df)
    save_and_show(fig, args.output_dir, args.output_prefix, "time_heatmap")

    print("Plotting time by matches...")
    fig = plot_time_by_matches(df, args.sample_size)
    save_and_show(fig, args.output_dir, args.output_prefix, "time_by_matches")

    # print("Plotting time vs recursion...")
    # fig = plot_time_vs_recursion(df, args.sample_size)
    # save_and_show(fig, args.output_dir, args.output_prefix, "time_vs_recursion")

    print("Plotting time distribution...")
    fig = plot_time_distribution(df)
    save_and_show(fig, args.output_dir, args.output_prefix, "time_distribution")

    print("Plotting mean time by target size...")
    fig = plot_aggregated_by_target(df)
    save_and_show(fig, args.output_dir, args.output_prefix, "time_by_target")

    print("\nDone.")


if __name__ == "__main__":
    main()
