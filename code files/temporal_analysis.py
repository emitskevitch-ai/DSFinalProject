# =============================================================================
# temporal_analysis.py
# =============================================================================
# PURPOSE:
#   This script analyzes HOW water quality changes over time relative to
#   wildfire events. It answers questions like:
#     - Did water quality get worse after a fire?
#     - How quickly did it recover?
#     - Are the changes statistically significant?
#     - Does the effect differ for large fires vs small ones?
#
#   All analyses in this file share the same core approach:
#     1. Load the pre-built fire/station join from pipeline.py
#     2. Calculate "days since fire" for each reading
#     3. Compare readings taken before the fire vs after it
#
# PREREQUISITE: Run pipeline.py first to generate the required CSVs.
#
# INPUTS (from csvs/):
#   - all_stations_fire_joined.csv       — All stations with fire info (from pipeline.py)
#
# OUTPUTS:
#   Graphs (to graphs/):
#     - four_scenario_comparison.png   — Grouped bar chart: mean change per analyte
#                                        across 4 scenarios (1yr/5yr × all/large fires)
#
#   CSVs (to csvs/):
#     - fire_paired_results.csv        — Per-fire mean before/after values
#     - large_fires_5yr_results.csv    — 5-year results for large fires only
#     - impacted_within_1month.csv     — Readings in the first month post-fire
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------------------------------------------------------
# PATH SETUP
# This script lives in "code files/" — navigate up one level to reach the root.
# -----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_CSVS = os.path.join(_ROOT, "csvs")
_GRAPHS = os.path.join(_ROOT, "graphs")

# -----------------------------------------------------------------------------
# ANALYTES TO ANALYZE
# These are the 6 water quality measures we focus on for temporal analysis.
# The full analyte names match column headers in the master CSV exactly.
# -----------------------------------------------------------------------------
ANALYTES = [
    'Oxygen, Dissolved, Total', # Dissolved oxygen — drops when water quality degrades
    'Turbidity, Total', # Cloudiness/suspended particles — increases after fires
    'Nitrogen, Total, Total', # Nutrients — can spike from ash and soil runoff
    'pH', # Acidity — fires can affect soil chemistry and runoff
    'Phosphorus as P, Total', # Nutrients — elevated by ash and erosion
    'Total Organic Carbon, Total', # Organic matter — increases from burned vegetation runoff
]

# Fires larger than this (in acres) are considered "large fires" in scenario analysis
LARGE_FIRE_THRESHOLD = 50000


# =============================================================================
# DATA LOADING
# =============================================================================
# pipeline.py already did the spatial join and wrote all_stations_fire_joined.csv.
# Rows with no fire match have NaN in FIRE_NAME — drop those to get only stations
# that fell inside a fire perimeter.
# =============================================================================
print("Loading data...")

joined = pd.read_csv(os.path.join(_CSVS, "all_stations_fire_joined.csv"))
joined = joined.dropna(subset=['FIRE_NAME'])
joined['SampleDate'] = pd.to_datetime(joined['SampleDate'])
joined['ALARM_DATE'] = pd.to_datetime(joined['ALARM_DATE'], utc=True).dt.tz_localize(None)
joined['days_since_fire'] = (joined['SampleDate'] - joined['ALARM_DATE']).dt.days

print(f"  Readings inside any fire perimeter: {len(joined)}")
print(f"  Unique fires with water stations: {joined['FIRE_NAME'].nunique()}")
print(f"  Unique water stations affected: {joined['StationCode'].nunique()}")


# =============================================================================
# SECTION 1: PAIRED BEFORE/AFTER ANALYSIS — PER FIRE PERIMETER
# =============================================================================
# For each individual fire, compare the mean analyte values measured at stations
# BEFORE the fire started vs AFTER (within 1 year). Only fires that have readings
# in BOTH time windows are included (we need a "before" and "after" to compare).
#
# This is the most direct way to see if a specific fire degraded water quality —
# the same stations, the same fire, just different time windows.
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: Paired before/after analysis per fire")
print("=" * 70)

fire_results = []

for fire_name, fire_group in joined.groupby('FIRE_NAME'):
    # Split this fire's readings into before and after (within 1 year)
    fire_before = fire_group[fire_group['days_since_fire'] < 0]
    fire_after = fire_group[
        (fire_group['days_since_fire'] >= 0) &
        (fire_group['days_since_fire'] <= 365)
    ]

    # Skip fires where we only have readings on one side — we need both to compare
    if len(fire_before) == 0 or len(fire_after) == 0:
        continue

    # Build a summary row for this fire
    row = {
        'Fire': fire_name,
        'GIS_Acres': fire_group['GIS_ACRES'].iloc[0],
        'n_before': len(fire_before),
        'n_after': len(fire_after),
    }

    # For each analyte, compute mean before and after, and the difference
    for analyte in ANALYTES:
        before_vals = fire_before[analyte].dropna()
        after_vals = fire_after[analyte].dropna()
        if len(before_vals) > 0 and len(after_vals) > 0:
            short = analyte.split(',')[0]
            row[f'{short}_before'] = before_vals.mean()
            row[f'{short}_after'] = after_vals.mean()
            row[f'{short}_change'] = after_vals.mean() - before_vals.mean() # positive = got worse / higher

    fire_results.append(row)

fire_df = pd.DataFrame(fire_results)
print(f"  Fires with both before AND after readings: {len(fire_df)}")
print(f"\n{fire_df[['Fire', 'GIS_Acres', 'n_before', 'n_after']].to_string(index=False)}")

# --- Average change across all fires ---
print("\n" + "=" * 70)
print("Average change across all fires (Before → After 1yr)")
print("=" * 70)
for col in [c for c in fire_df.columns if c.endswith('_change')]:
    analyte_name = col.replace('_change', '')
    vals = fire_df[col].dropna()
    if len(vals) == 0:
        continue
    mean_change = vals.mean()
    direction = '↑ increased' if mean_change > 0 else '↓ decreased'
    print(f"  {analyte_name}: {mean_change:+.4f} {direction} (n={len(vals)} fires)")

# --- Paired t-test for statistical significance ---
# A paired t-test compares the before and after means for each fire together,
# treating each fire as a matched pair. p < 0.05 means the difference is
# unlikely to be due to chance alone.
print("\n" + "=" * 70)
print("Statistical significance — Paired t-tests (p < 0.05 = significant)")
print("=" * 70)
for analyte in ANALYTES:
    short = analyte.split(',')[0]
    before_col = f'{short}_before'
    after_col = f'{short}_after'
    if before_col not in fire_df.columns or after_col not in fire_df.columns:
        continue
    pairs = fire_df[[before_col, after_col]].dropna()
    if len(pairs) < 2:
        print(f"  {short}: not enough paired fires (n={len(pairs)})")
        continue
    _, p = stats.ttest_rel(pairs[before_col], pairs[after_col])
    sig = 'significant' if p < 0.05 else 'not significant'
    print(f"  {short}: p={p:.4f} {sig} (n={len(pairs)} fires)")

# Save the per-fire summary table
fire_df.to_csv(os.path.join(_CSVS, "fire_paired_results.csv"), index=False)
print("  CSV saved: fire_paired_results.csv")


# =============================================================================
# SECTION 2: FOUR-SCENARIO COMPARISON — GROUPED BAR CHART
# =============================================================================
# The primary visualization of this script. Compares mean analyte change
# across four combinations of time window and fire size:
#   - 1 year after fire  vs  5 years after fire
#   - All fires          vs  large fires only (>50,000 acres)
#
# Each analyte gets its own subplot with 4 bars — one per scenario.
# Blue = analyte increased after fire, red = decreased.
# Each analyte has its own y-axis scale so turbidity's large values
# don't compress the smaller changes in pH, oxygen, etc.
#
# Answers: do effects persist beyond 1 year? Are large fires worse?
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: Four-scenario comparison (1yr/5yr × all/large fires)")
print("=" * 70)

def build_fire_results(joined_data, after_days, min_acres=0):
    """
    For each fire with readings both before AND after, computes mean analyte
    values in both windows and the difference (after - before).

    Parameters:
        joined_data (DataFrame): Spatially joined water/fire data with days_since_fire.
        after_days (int): How many days after the fire to include in the "after" window.
        min_acres (float): Minimum fire size in acres. Fires smaller than this are excluded.

    Returns:
        DataFrame with one row per fire, columns for before/after/change per analyte.
    """
    # Filter to only fires meeting the minimum size threshold
    subset = joined_data[joined_data['GIS_ACRES'] >= min_acres]

    # Split into before-fire and after-fire windows globally
    before = subset[subset['days_since_fire'] < 0]
    after = subset[
        (subset['days_since_fire'] >= 0) &
        (subset['days_since_fire'] <= after_days)
    ]

    results = []
    for fire_name, fire_group in subset.groupby('FIRE_NAME'):
        fire_before = before[before['FIRE_NAME'] == fire_name]
        fire_after = after[after['FIRE_NAME'] == fire_name]

        # Only include fires where we have readings on both sides of the fire date
        if len(fire_before) == 0 or len(fire_after) == 0:
            continue

        row = {
            'Fire': fire_name,
            'GIS_Acres': fire_group['GIS_ACRES'].iloc[0],
            'n_before': len(fire_before),
            'n_after': len(fire_after),
        }
        for analyte in ANALYTES:
            short = analyte.split(',')[0]
            b = fire_before[analyte].dropna()
            a = fire_after[analyte].dropna()
            if len(b) > 0 and len(a) > 0:
                row[f'{short}_before'] = b.mean()
                row[f'{short}_after'] = a.mean()
                row[f'{short}_change'] = a.mean() - b.mean()
        results.append(row)

    return pd.DataFrame(results)


def print_scenario_summary(fire_df, label):
    """
    Prints a summary of mean changes and paired t-test p-values for a set of fires.

    Parameters:
        fire_df (DataFrame): Output from build_fire_results().
        label (str): A descriptive label for this scenario, printed in the header.

    Returns:
        DataFrame summarizing analyte, mean change, p-value, and n fires.
    """
    print(f"\n{'=' * 70}")
    print(f"{label}  (n={len(fire_df)} fires)")
    print(f"{'=' * 70}")

    summary = []
    for analyte in ANALYTES:
        short = analyte.split(',')[0]
        change_col = f'{short}_change'
        before_col = f'{short}_before'
        after_col = f'{short}_after'

        if change_col not in fire_df.columns:
            continue

        vals = fire_df[change_col].dropna()
        pairs = fire_df[[before_col, after_col]].dropna()
        if len(vals) == 0:
            continue

        mean_change = vals.mean()
        direction = '↑' if mean_change > 0 else '↓'

        if len(pairs) >= 2:
            _, p = stats.ttest_rel(pairs[before_col], pairs[after_col])
            sig = 'significant' if p < 0.05 else 'not significant'
            print(f"  {short}: {mean_change:+.4f} {direction} p={p:.4f} {sig} (n={len(pairs)} fires)")
            summary.append({'Analyte': short, 'Mean Change': mean_change, 'p-value': p, 'n_fires': len(pairs)})
        else:
            print(f"  {short}: {mean_change:+.4f} {direction} (not enough fires)")

    return pd.DataFrame(summary)


# Run all four scenarios
df_1yr_all = build_fire_results(joined, after_days=365, min_acres=0)
df_5yr_all = build_fire_results(joined, after_days=365 * 5, min_acres=0)
df_1yr_large = build_fire_results(joined, after_days=365, min_acres=LARGE_FIRE_THRESHOLD)
df_5yr_large = build_fire_results(joined, after_days=365 * 5, min_acres=LARGE_FIRE_THRESHOLD)

print_scenario_summary(df_1yr_all, "1 YEAR AFTER — All Fire Sizes")
print_scenario_summary(df_5yr_all, "5 YEARS AFTER — All Fire Sizes")
print_scenario_summary(df_1yr_large, f"1 YEAR AFTER — Large Fires Only (>{LARGE_FIRE_THRESHOLD:,} acres)")
print_scenario_summary(df_5yr_large, f"5 YEARS AFTER — Large Fires Only (>{LARGE_FIRE_THRESHOLD:,} acres)")

# --- Visualization: grouped bar chart ---
# One subplot per analyte (each has its own y-axis scale so turbidity doesn't
# crush the smaller analytes). Four bars per subplot — one per scenario.
# Bar color shows direction: steelblue = increased, tomato = decreased.
scenarios = [
    (df_1yr_all, "1yr\nAll Fires"),
    (df_5yr_all, "5yr\nAll Fires"),
    (df_1yr_large, "1yr\nLarge Fires"),
    (df_5yr_large, "5yr\nLarge Fires"),
]
plot_analytes = ['Oxygen', 'Turbidity', 'Nitrogen', 'pH', 'Phosphorus as P']

fig, axes = plt.subplots(1, len(plot_analytes), figsize=(16, 5))
fig.suptitle('Mean Change in Water Quality After Wildfire by Scenario\n(After − Before)', fontsize=13)

x = np.arange(len(scenarios))

for i, short in enumerate(plot_analytes):
    ax = axes[i]
    means = []
    for df, _ in scenarios:
        change_col = f'{short}_change'
        if change_col in df.columns and not df[change_col].dropna().empty:
            means.append(df[change_col].dropna().mean())
        else:
            means.append(0)

    colors = ['steelblue' if v >= 0 else 'tomato' for v in means]
    ax.bar(x, means, color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(short, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in scenarios], fontsize=8)
    if i == 0:
        ax.set_ylabel('Mean Change (After − Before)')

    # Label each bar with its value
    for j, v in enumerate(means):
        ax.text(j, v + (max(means) - min(means)) * 0.03 * (1 if v >= 0 else -1),
                f'{v:+.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(_GRAPHS, "four_scenario_comparison.png"), dpi=150)
plt.show()
print("  Graph saved: four_scenario_comparison.png")

# Save the 5-year large fires results (most conservative, most interesting scenario)
df_5yr_large.to_csv(os.path.join(_CSVS, "large_fires_5yr_results.csv"), index=False)
print("  CSV saved: large_fires_5yr_results.csv")


# =============================================================================
# SECTION 3: ONE-MONTH IMPACT FILTER
# =============================================================================
# Produces a focused dataset of only water readings that were:
#   1. Taken at a station physically inside a fire perimeter, AND
#   2. Taken within 30 days of the fire's start date
#
# This is the strictest definition of "immediately fire-impacted" data —
# useful as a high-confidence subset for any further analysis.
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: One-month post-fire impact filter")
print("=" * 70)

# Filter to readings taken 0–30 days after the fire start using days_since_fire,
# which was already computed from the pipeline output
impacted_1month = joined[
    (joined['days_since_fire'] >= 0) &
    (joined['days_since_fire'] <= 30)
]

print(f"  Readings inside a fire perimeter: {len(joined)}")
print(f"  Readings within 1 month of fire: {len(impacted_1month)}")
print(f"\n  Fires represented: {impacted_1month['FIRE_NAME'].value_counts().to_string()}")

if len(impacted_1month) > 0:
    print(f"\n  Sample date range: "
            f"{impacted_1month['SampleDate'].min()} to {impacted_1month['SampleDate'].max()}")
    print("\n  Preview:")
    print(impacted_1month[
        ['StationName', 'SampleDate', 'FIRE_NAME', 'GIS_ACRES', 'Oxygen, Dissolved, Total']
    ].head(10))

impacted_1month.to_csv(os.path.join(_CSVS, "impacted_within_1month.csv"), index=False)
print("  CSV saved: impacted_within_1month.csv")

print("\n" + "=" * 70)
print("temporal_analysis.py complete.")
print("=" * 70)
