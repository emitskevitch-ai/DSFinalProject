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
#     1. Spatially join water stations to fire perimeters (stations inside fires)
#     2. Calculate "days since fire" for each reading
#     3. Compare readings taken before the fire vs after it
#
# PREREQUISITE: Run pipeline.py first to generate the required CSVs.
#
# INPUTS (from csvs/):
#   - water_quality_allinfo_master.csv   — All water quality readings
#   - fire24_1.gdb                       — Fire perimeter polygons
#
# OUTPUTS:
#   Graphs (to graphs/):
#     - paired_before_after.png        — Per-fire before/after line plot
#     - four_scenario_comparison.png   — Bar charts across 4 time/size scenarios
#     - short_vs_long_term.png         — Boxplots: baseline vs 30d vs 365d
#
#   CSVs (to csvs/):
#     - fire_paired_results.csv        — Per-fire mean before/after values
#     - large_fires_5yr_results.csv    — 5-year results for large fires only
#     - impacted_short_term.csv        — Readings 0-30 days post-fire
#     - impacted_long_term.csv         — Readings 0-365 days post-fire
#     - impacted_within_1month.csv     — Readings in the first month post-fire
# =============================================================================

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -----------------------------------------------------------------------------
# PATH SETUP
# This script lives in "code files/" — navigate up one level to reach the root.
# -----------------------------------------------------------------------------
_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.join(_HERE, "..")
_CSVS  = os.path.join(_ROOT, "csvs")
_GRAPHS = os.path.join(_ROOT, "graphs")

# -----------------------------------------------------------------------------
# ANALYTES TO ANALYZE
# These are the 6 water quality measures we focus on for temporal analysis.
# The full analyte names match column headers in the master CSV exactly.
# -----------------------------------------------------------------------------
ANALYTES = [
    'Oxygen, Dissolved, Total',    # Dissolved oxygen — drops when water quality degrades
    'Turbidity, Total',            # Cloudiness/suspended particles — increases after fires
    'Nitrogen, Total, Total',      # Nutrients — can spike from ash and soil runoff
    'pH',                          # Acidity — fires can affect soil chemistry and runoff
    'Phosphorus as P, Total',      # Nutrients — elevated by ash and erosion
    'Total Organic Carbon, Total', # Organic matter — increases from burned vegetation runoff
]

# Fires larger than this (in acres) are considered "large fires" in scenario analysis
LARGE_FIRE_THRESHOLD = 50000


# =============================================================================
# DATA LOADING
# =============================================================================
print("Loading water quality data and fire perimeters...")

# Load the master water quality table produced by pipeline.py
water = pd.read_csv(os.path.join(_CSVS, "water_quality_allinfo_master.csv"))

# Parse the sample date column from string to a proper datetime object
# so we can do date arithmetic (e.g. calculating days between fire and sample)
water['SampleDate'] = pd.to_datetime(water['SampleDate'])

# Load the fire perimeter geodatabase
# EPSG:4326 is standard WGS84 lat/lon — we need both datasets in the same
# coordinate system before we can do spatial operations
gdb_path = os.path.join(_ROOT, "fire24_1.gdb")
fires = gpd.read_file(gdb_path, engine="pyogrio", layer="firep24_1")
fires = fires.to_crs("EPSG:4326")

# Parse fire alarm dates, stripping timezone info so both date columns
# are timezone-naive (otherwise subtraction for days_since_fire fails)
fires['ALARM_DATE'] = pd.to_datetime(fires['ALARM_DATE'], utc=True).dt.tz_localize(None)
fires['CONT_DATE']  = pd.to_datetime(fires['CONT_DATE'],  utc=True).dt.tz_localize(None)

# Convert water station lat/lon to point geometries so GeoPandas can
# spatially join them against the fire perimeter polygons
water_gdf = gpd.GeoDataFrame(
    water,
    geometry=gpd.points_from_xy(water['TargetLongitude'], water['TargetLatitude']),
    crs="EPSG:4326"
)


# =============================================================================
# SHARED SPATIAL JOIN
# =============================================================================
# All temporal analyses need to know which water stations are inside fire
# perimeters and how many days before/after the fire each reading was taken.
# We do this join once here and reuse the result in all sections below.
#
# how='inner' — only keep water stations that fall inside at least one fire
# predicate='within' — station point must be inside the fire polygon
# =============================================================================
print("Running spatial join (water stations inside fire perimeters)...")
joined = gpd.sjoin(
    water_gdf,
    fires[['FIRE_NAME', 'ALARM_DATE', 'CONT_DATE', 'GIS_ACRES', 'geometry']],
    how='inner',
    predicate='within'
)

# Re-parse SampleDate after the join (GeoPandas can lose dtype info during joins)
joined['SampleDate'] = pd.to_datetime(joined['SampleDate'])

# Calculate how many days before or after the fire's start date each reading was taken.
# Negative values = reading was taken before the fire started.
# Positive values = reading was taken after the fire started.
joined['days_since_fire'] = (joined['SampleDate'] - joined['ALARM_DATE']).dt.days

print(f"  Readings inside any fire perimeter: {len(joined)}")
print(f"  Unique fires with water stations:   {joined['FIRE_NAME'].nunique()}")
print(f"  Unique water stations affected:     {joined['StationCode'].nunique()}")


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
    fire_after  = fire_group[
        (fire_group['days_since_fire'] >= 0) &
        (fire_group['days_since_fire'] <= 365)
    ]

    # Skip fires where we only have readings on one side — we need both to compare
    if len(fire_before) == 0 or len(fire_after) == 0:
        continue

    # Build a summary row for this fire
    row = {
        'Fire':      fire_name,
        'GIS_Acres': fire_group['GIS_ACRES'].iloc[0],
        'n_before':  len(fire_before),
        'n_after':   len(fire_after),
    }

    # For each analyte, compute mean before and after, and the difference
    for analyte in ANALYTES:
        before_vals = fire_before[analyte].dropna()
        after_vals  = fire_after[analyte].dropna()
        if len(before_vals) > 0 and len(after_vals) > 0:
            short = analyte.split(',')[0]
            row[f'{short}_before'] = before_vals.mean()
            row[f'{short}_after']  = after_vals.mean()
            row[f'{short}_change'] = after_vals.mean() - before_vals.mean()  # positive = got worse / higher

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
    print(f"  {analyte_name:30s}: {mean_change:+.4f}  {direction}  (n={len(vals)} fires)")

# --- Paired t-test for statistical significance ---
# A paired t-test compares the before and after means for each fire together,
# treating each fire as a matched pair. p < 0.05 means the difference is
# unlikely to be due to chance alone.
print("\n" + "=" * 70)
print("Statistical significance — Paired t-tests (p < 0.05 = significant)")
print("=" * 70)
for analyte in ANALYTES:
    short      = analyte.split(',')[0]
    before_col = f'{short}_before'
    after_col  = f'{short}_after'
    if before_col not in fire_df.columns or after_col not in fire_df.columns:
        continue
    pairs = fire_df[[before_col, after_col]].dropna()
    if len(pairs) < 2:
        print(f"  {short:30s}: not enough paired fires (n={len(pairs)})")
        continue
    t, p = stats.ttest_rel(pairs[before_col], pairs[after_col])
    sig = '✅ significant' if p < 0.05 else '❌ not significant'
    print(f"  {short:30s}: p={p:.4f}  {sig}  (n={len(pairs)} fires)")

# --- Visualization: line plot per fire, before → after ---
# Each thin line is one fire. The red line is the mean across all fires.
# A downward trend means most fires caused that analyte to decrease.
change_analytes = [
    a.split(',')[0] for a in ANALYTES
    if f"{a.split(',')[0]}_change" in fire_df.columns
]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Before vs After Wildfire — Per Fire Perimeter', fontsize=13)
axes = axes.flatten()
for i, short in enumerate(change_analytes[:6]):
    ax = axes[i]
    before_col = f'{short}_before'
    after_col  = f'{short}_after'
    if before_col not in fire_df.columns:
        continue
    pairs = fire_df[[before_col, after_col, 'Fire']].dropna()
    # Draw one thin line per fire
    for _, row in pairs.iterrows():
        ax.plot([0, 1], [row[before_col], row[after_col]], 'o-', alpha=0.5, color='steelblue')
    # Draw the mean line in red so it stands out
    ax.plot([0, 1], [pairs[before_col].mean(), pairs[after_col].mean()],
            'o-', color='red', linewidth=3, label='Mean')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before', 'After'])
    ax.set_title(short)
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(_GRAPHS, "paired_before_after.png"), dpi=150)
plt.show()
print("  Graph saved: paired_before_after.png")

# Save the per-fire summary table
fire_df.to_csv(os.path.join(_CSVS, "fire_paired_results.csv"), index=False)
print("  CSV saved: fire_paired_results.csv")


# =============================================================================
# SECTION 2: FOUR-SCENARIO COMPARISON
# =============================================================================
# Extends the before/after analysis across four combinations of:
#   - Time window:  1 year after fire  vs  5 years after fire
#   - Fire size:    all fires           vs  large fires only (>50,000 acres)
#
# This helps answer: do effects last longer than 1 year? Are large fires worse?
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
    after  = subset[
        (subset['days_since_fire'] >= 0) &
        (subset['days_since_fire'] <= after_days)
    ]

    results = []
    for fire_name, fire_group in subset.groupby('FIRE_NAME'):
        fire_before = before[before['FIRE_NAME'] == fire_name]
        fire_after  = after[after['FIRE_NAME'] == fire_name]

        # Only include fires where we have readings on both sides of the fire date
        if len(fire_before) == 0 or len(fire_after) == 0:
            continue

        row = {
            'Fire':      fire_name,
            'GIS_Acres': fire_group['GIS_ACRES'].iloc[0],
            'n_before':  len(fire_before),
            'n_after':   len(fire_after),
        }
        for analyte in ANALYTES:
            short = analyte.split(',')[0]
            b = fire_before[analyte].dropna()
            a = fire_after[analyte].dropna()
            if len(b) > 0 and len(a) > 0:
                row[f'{short}_before'] = b.mean()
                row[f'{short}_after']  = a.mean()
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
        short      = analyte.split(',')[0]
        change_col = f'{short}_change'
        before_col = f'{short}_before'
        after_col  = f'{short}_after'

        if change_col not in fire_df.columns:
            continue

        vals  = fire_df[change_col].dropna()
        pairs = fire_df[[before_col, after_col]].dropna()
        if len(vals) == 0:
            continue

        mean_change = vals.mean()
        direction   = '↑' if mean_change > 0 else '↓'

        if len(pairs) >= 2:
            _, p = stats.ttest_rel(pairs[before_col], pairs[after_col])
            sig  = '✅' if p < 0.05 else '❌'
            print(f"  {short:30s}: {mean_change:+.4f} {direction}  p={p:.4f} {sig}  (n={len(pairs)} fires)")
            summary.append({'Analyte': short, 'Mean Change': mean_change, 'p-value': p, 'n_fires': len(pairs)})
        else:
            print(f"  {short:30s}: {mean_change:+.4f} {direction}  (not enough fires)")

    return pd.DataFrame(summary)


# Run all four scenarios
df_1yr_all   = build_fire_results(joined, after_days=365,       min_acres=0)
df_5yr_all   = build_fire_results(joined, after_days=365 * 5,   min_acres=0)
df_1yr_large = build_fire_results(joined, after_days=365,       min_acres=LARGE_FIRE_THRESHOLD)
df_5yr_large = build_fire_results(joined, after_days=365 * 5,   min_acres=LARGE_FIRE_THRESHOLD)

print_scenario_summary(df_1yr_all,   "1 YEAR AFTER — All Fire Sizes")
print_scenario_summary(df_5yr_all,   "5 YEARS AFTER — All Fire Sizes")
print_scenario_summary(df_1yr_large, f"1 YEAR AFTER — Large Fires Only (>{LARGE_FIRE_THRESHOLD:,} acres)")
print_scenario_summary(df_5yr_large, f"5 YEARS AFTER — Large Fires Only (>{LARGE_FIRE_THRESHOLD:,} acres)")

# --- Visualization: 4-column bar chart grid ---
# Each column is one scenario, each row is one analyte.
# Blue bars = positive change (analyte went up), red bars = negative (went down).
# Orange dashed line = mean change across all fires in that scenario.
scenarios = [
    (df_1yr_all,   "1yr — All Fires"),
    (df_5yr_all,   "5yr — All Fires"),
    (df_1yr_large, "1yr — Large Fires"),
    (df_5yr_large, "5yr — Large Fires"),
]
plot_analytes = ['Oxygen', 'Turbidity', 'Nitrogen', 'pH', 'Phosphorus as P']

fig, axes = plt.subplots(len(plot_analytes), 4, figsize=(18, 16))
fig.suptitle('Mean Change in Water Quality After Wildfire\n(After − Before)', fontsize=14)

for row_i, short in enumerate(plot_analytes):
    for col_i, (df, label) in enumerate(scenarios):
        ax = axes[row_i, col_i]
        change_col = f'{short}_change'

        if change_col not in df.columns or df[change_col].dropna().empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(label if row_i == 0 else '')
            continue

        vals   = df[change_col].dropna()
        colors = ['red' if v < 0 else 'steelblue' for v in vals]
        ax.bar(range(len(vals)), vals.values, color=colors, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axhline(vals.mean(), color='orange', linewidth=2, linestyle='--',
                   label=f'Mean: {vals.mean():+.2f}')
        ax.legend(fontsize=7)

        if row_i == 0:
            ax.set_title(label, fontsize=10)
        if col_i == 0:
            ax.set_ylabel(short, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(_GRAPHS, "four_scenario_comparison.png"), dpi=150)
plt.show()
print("  Graph saved: four_scenario_comparison.png")

# Save the 5-year large fires results (most conservative, most interesting scenario)
df_5yr_large.to_csv(os.path.join(_CSVS, "large_fires_5yr_results.csv"), index=False)
print("  CSV saved: large_fires_5yr_results.csv")


# =============================================================================
# SECTION 3: BASELINE vs SHORT-TERM vs LONG-TERM
# =============================================================================
# Compares three groups to understand the trajectory of water quality recovery:
#
#   Baseline  — All readings in the master dataset (represents "normal" conditions)
#   Short-term — Readings taken 0–30 days after a fire (immediate impact)
#   Long-term  — Readings taken 0–365 days after a fire (sustained impact)
#
# Unlike Section 1, this doesn't pair readings to specific fires — it just
# compares population-level means across the three groups. The baseline includes
# all stations, not just ones inside fire perimeters, which makes it a better
# representation of typical water quality statewide.
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: Baseline vs short-term (0-30d) vs long-term (0-365d)")
print("=" * 70)

# Subset of joined data to only post-fire readings
after_fire = joined[joined['days_since_fire'] >= 0]

# Short-term: first 30 days after fire ignition
short_term = after_fire[after_fire['days_since_fire'] <= 30]

# Long-term: full first year after fire ignition
long_term  = after_fire[after_fire['days_since_fire'] <= 365]

# Baseline: the full unfiltered master dataset (all readings, all stations)
# This is read fresh from the CSV rather than using the spatially joined data,
# so it includes stations that were never near a fire — a true population baseline.
baseline = pd.read_csv(os.path.join(_CSVS, "water_quality_allinfo_master.csv"))

print(f"  Baseline samples:         {len(baseline)}")
print(f"  Short-term (0-30 days):   {len(short_term)}")
print(f"  Long-term  (0-365 days):  {len(long_term)}")

# --- Mean comparison table ---
print("\n" + "=" * 70)
print("Mean comparison — Baseline vs Short-term vs Long-term")
print("=" * 70)
rows = []
for analyte in ANALYTES:
    base_mean  = baseline[analyte].dropna().mean()
    short_mean = short_term[analyte].dropna().mean()
    long_mean  = long_term[analyte].dropna().mean()
    rows.append({
        'Analyte':           analyte.split(',')[0],
        'Baseline':          round(base_mean, 3),
        'Short-term (0-30d)': round(short_mean, 3),
        'Long-term (0-365d)': round(long_mean, 3),
        'Short % Change':    round(((short_mean - base_mean) / base_mean) * 100, 1),
        'Long % Change':     round(((long_mean  - base_mean) / base_mean) * 100, 1),
    })
print(pd.DataFrame(rows).to_string(index=False))

# --- Independent t-tests: are the differences statistically significant? ---
# Unlike Section 1's paired t-test, here we use an independent samples t-test
# (ttest_ind) because the short/long-term groups are different observations
# from the baseline — they're not paired by fire.
print("\n" + "=" * 70)
print("Statistical significance — Independent t-tests vs baseline")
print("=" * 70)
for analyte in ANALYTES:
    base_vals  = baseline[analyte].dropna()
    short_vals = short_term[analyte].dropna()
    long_vals  = long_term[analyte].dropna()
    print(f"\n  {analyte.split(',')[0]}")
    if len(short_vals) >= 2:
        _, p = stats.ttest_ind(base_vals, short_vals)
        sig = '✅ significant' if p < 0.05 else '❌ not significant'
        print(f"    Baseline vs Short-term: p={p:.4f} {sig}  (n={len(short_vals)})")
    else:
        print(f"    Baseline vs Short-term: not enough samples (n={len(short_vals)})")
    if len(long_vals) >= 2:
        _, p = stats.ttest_ind(base_vals, long_vals)
        sig = '✅ significant' if p < 0.05 else '❌ not significant'
        print(f"    Baseline vs Long-term:  p={p:.4f} {sig}  (n={len(long_vals)})")

# --- Time trend: does the analyte improve as time passes after the fire? ---
# We fit a simple linear regression of analyte value vs days_since_fire.
# A positive slope means the analyte increases over time (may indicate recovery
# for dissolved oxygen, or continued contamination for turbidity/nutrients).
print("\n" + "=" * 70)
print("Time trend: how does each analyte change over the year after a fire?")
print("(Positive coefficient = analyte increases as time passes post-fire)")
print("=" * 70)
for analyte in ANALYTES:
    df_time = long_term[['days_since_fire', analyte]].dropna()
    if len(df_time) < 5:
        continue
    X = df_time[['days_since_fire']].values
    y = df_time[analyte].values
    model = LinearRegression()
    model.fit(X, y)
    r2        = r2_score(y, model.predict(X))
    direction = 'improves ↑' if model.coef_[0] > 0 else 'worsens ↓'
    print(f"  {analyte.split(',')[0]:30s}: {direction} over time  |  coef={model.coef_[0]:.6f}  R²={r2:.4f}")

# --- Boxplot visualization ---
# Boxplots show the distribution of each analyte across the three groups.
# The box covers the middle 50% of values; the line inside is the median.
# Outliers appear as individual dots.
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Water Quality: Baseline vs Short-term vs Long-term Post-Fire', fontsize=13)
axes = axes.flatten()
for i, analyte in enumerate(ANALYTES):
    ax         = axes[i]
    base_vals  = baseline[analyte].dropna()
    short_vals = short_term[analyte].dropna()
    long_vals  = long_term[analyte].dropna()
    ax.boxplot(
        [base_vals, short_vals, long_vals],
        tick_labels=[
            f'Baseline\n(n={len(base_vals)})',
            f'Short-term\n0-30d\n(n={len(short_vals)})',
            f'Long-term\n0-365d\n(n={len(long_vals)})'
        ]
    )
    ax.set_title(analyte.split(',')[0])
    ax.set_ylabel('Value')
plt.tight_layout()
plt.savefig(os.path.join(_GRAPHS, "short_vs_long_term.png"), dpi=150)
plt.show()
print("  Graph saved: short_vs_long_term.png")

# Save the two impacted subsets for use in other analyses
short_term.to_csv(os.path.join(_CSVS, "impacted_short_term.csv"), index=False)
long_term.to_csv(os.path.join(_CSVS, "impacted_long_term.csv"), index=False)
print("  CSV saved: impacted_short_term.csv")
print("  CSV saved: impacted_long_term.csv")


# =============================================================================
# SECTION 4: ONE-MONTH IMPACT FILTER
# =============================================================================
# Produces a focused dataset of only water readings that were:
#   1. Taken at a station physically inside a fire perimeter, AND
#   2. Taken within 30 days of the fire's start date
#
# This is the strictest definition of "immediately fire-impacted" data —
# useful as a high-confidence subset for any further analysis.
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: One-month post-fire impact filter")
print("=" * 70)

# Add a 30-day window end date to each fire for temporal filtering
fires['window_start'] = fires['ALARM_DATE']
fires['window_end']   = fires['ALARM_DATE'] + timedelta(days=30)

# Re-run the spatial join, this time including window columns
# so we can do the temporal filter after joining
joined_with_window = gpd.sjoin(
    water_gdf,
    fires[['FIRE_NAME', 'ALARM_DATE', 'window_start', 'window_end', 'GIS_ACRES', 'geometry']],
    how='inner',
    predicate='within'
)
joined_with_window['SampleDate'] = pd.to_datetime(joined_with_window['SampleDate'])

# Keep only readings taken between the fire start and 30 days after
impacted_1month = joined_with_window[
    (joined_with_window['SampleDate'] >= joined_with_window['window_start']) &
    (joined_with_window['SampleDate'] <= joined_with_window['window_end'])
]

print(f"  Readings inside a fire perimeter: {len(joined_with_window)}")
print(f"  Readings within 1 month of fire:  {len(impacted_1month)}")
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
