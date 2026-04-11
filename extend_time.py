import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ============================================================
# LOAD DATA
# ============================================================

water = pd.read_csv("water_quality_allinfo_master.csv")
gdb_path = "C:\\Users\\Val\\CS2500\\DSFinalProject\\fire24_1.gdb" 

fires = gpd.read_file(gdb_path, engine="pyogrio", layer="firep24_1")
fires = fires.to_crs("EPSG:4326")

water['SampleDate'] = pd.to_datetime(water['SampleDate'])
water_gdf = gpd.GeoDataFrame(
    water,
    geometry=gpd.points_from_xy(water['TargetLongitude'], water['TargetLatitude']),
    crs="EPSG:4326"
)

fires['ALARM_DATE'] = pd.to_datetime(fires['ALARM_DATE'], utc=True).dt.tz_localize(None)

analytes = [
    'Oxygen, Dissolved, Total',
    'Turbidity, Total',
    'Nitrogen, Total, Total',
    'pH',
    'Phosphorus as P, Total',
    'Total Organic Carbon, Total',
]

# ============================================================
# SPATIAL JOIN
# ============================================================

print("Running spatial join...")
joined = gpd.sjoin(
    water_gdf,
    fires[['FIRE_NAME', 'ALARM_DATE', 'GIS_ACRES', 'geometry']],
    how='inner',
    predicate='within'
)

joined['SampleDate'] = pd.to_datetime(joined['SampleDate'])
joined['days_since_fire'] = (joined['SampleDate'] - joined['ALARM_DATE']).dt.days

# ============================================================
# HELPER: build paired fire results for a given after window
# ============================================================

def build_fire_results(joined, after_days, min_acres=0):
    """
    For each fire with both before and after readings,
    compute mean analyte values before and after the fire.
    
    after_days: max days after fire to include
    min_acres: minimum fire size to include
    """
    subset = joined[joined['GIS_ACRES'] >= min_acres]
    before = subset[subset['days_since_fire'] < 0]
    after = subset[(subset['days_since_fire'] >= 0) &
                   (subset['days_since_fire'] <= after_days)]

    results = []
    for fire_name, fire_group in subset.groupby('FIRE_NAME'):
        fire_before = before[before['FIRE_NAME'] == fire_name]
        fire_after = after[after['FIRE_NAME'] == fire_name]

        if len(fire_before) == 0 or len(fire_after) == 0:
            continue

        row = {
            'Fire': fire_name,
            'GIS_Acres': fire_group['GIS_ACRES'].iloc[0],
            'n_before': len(fire_before),
            'n_after': len(fire_after),
        }
        for analyte in analytes:
            short = analyte.split(',')[0]
            b = fire_before[analyte].dropna()
            a = fire_after[analyte].dropna()
            if len(b) > 0 and len(a) > 0:
                row[f'{short}_before'] = b.mean()
                row[f'{short}_after'] = a.mean()
                row[f'{short}_change'] = a.mean() - b.mean()
        results.append(row)

    return pd.DataFrame(results)

def summarize(fire_df, label):
    """Print average change and t-test results for a set of fires."""
    print(f"\n{'='*70}")
    print(f"{label}  (n={len(fire_df)} fires)")
    print(f"{'='*70}")

    summary = []
    for analyte in analytes:
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
            sig = '✅' if p < 0.05 else '❌'
            print(f"{short:30s}: {mean_change:+.4f} {direction}  p={p:.4f} {sig}  (n={len(pairs)} fires)")
            summary.append({'Analyte': short, 'Mean Change': mean_change, 'p-value': p, 'n_fires': len(pairs)})
        else:
            print(f"{short:30s}: {mean_change:+.4f} {direction}  (not enough fires)")

    return pd.DataFrame(summary)

# ============================================================
# SCENARIO 1 — 1 year after, all fire sizes
# ============================================================

df_1yr_all = build_fire_results(joined, after_days=365, min_acres=0)
summary_1yr_all = summarize(df_1yr_all, "1 YEAR AFTER — All Fire Sizes")

# ============================================================
# SCENARIO 2 — 5 years after, all fire sizes
# ============================================================

df_5yr_all = build_fire_results(joined, after_days=365*5, min_acres=0)
summary_5yr_all = summarize(df_5yr_all, "5 YEARS AFTER — All Fire Sizes")

# ============================================================
# SCENARIO 3 — 1 year after, large fires only (>50,000 acres)
# ============================================================

LARGE_FIRE_THRESHOLD = 50000
df_1yr_large = build_fire_results(joined, after_days=365, min_acres=LARGE_FIRE_THRESHOLD)
summary_1yr_large = summarize(df_1yr_large, f"1 YEAR AFTER — Large Fires Only (>{LARGE_FIRE_THRESHOLD:,} acres)")

# ============================================================
# SCENARIO 4 — 5 years after, large fires only
# ============================================================

df_5yr_large = build_fire_results(joined, after_days=365*5, min_acres=LARGE_FIRE_THRESHOLD)
summary_5yr_large = summarize(df_5yr_large, f"5 YEARS AFTER — Large Fires Only (>{LARGE_FIRE_THRESHOLD:,} acres)")

# ============================================================
# VISUALIZATION — 4 panel comparison
# ============================================================

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

        vals = df[change_col].dropna()
        colors = ['red' if v < 0 else 'steelblue' for v in vals]
        ax.bar(range(len(vals)), vals.values, color=colors, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axhline(vals.mean(), color='orange', linewidth=2, linestyle='--', label=f'Mean: {vals.mean():+.2f}')
        ax.legend(fontsize=7)

        if row_i == 0:
            ax.set_title(label, fontsize=10)
        if col_i == 0:
            ax.set_ylabel(short, fontsize=9)

plt.tight_layout()
plt.savefig("four_scenario_comparison.png", dpi=150)
plt.show()
print("\nPlot saved to four_scenario_comparison.png")

# ============================================================
# SAVE RESULTS
# ============================================================

df_5yr_large.to_csv("large_fires_5yr_results.csv", index=False)
print("Saved large_fires_5yr_results.csv")