import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# LOAD DATA
# ============================================================

water = pd.read_csv("water_quality_allinfo_master.csv")
gdb_path = "C:\\Users\\Val\\CS2500\\DSFinalProject\\fire24_1.gdb"  

fires = gpd.read_file(gdb_path, engine="pyogrio", layer="firep24_1")
fires = fires.to_crs("EPSG:4326")

# ============================================================
# PREPARE DATA
# ============================================================

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
# SPATIAL JOIN — match each water reading to a fire perimeter
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

print(f"Total readings inside any fire perimeter: {len(joined)}")
print(f"Unique fires with water stations inside: {joined['FIRE_NAME'].nunique()}")
print(f"Unique stations inside fire perimeters: {joined['StationCode'].nunique()}")

# ============================================================
# SPLIT INTO BEFORE AND AFTER PER FIRE
# ============================================================

before = joined[joined['days_since_fire'] < 0].copy()
after_1yr = joined[(joined['days_since_fire'] >= 0) & 
                   (joined['days_since_fire'] <= 365)].copy()
after_all = joined[joined['days_since_fire'] >= 0].copy()

print(f"\nReadings BEFORE fire (same perimeter): {len(before)}")
print(f"Readings AFTER fire 0-365 days:        {len(after_1yr)}")
print(f"Readings AFTER fire (all time):        {len(after_all)}")

# ============================================================
# PAIRED ANALYSIS — per fire perimeter
# ============================================================

print("\n" + "="*70)
print("PAIRED ANALYSIS — Before vs After by Fire")
print("="*70)

fire_results = []

for fire_name, fire_group in joined.groupby('FIRE_NAME'):
    fire_before = fire_group[fire_group['days_since_fire'] < 0]
    fire_after = fire_group[(fire_group['days_since_fire'] >= 0) &
                            (fire_group['days_since_fire'] <= 365)]

    # only include fires that have BOTH before and after readings
    if len(fire_before) == 0 or len(fire_after) == 0:
        continue

    row = {
        'Fire': fire_name,
        'GIS_Acres': fire_group['GIS_ACRES'].iloc[0],
        'n_before': len(fire_before),
        'n_after': len(fire_after),
    }

    for analyte in analytes:
        before_vals = fire_before[analyte].dropna()
        after_vals = fire_after[analyte].dropna()
        if len(before_vals) > 0 and len(after_vals) > 0:
            short_name = analyte.split(',')[0]
            row[f'{short_name}_before'] = before_vals.mean()
            row[f'{short_name}_after'] = after_vals.mean()
            row[f'{short_name}_change'] = after_vals.mean() - before_vals.mean()

    fire_results.append(row)

fire_df = pd.DataFrame(fire_results)
print(f"\nFires with both before AND after readings: {len(fire_df)}")
print(f"\n{fire_df[['Fire', 'GIS_Acres', 'n_before', 'n_after']].to_string(index=False)}")

# ============================================================
# AGGREGATE — average change across all fires
# ============================================================

print("\n" + "="*70)
print("AVERAGE CHANGE ACROSS ALL FIRES (Before → After)")
print("="*70)

change_cols = [c for c in fire_df.columns if c.endswith('_change')]
for col in change_cols:
    analyte_name = col.replace('_change', '')
    vals = fire_df[col].dropna()
    if len(vals) == 0:
        continue
    mean_change = vals.mean()
    direction = '↑ increased' if mean_change > 0 else '↓ decreased'
    print(f"{analyte_name:30s}: {mean_change:+.4f}  {direction}  (n={len(vals)} fires)")

# ============================================================
# STATISTICAL SIGNIFICANCE — paired t-test per analyte
# ============================================================

print("\n" + "="*70)
print("STATISTICAL SIGNIFICANCE — Paired t-tests")
print("="*70)

for analyte in analytes:
    short_name = analyte.split(',')[0]
    before_col = f'{short_name}_before'
    after_col = f'{short_name}_after'

    if before_col not in fire_df.columns or after_col not in fire_df.columns:
        continue

    pairs = fire_df[[before_col, after_col]].dropna()
    if len(pairs) < 2:
        print(f"{short_name:30s}: not enough paired fires (n={len(pairs)})")
        continue

    t, p = stats.ttest_rel(pairs[before_col], pairs[after_col])
    sig = '✅ significant' if p < 0.05 else '❌ not significant'
    print(f"{short_name:30s}: p={p:.4f}  {sig}  (n={len(pairs)} fires)")

# ============================================================
# VISUALIZATION — before vs after per analyte
# ============================================================

change_analytes = [a.split(',')[0] for a in analytes 
                   if f"{a.split(',')[0]}_change" in fire_df.columns]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Before vs After Wildfire — Per Fire Perimeter', fontsize=13)
axes = axes.flatten()

for i, short_name in enumerate(change_analytes[:6]):
    ax = axes[i]
    before_col = f'{short_name}_before'
    after_col = f'{short_name}_after'

    if before_col not in fire_df.columns:
        continue

    pairs = fire_df[[before_col, after_col, 'Fire']].dropna()
    
    # plot a line per fire showing before -> after
    for _, row in pairs.iterrows():
        ax.plot([0, 1], [row[before_col], row[after_col]], 
                'o-', alpha=0.5, color='steelblue')

    # plot the mean line
    ax.plot([0, 1], [pairs[before_col].mean(), pairs[after_col].mean()],
            'o-', color='red', linewidth=3, label='Mean')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before', 'After'])
    ax.set_title(short_name)
    ax.legend()

plt.tight_layout()
plt.savefig("paired_before_after.png", dpi=150)
plt.show()
print("\nPlot saved to paired_before_after.png")

fire_df.to_csv("fire_paired_results.csv", index=False)
print("Saved fire_paired_results.csv")