import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import timedelta

# ============================================================
# LOAD DATA
# ============================================================

water = pd.read_csv("water_quality_allinfo_master.csv")
gdb_path = "C:\\Users\\Val\\CS2500\\DSFinalProject\\fire24_1.gdb"  # update this

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

# ============================================================
# SPATIAL JOIN — stations inside fire perimeters
# ============================================================

print("Running spatial join...")
spatially_joined = gpd.sjoin(
    water_gdf,
    fires[['FIRE_NAME', 'ALARM_DATE', 'GIS_ACRES', 'geometry']],
    how='inner',
    predicate='within'
)

spatially_joined['SampleDate'] = pd.to_datetime(spatially_joined['SampleDate'])
spatially_joined['days_since_fire'] = (
    spatially_joined['SampleDate'] - spatially_joined['ALARM_DATE']
).dt.days

# only keep samples taken AFTER the fire started
after_fire = spatially_joined[spatially_joined['days_since_fire'] >= 0]

# ============================================================
# CREATE THREE GROUPS
# ============================================================

short_term = after_fire[after_fire['days_since_fire'] <= 30]    # 0-30 days
long_term = after_fire[after_fire['days_since_fire'] <= 365]    # 0-365 days
baseline = pd.read_csv("water_quality_allinfo_master.csv")

print(f"\nBaseline samples:         {len(baseline)}")
print(f"Short-term (0-30 days):   {len(short_term)}")
print(f"Long-term  (0-365 days):  {len(long_term)}")

# ============================================================
# ANALYTES
# ============================================================

analytes = [
    'Oxygen, Dissolved, Total',
    'Turbidity, Total',
    'Nitrogen, Total, Total',
    'pH',
    'Phosphorus as P, Total',
    'Total Organic Carbon, Total',
]

# ============================================================
# PART 1: MEAN COMPARISON — 3 groups
# ============================================================

print("\n" + "="*70)
print("PART 1: MEAN COMPARISON — Baseline vs Short-term vs Long-term")
print("="*70)

rows = []
for analyte in analytes:
    base_mean = baseline[analyte].dropna().mean()
    short_mean = short_term[analyte].dropna().mean()
    long_mean = long_term[analyte].dropna().mean()

    rows.append({
        'Analyte': analyte.split(',')[0],
        'Baseline': round(base_mean, 3),
        'Short-term (0-30d)': round(short_mean, 3),
        'Long-term (0-365d)': round(long_mean, 3),
        'Short % Change': round(((short_mean - base_mean) / base_mean) * 100, 1),
        'Long % Change': round(((long_mean - base_mean) / base_mean) * 100, 1),
    })

comparison_df = pd.DataFrame(rows)
print(comparison_df.to_string(index=False))

# ============================================================
# PART 2: STATISTICAL SIGNIFICANCE — t-tests
# ============================================================

print("\n" + "="*70)
print("PART 2: STATISTICAL SIGNIFICANCE (t-tests)")
print("="*70)

for analyte in analytes:
    base_vals = baseline[analyte].dropna()
    short_vals = short_term[analyte].dropna()
    long_vals = long_term[analyte].dropna()

    print(f"\n{analyte.split(',')[0]}")

    if len(short_vals) >= 2:
        t, p = stats.ttest_ind(base_vals, short_vals)
        sig = '✅ significant' if p < 0.05 else '❌ not significant'
        print(f"  Baseline vs Short-term: p={p:.4f} {sig}  (n={len(short_vals)})")
    else:
        print(f"  Baseline vs Short-term: not enough samples (n={len(short_vals)})")

    if len(long_vals) >= 2:
        t, p = stats.ttest_ind(base_vals, long_vals)
        sig = '✅ significant' if p < 0.05 else '❌ not significant'
        print(f"  Baseline vs Long-term:  p={p:.4f} {sig}  (n={len(long_vals)})")

# ============================================================
# PART 3: DAYS SINCE FIRE vs DO — regression over time
# ============================================================

print("\n" + "="*70)
print("PART 3: HOW DOES DO CHANGE OVER TIME AFTER A FIRE?")
print("="*70)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

for analyte in analytes:
    df_time = long_term[['days_since_fire', analyte]].dropna()
    if len(df_time) < 5:
        continue

    X = df_time[['days_since_fire']].values
    y = df_time[analyte].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))

    direction = 'improves ↑' if model.coef_[0] > 0 else 'worsens ↓'
    print(f"{analyte.split(',')[0]}: {direction} over time  |  coef={model.coef_[0]:.6f}  R²={r2:.4f}")

# ============================================================
# PART 4: VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Water Quality: Baseline vs Short-term vs Long-term Post-Fire', fontsize=13)
axes = axes.flatten()

for i, analyte in enumerate(analytes):
    ax = axes[i]
    base_vals = baseline[analyte].dropna()
    short_vals = short_term[analyte].dropna()
    long_vals = long_term[analyte].dropna()

    data_to_plot = [base_vals, short_vals, long_vals]
    labels = [f'Baseline\n(n={len(base_vals)})',
              f'Short-term\n0-30d\n(n={len(short_vals)})',
              f'Long-term\n0-365d\n(n={len(long_vals)})']

    ax.boxplot(data_to_plot, tick_labels=labels)
    ax.set_title(analyte.split(',')[0])
    ax.set_ylabel('Value')

plt.tight_layout()
plt.savefig("short_vs_long_term.png", dpi=150)
plt.show()
print("\nPlot saved to short_vs_long_term.png")

# ============================================================
# PART 5: SAVE DATASETS
# ============================================================

short_term.to_csv("impacted_short_term.csv", index=False)
long_term.to_csv("impacted_long_term.csv", index=False)
print("Saved impacted_short_term.csv and impacted_long_term.csv")