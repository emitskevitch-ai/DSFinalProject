import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

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
    'Arsenic, Total',
    'Cadmium, Total',
    'Chromium, Total',
    'Total Organic Carbon, Total'
]

# ============================================================
# STEP 1: SPATIAL JOIN WITH BUFFER ZONES
# reproject to meters for accurate distance buffering
# EPSG:3310 is California Albers — accurate for CA distances
# ============================================================

print("Reprojecting to meters for buffering...")
water_m = water_gdf.to_crs("EPSG:3310")
fires_m = fires.to_crs("EPSG:3310")

BUFFER_KM = 30  # 30km downstream buffer
buffer_m = BUFFER_KM * 1000

print(f"Creating {BUFFER_KM}km buffer around fire perimeters...")
fires_buffered = fires_m.copy()
fires_buffered['geometry'] = fires_m.geometry.buffer(buffer_m)

# spatial join — stations inside the buffered fire perimeters
print("Running spatial join with buffer...")
joined_buffer = gpd.sjoin(
    water_m,
    fires_buffered[['FIRE_NAME', 'ALARM_DATE', 'GIS_ACRES', 'geometry']],
    how='inner',
    predicate='within'
)

joined_buffer['SampleDate'] = pd.to_datetime(joined_buffer['SampleDate'])
joined_buffer['days_since_fire'] = (
    joined_buffer['SampleDate'] - joined_buffer['ALARM_DATE']
).dt.days

# also do the original (no buffer) join for comparison
print("Running original spatial join (no buffer)...")
joined_orig = gpd.sjoin(
    water_m,
    fires_m[['FIRE_NAME', 'ALARM_DATE', 'GIS_ACRES', 'geometry']],
    how='inner',
    predicate='within'
)
joined_orig['SampleDate'] = pd.to_datetime(joined_orig['SampleDate'])
joined_orig['days_since_fire'] = (
    joined_orig['SampleDate'] - joined_orig['ALARM_DATE']
).dt.days

print(f"\nWithout buffer — total readings: {len(joined_orig)}")
print(f"With {BUFFER_KM}km buffer — total readings: {len(joined_buffer)}")

before_buf = joined_buffer[joined_buffer['days_since_fire'] < 0]
after_buf = joined_buffer[(joined_buffer['days_since_fire'] >= 0) & 
                          (joined_buffer['days_since_fire'] <= 365)]

print(f"\nWith buffer — before fire readings: {len(before_buf)}")
print(f"With buffer — after fire readings (0-365 days): {len(after_buf)}")
print(f"Unique stations in buffered set: {joined_buffer['StationCode'].nunique()}")
print(f"Unique fires in buffered set: {joined_buffer['FIRE_NAME'].nunique()}")

# ============================================================
# STEP 2: PREPARE DATA FOR MIXED EFFECTS MODEL
# ============================================================

# use 5 year window to maximize post-fire readings
df_model = joined_buffer[
    (joined_buffer['days_since_fire'] >= -365*5) &  # up to 5 years before
    (joined_buffer['days_since_fire'] <= 365*5)      # up to 5 years after
].copy()

# binary flag: 0 = before fire, 1 = after fire
df_model['post_fire'] = (df_model['days_since_fire'] >= 0).astype(int)

# log transform fire size to reduce skew
df_model['log_acres'] = np.log1p(df_model['GIS_ACRES'])

# add season as a control variable
df_model['month'] = df_model['SampleDate'].dt.month
df_model['season'] = df_model['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

print(f"\nModel dataset size: {len(df_model)}")
print(f"Stations (random effect groups): {df_model['StationCode'].nunique()}")

# ============================================================
# STEP 3: MIXED EFFECTS MODEL — one per analyte
# ============================================================

print("\n" + "="*70)
print("MIXED EFFECTS MODEL RESULTS")
print("Fixed effects: post_fire + days_since_fire + log_acres + season")
print("Random effect: StationCode (each station gets its own baseline)")
print("="*70)

model_results = []

for analyte in analytes:
    short = analyte.split(',')[0]
    col = analyte

    df_a = df_model[['StationCode', 'post_fire', 'days_since_fire',
                      'log_acres', 'season', col]].dropna()

    if len(df_a) < 20 or df_a['StationCode'].nunique() < 3:
        print(f"\n{short}: not enough data (n={len(df_a)})")
        continue

    try:
        # mixed effects model:
        # analyte ~ post_fire + days_since_fire + log_acres + season
        # random intercept per station (each station has its own baseline)
        formula = f'Q("{col}") ~ post_fire + days_since_fire + log_acres + season'
        model = smf.mixedlm(formula, df_a, groups=df_a['StationCode'])
        result = model.fit(reml=True)

        coef = result.params['post_fire']
        pval = result.pvalues['post_fire']
        sig = '✅ significant' if pval < 0.05 else '❌ not significant'
        direction = '↑ increases' if coef > 0 else '↓ decreases'

        print(f"\n{short}")
        print(f"  post_fire coefficient: {coef:+.4f}  {direction} after fire")
        print(f"  p-value:               {pval:.4f}  {sig}")
        print(f"  n readings:            {len(df_a)}")
        print(f"  n stations:            {df_a['StationCode'].nunique()}")

        model_results.append({
            'Analyte': short,
            'Coefficient': round(coef, 4),
            'p-value': round(pval, 4),
            'Significant': pval < 0.05,
            'Direction': direction,
            'n_readings': len(df_a),
            'n_stations': df_a['StationCode'].nunique()
        })

    except Exception as e:
        print(f"\n{short}: model failed — {e}")

# ============================================================
# STEP 4: SUMMARY TABLE
# ============================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

results_df = pd.DataFrame(model_results)
if not results_df.empty:
    print(results_df.to_string(index=False))
    results_df.to_csv("mixed_effects_results.csv", index=False)
    print("\nSaved to mixed_effects_results.csv")

# ============================================================
# STEP 5: VISUALIZATION — coefficient plot
# ============================================================

if not results_df.empty:
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['green' if s else 'gray' for s in results_df['Significant']]
    bars = ax.barh(results_df['Analyte'], results_df['Coefficient'], color=colors, alpha=0.7)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Coefficient (change in analyte after fire)')
    ax.set_title('Mixed Effects Model — Effect of Wildfire on Water Quality\n(green = statistically significant, gray = not significant)')

    # add p-value labels
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax.text(row['Coefficient'], i, f"  p={row['p-value']:.3f}", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("mixed_effects_coefficients.png", dpi=150)
    plt.show()
    print("Plot saved to mixed_effects_coefficients.png")