# =============================================================================
# models.py
# =============================================================================
# PURPOSE:
#   This script applies three different statistical/ML modeling approaches to
#   quantify the relationship between wildfires and water quality. It answers:
#     - After accounting for season, location, and fire size, is there still
#       a statistically significant effect of fire on water quality? (Mixed Effects)
#     - Which factors (fire, location, season, fire size) matter most in
#       predicting each analyte? (Random Forest)
#     - What does the non-linear shape of water quality change look like over
#       time after a fire? (GAM)
#     - Is raw fire size (acres) correlated with water quality degradation?
#       (Linear Regression)
#
# PREREQUISITE: Run pipeline.py first to generate the required CSVs.
#
# INPUTS (from csvs/):
#   - water_quality_allinfo_master.csv   — All water quality readings
#   - stations_within_fires_sameyear.csv — Fire-impacted stations (same year)
#   - all_stations_fire_joined.csv       — All stations with fire info attached
#   - fire24_1.gdb                       — Fire perimeter polygons
#
# OUTPUTS:
#   Graphs (to graphs/):
#     - mixed_effects_coefficients.png  — Bar chart of model coefficients
#     - rf_feature_importance.png       — Heatmap of Random Forest feature importance
#     - gam_effects.png                 — GAM smooth curves per analyte
#     - water_quality_comparison.png    — Boxplots: impacted vs baseline
#     - fire_size_vs_do.png             — Scatter + regression: fire size vs DO
#
#   CSVs (to csvs/):
#     - mixed_effects_results.csv       — Model coefficients and p-values
#     - random_forest_results.csv       — RF R² and post_fire importance per analyte
#     - gam_results.csv                 — GAM R² and post-fire effect per analyte
# =============================================================================

import os
import warnings
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s, f

# Suppress convergence and deprecation warnings from statsmodels/pygam
# so the terminal output stays readable
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# PATH SETUP
# This script lives in "code files/" — navigate up one level to reach the root.
# -----------------------------------------------------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.join(_HERE, "..")
_CSVS   = os.path.join(_ROOT, "csvs")
_GRAPHS = os.path.join(_ROOT, "graphs")

# -----------------------------------------------------------------------------
# ANALYTES TO ANALYZE
# All 9 analytes are used for the advanced models since they can handle
# more data. The mixed effects and RF/GAM models are more robust to noise
# than simple t-tests, so we include the heavier metals here too.
# -----------------------------------------------------------------------------
ANALYTES_FULL = [
    'Oxygen, Dissolved, Total',    # Dissolved oxygen — key indicator of water health
    'Turbidity, Total',            # Cloudiness from suspended particles
    'Nitrogen, Total, Total',      # Nutrient pollution from ash/soil runoff
    'pH',                          # Acidity
    'Phosphorus as P, Total',      # Nutrient pollution
    'Total Organic Carbon, Total', # Organic matter from burned vegetation
    'Arsenic, Total',              # Heavy metal — can leach from burned soil
    'Cadmium, Total',              # Heavy metal — toxic at low concentrations
    'Chromium, Total',             # Heavy metal — found in some soil types
]

# Subset used for simpler regression analyses (excludes rare heavy metals)
ANALYTES_CORE = [
    'Oxygen, Dissolved, Total',
    'Turbidity, Total',
    'Nitrogen, Total, Total',
    'pH',
    'Phosphorus as P, Total',
    'Total Organic Carbon, Total',
    'Arsenic, Total',
]


# =============================================================================
# DATA LOADING
# =============================================================================
print("Loading data...")

# Master water quality table (all stations, all readings)
water = pd.read_csv(os.path.join(_CSVS, "water_quality_allinfo_master.csv"))
water['SampleDate'] = pd.to_datetime(water['SampleDate'])

# Fire perimeters geodatabase
gdb_path = os.path.join(_ROOT, "fire24_1.gdb")
fires = gpd.read_file(gdb_path, engine="pyogrio", layer="firep24_1")
fires = fires.to_crs("EPSG:4326")
fires['ALARM_DATE'] = pd.to_datetime(fires['ALARM_DATE'], utc=True).dt.tz_localize(None)

# Convert water stations to geographic point features for spatial operations
water_gdf = gpd.GeoDataFrame(
    water,
    geometry=gpd.points_from_xy(water['TargetLongitude'], water['TargetLatitude']),
    crs="EPSG:4326"
)

# Pre-built CSVs from pipeline.py — used by the regression section
# "baseline" = all water readings (includes non-impacted stations)
# "impacted"  = only stations inside a fire perimeter in the same year as the fire
# "all_joined"= all stations with fire columns attached (NaN where no fire match)
baseline   = pd.read_csv(os.path.join(_CSVS, "water_quality_allinfo_master.csv"))
impacted   = pd.read_csv(os.path.join(_CSVS, "stations_within_fires_sameyear.csv"))
all_joined = pd.read_csv(os.path.join(_CSVS, "all_stations_fire_joined.csv"))

# Label each dataset so we can combine them later for regression
baseline['impacted']  = 0  # 0 = not fire-impacted
impacted['impacted']  = 1  # 1 = fire-impacted


# =============================================================================
# MODEL 1: MIXED EFFECTS MODEL
# =============================================================================
# A mixed effects model (also called a multilevel model) is a more rigorous
# statistical test than a simple t-test. It accounts for:
#
#   Fixed effects (factors we explicitly control for):
#     - post_fire:       was this reading taken after the fire? (0 or 1)
#     - days_since_fire: how many days since the fire started?
#     - log_acres:       how large was the fire? (log-transformed to reduce skew)
#     - season:          what time of year was it? (Winter/Spring/Summer/Fall)
#
#   Random effect (factor that varies by group but we don't directly interpret):
#     - StationCode: each water station gets its own baseline level
#       This is important because some stations always have higher DO, lower pH,
#       etc. just due to their location — not because of fire. The random effect
#       removes that baseline variation so we only see the fire's contribution.
#
# The key output is the "post_fire coefficient" — how much does the analyte
# change, on average, after a fire, after controlling for all other factors?
# A significant p-value (< 0.05) means the change is unlikely to be random.
#
# SPATIAL NOTE: We use a 30km buffer around fire perimeters here instead of
# exact containment. This captures downstream effects — contaminants from a
# fire can travel through waterways for kilometers beyond the burn area.
# EPSG:3310 (California Albers) is used instead of EPSG:4326 for the buffer
# because buffering in degrees (lat/lon) gives distorted distances. Albers
# is a meter-based projection accurate for California.
# =============================================================================
print("\n" + "=" * 70)
print("MODEL 1: Mixed Effects Model (with 30km downstream buffer)")
print("=" * 70)

# --- Reproject to California Albers for accurate meter-based buffering ---
BUFFER_KM = 30
print(f"  Reprojecting to EPSG:3310 (California Albers) for accurate distances...")
water_m  = water_gdf.to_crs("EPSG:3310")
fires_m  = fires.to_crs("EPSG:3310")

# --- Create buffered fire perimeters ---
# Each fire polygon is expanded outward by BUFFER_KM kilometers.
# A water station that falls inside the expanded polygon is considered
# "downstream" of the fire even if it wasn't inside the original burn area.
print(f"  Creating {BUFFER_KM}km buffer around fire perimeters...")
fires_buffered = fires_m.copy()
fires_buffered['geometry'] = fires_m.geometry.buffer(BUFFER_KM * 1000)  # convert km to meters

# --- Spatial join with the buffered perimeters ---
print("  Running spatial join with buffer...")
joined_buffer = gpd.sjoin(
    water_m,
    fires_buffered[['FIRE_NAME', 'ALARM_DATE', 'GIS_ACRES', 'geometry']],
    how='inner',
    predicate='within'
)
joined_buffer['SampleDate']      = pd.to_datetime(joined_buffer['SampleDate'])
joined_buffer['days_since_fire'] = (joined_buffer['SampleDate'] - joined_buffer['ALARM_DATE']).dt.days

# --- Also run the unbuffered join for comparison ---
# This lets us see how many more stations are captured by the buffer.
joined_orig = gpd.sjoin(
    water_m,
    fires_m[['FIRE_NAME', 'ALARM_DATE', 'GIS_ACRES', 'geometry']],
    how='inner',
    predicate='within'
)
joined_orig['SampleDate']      = pd.to_datetime(joined_orig['SampleDate'])
joined_orig['days_since_fire'] = (joined_orig['SampleDate'] - joined_orig['ALARM_DATE']).dt.days

print(f"\n  Without buffer — total readings: {len(joined_orig)}")
print(f"  With {BUFFER_KM}km buffer — total readings: {len(joined_buffer)}")

# --- Prepare modeling dataset ---
# Use a 5-year window (5 years before and after each fire) to maximize the
# number of readings available for the model. A narrower window would give
# fewer data points, making the model less reliable.
df_model = joined_buffer[
    (joined_buffer['days_since_fire'] >= -365 * 5) &  # up to 5 years before
    (joined_buffer['days_since_fire'] <= 365 * 5)     # up to 5 years after
].copy()

# Binary flag: 0 = before fire, 1 = after fire
# This is the main variable of interest in the model
df_model['post_fire'] = (df_model['days_since_fire'] >= 0).astype(int)

# Log-transform fire size to reduce the influence of extreme outliers
# (a 500,000 acre fire shouldn't dominate the model just because of its size)
df_model['log_acres'] = np.log1p(df_model['GIS_ACRES'])

# Encode season as a categorical variable — water quality varies naturally
# by season, so we need to control for it
df_model['month']  = df_model['SampleDate'].dt.month
df_model['season'] = df_model['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3:  'Spring', 4: 'Spring', 5: 'Spring',
    6:  'Summer', 7: 'Summer', 8: 'Summer',
    9:  'Fall',   10: 'Fall',  11: 'Fall'
})

print(f"\n  Model dataset size: {len(df_model)} readings")
print(f"  Number of unique stations (random effects): {df_model['StationCode'].nunique()}")

# --- Fit one mixed effects model per analyte ---
model_results = []
for analyte in ANALYTES_FULL:
    short = analyte.split(',')[0]

    # Drop rows missing any variable needed by this model
    df_a = df_model[
        ['StationCode', 'post_fire', 'days_since_fire', 'log_acres', 'season', analyte]
    ].dropna()

    # Need at least 20 readings and at least 3 stations for a valid random effect
    if len(df_a) < 20 or df_a['StationCode'].nunique() < 3:
        print(f"\n  {short}: not enough data (n={len(df_a)})")
        continue

    try:
        # Model formula:
        #   analyte ~ post_fire + days_since_fire + log_acres + season
        #   groups=StationCode means each station gets a random intercept
        # Q() wraps analyte names that contain spaces or commas
        formula = f'Q("{analyte}") ~ post_fire + days_since_fire + log_acres + season'
        model   = smf.mixedlm(formula, df_a, groups=df_a['StationCode'])
        result  = model.fit(reml=True)  # REML = Restricted Maximum Likelihood (standard for mixed models)

        coef      = result.params['post_fire']    # how much the analyte changes after fire
        pval      = result.pvalues['post_fire']   # probability this is due to chance
        sig       = '✅ significant' if pval < 0.05 else '❌ not significant'
        direction = '↑ increases' if coef > 0 else '↓ decreases'

        print(f"\n  {short}")
        print(f"    post_fire coefficient: {coef:+.4f}  ({direction} after fire)")
        print(f"    p-value:               {pval:.4f}  {sig}")
        print(f"    n readings:            {len(df_a)}")
        print(f"    n stations:            {df_a['StationCode'].nunique()}")

        model_results.append({
            'Analyte':     short,
            'Coefficient': round(coef, 4),
            'p-value':     round(pval, 4),
            'Significant': pval < 0.05,
            'Direction':   direction,
            'n_readings':  len(df_a),
            'n_stations':  df_a['StationCode'].nunique()
        })

    except Exception as e:
        print(f"\n  {short}: model failed — {e}")

# --- Summary table and visualization ---
results_df = pd.DataFrame(model_results)
if not results_df.empty:
    print("\n" + "=" * 70)
    print("MIXED EFFECTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(_CSVS, "mixed_effects_results.csv"), index=False)
    print("\n  CSV saved: mixed_effects_results.csv")

    # Coefficient bar chart:
    # Each bar is one analyte. Bars right of zero = analyte increases after fire.
    # Green = statistically significant effect. Gray = not significant.
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['green' if s else 'gray' for s in results_df['Significant']]
    ax.barh(results_df['Analyte'], results_df['Coefficient'], color=colors, alpha=0.7)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Coefficient (change in analyte value after fire)')
    ax.set_title(
        'Mixed Effects Model — Effect of Wildfire on Water Quality\n'
        '(green = statistically significant, gray = not significant)'
    )
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax.text(row['Coefficient'], i, f"  p={row['p-value']:.3f}", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(_GRAPHS, "mixed_effects_coefficients.png"), dpi=150)
    plt.show()
    print("  Graph saved: mixed_effects_coefficients.png")


# =============================================================================
# MODEL 2: RANDOM FOREST
# =============================================================================
# A Random Forest is an ensemble of decision trees. Unlike the mixed effects
# model (which gives a single coefficient), Random Forest can capture non-linear
# relationships and interactions between variables.
#
# Key output: FEATURE IMPORTANCE — how much does each predictor variable
# contribute to the model's predictions? A high "post_fire" importance means
# that knowing whether a reading was taken before or after a fire is one of
# the best predictors of that analyte's value.
#
# Features used:
#   - post_fire:         before/after fire (0 or 1)
#   - days_since_fire:   time since fire started
#   - log_acres:         log of fire size
#   - season_num:        season encoded as 0-3
#   - TargetLatitude:    station latitude (location matters for water quality)
#   - TargetLongitude:   station longitude
#   - Region:            broader geographic region
#
# We reuse the same 30km buffered, 5-year window dataset from Model 1.
# =============================================================================
print("\n" + "=" * 70)
print("MODEL 2: Random Forest — Feature Importance")
print("=" * 70)

# Encode season as a number (0-3) because Random Forest needs numeric inputs
# (the mixed effects model can handle string categories, RF cannot)
df_model['season_num'] = df_model['month'].map({
    12: 0, 1: 0, 2: 0,   # Winter = 0
    3:  1, 4: 1, 5: 1,   # Spring = 1
    6:  2, 7: 2, 8: 2,   # Summer = 2
    9:  3, 10: 3, 11: 3  # Fall   = 3
})

# The features (predictor variables) we give to the model
features = [
    'post_fire',        # Was this reading post-fire?
    'days_since_fire',  # How many days since the fire?
    'log_acres',        # How large was the fire?
    'season_num',       # What season was it?
    'TargetLatitude',   # Where is the station (north/south)?
    'TargetLongitude',  # Where is the station (east/west)?
    'Region'            # Broader geographic region
]

rf_results            = []   # Stores R² and importance scores per analyte
feature_importance_all = {}  # Stores full importance arrays for the heatmap

for analyte in ANALYTES_FULL:
    short = analyte.split(',')[0]
    df_a  = df_model[features + [analyte]].dropna()

    if len(df_a) < 100:
        print(f"\n  {short}: not enough data (n={len(df_a)})")
        continue

    X = df_a[features]
    y = df_a[analyte]

    # Split into 80% training data and 20% test data
    # random_state=42 makes this split reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # If training set is very large, sample down for speed
    # 50,000 rows is plenty for a robust Random Forest
    if len(X_train) > 50000:
        sample_idx = X_train.sample(50000, random_state=42).index
        X_train    = X_train.loc[sample_idx]
        y_train    = y_train.loc[sample_idx]

    # Train the Random Forest
    # n_estimators=100: use 100 trees (more = more accurate but slower)
    # n_jobs=-1: use all available CPU cores to speed up training
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate on the held-out test set
    # R² = proportion of variance explained (0 = useless, 1 = perfect)
    preds = model.predict(X_test)
    r2    = r2_score(y_test, preds)

    # Extract feature importances — how much each feature reduced prediction error
    importances    = pd.Series(model.feature_importances_, index=features)
    top_feature    = importances.idxmax()     # which feature mattered most?
    post_fire_imp  = importances['post_fire'] # specifically, how much did fire matter?
    feature_importance_all[short] = importances

    print(f"\n  {short}")
    print(f"    R²:                   {r2:.4f}  (proportion of variance explained)")
    print(f"    post_fire importance: {post_fire_imp:.4f}  (higher = fire matters more)")
    print(f"    top feature:          {top_feature}")

    rf_results.append({
        'Analyte':              short,
        'R²':                   round(r2, 4),
        'post_fire_importance': round(post_fire_imp, 4),
        'top_feature':          top_feature,
        'n':                    len(df_a)
    })

# Sort by post_fire importance so the most fire-sensitive analytes come first
rf_df = pd.DataFrame(rf_results).sort_values('post_fire_importance', ascending=False)

# --- Feature importance heatmap ---
# Rows = analytes, columns = features.
# Darker color = that feature is more important for predicting that analyte.
# This lets you compare across all analytes at once.
if feature_importance_all:
    imp_df = pd.DataFrame(feature_importance_all).T
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(imp_df.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df.index)
    plt.colorbar(im, ax=ax, label='Feature Importance')
    ax.set_title('Random Forest Feature Importance — All Analytes\n(darker = more important)')
    plt.tight_layout()
    plt.savefig(os.path.join(_GRAPHS, "rf_feature_importance.png"), dpi=150)
    plt.show()
    print("\n  Graph saved: rf_feature_importance.png")

rf_df.to_csv(os.path.join(_CSVS, "random_forest_results.csv"), index=False)
print("  CSV saved: random_forest_results.csv")


# =============================================================================
# MODEL 3: GENERALIZED ADDITIVE MODEL (GAM)
# =============================================================================
# A GAM is like a linear regression, but instead of assuming a straight-line
# relationship between predictors and the outcome, it fits a smooth curve
# (called a "spline") for each predictor. This is useful because the effect
# of fire on water quality is probably not linear — the impact might spike
# right after the fire and then gradually recover over months.
#
# The main output is a smooth curve showing how each analyte changes as a
# function of "days since fire". The shaded region is the 95% confidence
# interval — the wider it is, the less certain we are about that part of
# the curve.
#
# GAM formula terms:
#   s() = smooth spline (for continuous numeric predictors)
#   f() = factor/categorical term (for Region)
#
# IMPORTANT: The formula uses numeric indices (s(0), s(1), ...) that map to
# the position of each feature in the gam_features list. The order of
# gam_features must match the index used in the formula:
#   Index 0: days_since_fire
#   Index 1: log_acres
#   Index 2: season_num
#   Index 3: TargetLatitude
#   Index 4: TargetLongitude
#   Index 5: Region  (f(5) because it's categorical)
# =============================================================================
print("\n" + "=" * 70)
print("MODEL 3: GAM — Smooth effect of days since fire on water quality")
print("=" * 70)

# Feature list for GAM — ORDER MATTERS (must match indices in gam_formula)
gam_features = [
    'days_since_fire',  # index 0 — the main variable of interest
    'log_acres',        # index 1 — fire size control
    'season_num',       # index 2 — seasonal control
    'TargetLatitude',   # index 3 — location control
    'TargetLongitude',  # index 4 — location control
    'Region'            # index 5 — categorical region (uses f() not s())
]

# GAM formula: smooth splines for indices 0-4, categorical factor for index 5
gam_formula = s(0) + s(1) + s(2) + s(3) + s(4) + f(5)

gam_results = []

# Set up a 3x3 subplot grid (one panel per analyte)
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle(
    'GAM — Effect of Days Since Fire on Water Quality\n'
    '(shaded area = 95% confidence interval)',
    fontsize=13
)
axes = axes.flatten()

for i, analyte in enumerate(ANALYTES_FULL):
    short = analyte.split(',')[0]
    df_a  = df_model[gam_features + [analyte]].dropna()

    if len(df_a) < 100:
        print(f"\n  {short}: not enough data (n={len(df_a)})")
        axes[i].set_title(f"{short}\n(insufficient data)")
        continue

    # Sample down for speed — GAMs are slower to fit than RF
    if len(df_a) > 30000:
        df_a = df_a.sample(30000, random_state=42)

    X = df_a[gam_features].values
    y = df_a[analyte].values

    try:
        # Fit the GAM — it automatically finds the smoothness of each spline
        gam    = LinearGAM(gam_formula).fit(X, y)
        r2_gam = gam.statistics_['pseudo_r2']['McFadden']  # goodness-of-fit measure

        # Generate a smooth grid of days_since_fire values to plot the curve
        # term=0 means we're looking at the effect of the first feature (days_since_fire)
        XX             = gam.generate_X_grid(term=0, n=200)
        pdep, confi    = gam.partial_dependence(term=0, X=XX, width=0.95)

        # Plot the smooth curve with confidence band
        ax = axes[i]
        ax.plot(XX[:, 0], pdep, color='steelblue', linewidth=2)
        ax.fill_between(XX[:, 0], confi[:, 0], confi[:, 1], alpha=0.3, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Fire date')  # fire start line
        ax.set_title(f"{short}  (R²={r2_gam:.3f})")
        ax.set_xlabel('Days since fire')
        ax.set_ylabel('Partial effect on analyte')
        ax.legend(fontsize=8)

        # Summarize the direction of change: compare mean curve value before vs after day 0
        days          = XX[:, 0]
        before_effect = pdep[days < 0].mean() if any(days < 0) else np.nan
        after_effect  = pdep[days > 0].mean() if any(days > 0) else np.nan
        change        = after_effect - before_effect
        direction     = '↑ increases' if change > 0 else '↓ decreases'

        print(f"\n  {short}")
        print(f"    R² (pseudo):       {r2_gam:.4f}")
        print(f"    Effect post-fire:  {change:+.4f}  {direction}")
        print(f"    n readings:        {len(df_a)}")

        gam_results.append({
            'Analyte':          short,
            'R²':               round(r2_gam, 4),
            'Post-fire change': round(change, 4),
            'Direction':        direction,
            'n':                len(df_a)
        })

    except Exception as e:
        print(f"\n  {short}: GAM failed — {e}")
        axes[i].set_title(f"{short}\n(model failed)")

plt.tight_layout()
plt.savefig(os.path.join(_GRAPHS, "gam_effects.png"), dpi=150)
plt.show()
print("\n  Graph saved: gam_effects.png")

gam_df = pd.DataFrame(gam_results).sort_values('Post-fire change', ascending=False)
gam_df.to_csv(os.path.join(_CSVS, "gam_results.csv"), index=False)
print("  CSV saved: gam_results.csv")


# --- Combined RF + GAM summary ---
print("\n" + "=" * 70)
print("COMBINED MODEL SUMMARY")
print("=" * 70)
print("\nRandom Forest — sorted by post_fire importance:")
print(rf_df[['Analyte', 'R²', 'post_fire_importance', 'top_feature']].to_string(index=False))
print("\nGAM — post-fire effect direction:")
print(gam_df[['Analyte', 'R²', 'Post-fire change', 'Direction']].to_string(index=False))


# =============================================================================
# MODEL 4: LINEAR REGRESSION — FIRE SIZE vs WATER QUALITY
# =============================================================================
# Simpler than the models above but useful for a different question:
# Does fire SIZE predict the degree of water quality degradation?
# A bigger fire burning more acres might wash more ash and contaminants into
# waterways, resulting in larger changes to analyte levels.
#
# Part 1: Simple regression — fire size (GIS_ACRES) vs each analyte individually
# Part 2: Multi-feature regression — fire size + region + impacted flag
#         to predict dissolved oxygen with multiple controls at once
#
# This section reads from the pre-built CSVs (pipeline.py outputs) rather than
# running a new spatial join, so it uses same-year fire matches rather than
# the 30km buffer used by Models 1-3.
# =============================================================================
print("\n" + "=" * 70)
print("MODEL 4: Linear Regression — Fire Size vs Water Quality")
print("=" * 70)

# Tag and combine the two datasets for the multi-feature regression
# (baseline = not impacted, impacted = inside a fire perimeter)
shared_cols = [
    'WaterQualityID', 'StationCode', 'StationName',
    'TargetLatitude', 'TargetLongitude', 'Region', 'SampleDate',
    'Arsenic, Total', 'Cadmium, Total', 'Chromium, Total',
    'Nitrogen, Total, Total', 'Oxygen, Dissolved, Total',
    'Phosphorus as P, Total', 'Total Organic Carbon, Total',
    'Turbidity, Total', 'pH', 'impacted'
]
combined = pd.concat([baseline[shared_cols], impacted[shared_cols]], ignore_index=True)
print(f"  Baseline samples: {len(baseline)}")
print(f"  Impacted samples: {len(impacted)}")

# Only use rows where fire size is known
fire_data = all_joined.dropna(subset=['GIS_ACRES'])

# --- Part 1: simple regression per analyte ---
print("\nPart 1: Simple regression — fire size (GIS_ACRES) vs each analyte")
print("-" * 60)
for analyte in ANALYTES_CORE:
    df = fire_data[['GIS_ACRES', analyte]].dropna()
    if len(df) < 10:
        continue
    X = df[['GIS_ACRES']].values
    y = df[analyte].values
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    print(f"\n  {analyte}")
    print(f"    Coefficient: {model.coef_[0]:.6f}  (change per acre burned)")
    print(f"    R²:          {r2:.4f}  (higher = fire size explains more variation)")

# --- Part 2: multi-feature regression for dissolved oxygen ---
# Dissolved oxygen is the most directly meaningful water quality indicator
# (low DO = fish and aquatic life are stressed), so we build a more complete
# model specifically for it using fire size + region + impacted flag.
print("\nPart 2: Multi-feature regression predicting Dissolved Oxygen")
print("-" * 60)
target = 'Oxygen, Dissolved, Total'

# Build a combined dataset:
# - Impacted rows: use all_joined which has GIS_ACRES
# - Baseline rows: use all baseline readings, set GIS_ACRES = 0 (no fire)
df_fire    = all_joined[['Region', 'GIS_ACRES', target]].dropna()
df_fire['impacted'] = 1

df_baseline = baseline[['Region', target]].dropna()
df_baseline['GIS_ACRES']  = 0  # no fire → treat acres as 0
df_baseline['impacted']   = 0

df_full = pd.concat([df_fire, df_baseline], ignore_index=True).dropna()

X = df_full[['GIS_ACRES', 'Region', 'impacted']]
y = df_full[target]

# 80/20 train-test split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# StandardScaler normalizes each feature to have mean=0 and std=1
# This makes the coefficients comparable across features with very different scales
# (GIS_ACRES is in the hundreds of thousands; Region is encoded as a small integer)
scaler   = StandardScaler()
X_tr_s   = scaler.fit_transform(X_tr)   # fit on training data only
X_val_s  = scaler.transform(X_val)      # apply same scaling to test data

model_multi = LinearRegression()
model_multi.fit(X_tr_s, y_tr)
r2_val = r2_score(y_val, model_multi.predict(X_val_s))

print(f"\n  Target: {target}")
print(f"  R² on validation set: {r2_val:.4f}")
print(f"  Standardized coefficients (larger magnitude = stronger effect):")
for feat, coef in zip(['GIS_ACRES', 'Region', 'impacted'], model_multi.coef_):
    print(f"    {feat}: {coef:.4f}")

# --- Visualizations ---
# Boxplot: impacted vs baseline across all 6 core analytes
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Water Quality: Fire-Impacted vs Baseline', fontsize=14)
axes = axes.flatten()
for i, analyte in enumerate(ANALYTES_CORE[:6]):
    ax        = axes[i]
    base_vals = baseline[analyte].dropna()
    imp_vals  = impacted[analyte].dropna()
    ax.boxplot([base_vals, imp_vals], labels=['Baseline', 'Impacted'])
    ax.set_title(analyte.split(',')[0])
    ax.set_ylabel('Value')
plt.tight_layout()
plt.savefig(os.path.join(_GRAPHS, "water_quality_comparison.png"), dpi=150)
plt.show()
print("\n  Graph saved: water_quality_comparison.png")

# Scatter + regression line: fire size vs dissolved oxygen
# This directly visualizes whether larger fires cause lower DO
df_plot  = fire_data[['GIS_ACRES', 'Oxygen, Dissolved, Total']].dropna()
X_plot   = df_plot[['GIS_ACRES']].values
y_plot   = df_plot['Oxygen, Dissolved, Total'].values
model_do = LinearRegression()
model_do.fit(X_plot, y_plot)

plt.figure(figsize=(8, 5))
plt.scatter(X_plot, y_plot, alpha=0.3, label='Individual readings')
plt.plot(sorted(X_plot), model_do.predict(sorted(X_plot)),
         color='red', label='Regression line')
plt.xlabel('Fire Size (GIS Acres)')
plt.ylabel('Dissolved Oxygen')
plt.title('Fire Size vs Dissolved Oxygen\n(downward slope = larger fires → lower DO)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(_GRAPHS, "fire_size_vs_do.png"), dpi=150)
plt.show()
print("  Graph saved: fire_size_vs_do.png")


print("\n" + "=" * 70)
print("models.py complete.")
print("=" * 70)
