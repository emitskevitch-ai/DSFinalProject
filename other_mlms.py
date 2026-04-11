import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pygam import LinearGAM, s, f
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
    'Total Organic Carbon, Total',
    'Arsenic, Total',
    'Cadmium, Total',
    'Chromium, Total',
]

# ============================================================
# SPATIAL JOIN WITH 30KM BUFFER
# ============================================================

print("Reprojecting and buffering...")
water_m = water_gdf.to_crs("EPSG:3310")
fires_m = fires.to_crs("EPSG:3310")
fires_buffered = fires_m.copy()
fires_buffered['geometry'] = fires_m.geometry.buffer(30000)  # 30km

print("Running spatial join...")
joined = gpd.sjoin(
    water_m,
    fires_buffered[['FIRE_NAME', 'ALARM_DATE', 'GIS_ACRES', 'geometry']],
    how='inner',
    predicate='within'
)

joined['SampleDate'] = pd.to_datetime(joined['SampleDate'])
joined['days_since_fire'] = (joined['SampleDate'] - joined['ALARM_DATE']).dt.days

# use 5 year window
df_model = joined[
    (joined['days_since_fire'] >= -365*5) &
    (joined['days_since_fire'] <= 365*5)
].copy()

df_model['post_fire'] = (df_model['days_since_fire'] >= 0).astype(int)
df_model['log_acres'] = np.log1p(df_model['GIS_ACRES'])
df_model['month'] = df_model['SampleDate'].dt.month
df_model['season_num'] = df_model['month'].map({
    12: 0, 1: 0, 2: 0,   # Winter
    3: 1, 4: 1, 5: 1,    # Spring
    6: 2, 7: 2, 8: 2,    # Summer
    9: 3, 10: 3, 11: 3   # Fall
})

features = ['post_fire', 'days_since_fire', 'log_acres', 'season_num',
            'TargetLatitude', 'TargetLongitude', 'Region']

print(f"Model dataset size: {len(df_model)}")
print(f"Unique stations: {df_model['StationCode'].nunique()}")

# ============================================================
# MODEL 1: RANDOM FOREST
# ============================================================

print("\n" + "="*70)
print("RANDOM FOREST RESULTS")
print("="*70)

rf_results = []
feature_importance_all = {}

for analyte in analytes:
    short = analyte.split(',')[0]
    df_a = df_model[features + [analyte]].dropna()

    if len(df_a) < 100:
        print(f"\n{short}: not enough data (n={len(df_a)})")
        continue

    X = df_a[features]
    y = df_a[analyte]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # use a sample for speed if dataset is very large
    if len(X_train) > 50000:
        sample_idx = X_train.sample(50000, random_state=42).index
        X_train = X_train.loc[sample_idx]
        y_train = y_train.loc[sample_idx]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    importances = pd.Series(model.feature_importances_, index=features)
    top_feature = importances.idxmax()
    post_fire_imp = importances['post_fire']
    feature_importance_all[short] = importances

    print(f"\n{short}")
    print(f"  R²:                  {r2:.4f}")
    print(f"  post_fire importance: {post_fire_imp:.4f}  (higher = fire matters more)")
    print(f"  top feature:         {top_feature}")

    rf_results.append({
        'Analyte': short,
        'R²': round(r2, 4),
        'post_fire_importance': round(post_fire_imp, 4),
        'top_feature': top_feature,
        'n': len(df_a)
    })

rf_df = pd.DataFrame(rf_results).sort_values('post_fire_importance', ascending=False)

# ============================================================
# RANDOM FOREST VISUALIZATION — feature importance heatmap
# ============================================================

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
    plt.savefig("rf_feature_importance.png", dpi=150)
    plt.show()
    print("Saved rf_feature_importance.png")

# ============================================================
# MODEL 2: GAM
# ============================================================

print("\n" + "="*70)
print("GAM RESULTS — Effect of Days Since Fire on Each Analyte")
print("="*70)

gam_results = []

# GAM features: days_since_fire, log_acres, season_num, lat, lon, region
# s() = smooth spline, f() = categorical factor
# index:            0                1            2           3    4       5
gam_formula = s(0) + s(1) + s(2) + s(3) + s(4) + f(5)

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle('GAM — Effect of Days Since Fire on Water Quality\n(shaded area = 95% confidence interval)', fontsize=13)
axes = axes.flatten()

gam_features = ['days_since_fire', 'log_acres', 'season_num',
                'TargetLatitude', 'TargetLongitude', 'Region']

for i, analyte in enumerate(analytes):
    short = analyte.split(',')[0]
    df_a = df_model[gam_features + [analyte]].dropna()

    if len(df_a) < 100:
        print(f"\n{short}: not enough data (n={len(df_a)})")
        axes[i].set_title(f"{short}\n(insufficient data)")
        continue

    # sample for speed
    if len(df_a) > 30000:
        df_a = df_a.sample(30000, random_state=42)

    X = df_a[gam_features].values
    y = df_a[analyte].values

    try:
        gam = LinearGAM(gam_formula).fit(X, y)
        r2_gam = gam.statistics_['pseudo_r2']['McFadden']

        # plot the smooth effect of days_since_fire (feature index 0)
        XX = gam.generate_X_grid(term=0, n=200)
        pdep, confi = gam.partial_dependence(term=0, X=XX, width=0.95)

        ax = axes[i]
        ax.plot(XX[:, 0], pdep, color='steelblue', linewidth=2)
        ax.fill_between(XX[:, 0], confi[:, 0], confi[:, 1], alpha=0.3, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Fire date')
        ax.set_title(f"{short}  (R²={r2_gam:.3f})")
        ax.set_xlabel('Days since fire')
        ax.set_ylabel('Partial effect')
        ax.legend(fontsize=8)

        # check if post-fire trend is significant
        # compare mean effect before (days<0) vs after (days>0)
        days = XX[:, 0]
        before_effect = pdep[days < 0].mean() if any(days < 0) else np.nan
        after_effect = pdep[days > 0].mean() if any(days > 0) else np.nan
        change = after_effect - before_effect
        direction = '↑ increases' if change > 0 else '↓ decreases'

        print(f"\n{short}")
        print(f"  R²:                {r2_gam:.4f}")
        print(f"  Effect post-fire:  {change:+.4f}  {direction}")
        print(f"  n readings:        {len(df_a)}")

        gam_results.append({
            'Analyte': short,
            'R²': round(r2_gam, 4),
            'Post-fire change': round(change, 4),
            'Direction': direction,
            'n': len(df_a)
        })

    except Exception as e:
        print(f"\n{short}: GAM failed — {e}")
        axes[i].set_title(f"{short}\n(model failed)")

plt.tight_layout()
plt.savefig("gam_effects.png", dpi=150)
plt.show()
print("\nSaved gam_effects.png")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("\nRandom Forest — sorted by post_fire importance:")
print(rf_df[['Analyte', 'R²', 'post_fire_importance', 'top_feature']].to_string(index=False))

print("\nGAM — post-fire effect direction:")
gam_df = pd.DataFrame(gam_results).sort_values('Post-fire change', ascending=False)
print(gam_df[['Analyte', 'R²', 'Post-fire change', 'Direction']].to_string(index=False))

rf_df.to_csv("random_forest_results.csv", index=False)
gam_df.to_csv("gam_results.csv", index=False)
print("\nSaved random_forest_results.csv and gam_results.csv")