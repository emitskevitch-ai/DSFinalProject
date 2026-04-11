import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# LOAD DATA
# ============================================================

baseline = pd.read_csv("water_quality_allinfo_master.csv")
impacted = pd.read_csv("stations_within_fires_sameyear.csv")
all_joined = pd.read_csv("all_stations_fire_joined.csv")

# ============================================================
# LABEL AND COMBINE
# ============================================================

baseline['impacted'] = 0
impacted['impacted'] = 1

# keep only columns that exist in both
shared_cols = [
    'WaterQualityID', 'StationCode', 'StationName',
    'TargetLatitude', 'TargetLongitude', 'Region', 'SampleDate',
    'Arsenic, Total', 'Cadmium, Total', 'Chromium, Total',
    'Nitrogen, Total, Total', 'Oxygen, Dissolved, Total',
    'Phosphorus as P, Total', 'Total Organic Carbon, Total',
    'Turbidity, Total', 'pH', 'impacted'
]

combined = pd.concat([baseline[shared_cols], impacted[shared_cols]], ignore_index=True)

print(f"Baseline samples: {len(baseline)}")
print(f"Impacted samples: {len(impacted)}")
print(f"Combined samples: {len(combined)}")

# ============================================================
# ANALYTES TO ANALYZE
# ============================================================

analytes = [
    'Oxygen, Dissolved, Total',
    'Turbidity, Total',
    'Nitrogen, Total, Total',
    'pH',
    'Phosphorus as P, Total',
    'Total Organic Carbon, Total',
    'Arsenic, Total',
]

# ============================================================
# PART 1: MEAN COMPARISON — impacted vs baseline
# ============================================================

print("\n" + "="*60)
print("PART 1: MEAN COMPARISON — Impacted vs Baseline")
print("="*60)

comparison = []
for analyte in analytes:
    base_mean = baseline[analyte].dropna().mean()
    imp_mean = impacted[analyte].dropna().mean()
    pct_change = ((imp_mean - base_mean) / base_mean) * 100
    comparison.append({
        'Analyte': analyte,
        'Baseline Mean': round(base_mean, 4),
        'Impacted Mean': round(imp_mean, 4),
        '% Change': round(pct_change, 2)
    })

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string(index=False))

# ============================================================
# PART 2: LINEAR REGRESSION — GIS_ACRES vs each analyte
# ============================================================

print("\n" + "="*60)
print("PART 2: LINEAR REGRESSION — Fire Size vs Water Quality")
print("="*60)

# use all_joined which has GIS_ACRES
fire_data = all_joined.dropna(subset=['GIS_ACRES'])

results = []
for analyte in analytes:
    df = fire_data[['GIS_ACRES', analyte]].dropna()
    if len(df) < 10:
        continue

    X = df[['GIS_ACRES']].values
    y = df[analyte].values

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)

    results.append({
        'Analyte': analyte,
        'Coefficient': round(model.coef_[0], 6),  # change per acre
        'Intercept': round(model.intercept_, 4),
        'R²': round(r2, 4)
    })
    print(f"\n{analyte}")
    print(f"  Coefficient: {model.coef_[0]:.6f}  (change per acre burned)")
    print(f"  R²:          {r2:.4f}")

# ============================================================
# PART 3: MULTI-FEATURE REGRESSION
# ============================================================

print("\n" + "="*60)
print("PART 3: MULTI-FEATURE REGRESSION")
print("="*60)

# predict DO from fire size + region + impacted flag
target = 'Oxygen, Dissolved, Total'

df_multi = combined[['Region', 'impacted', target]].dropna()

# merge in GIS_ACRES from impacted set where available
impacted_acres = impacted[['StationCode', 'GIS_ACRES']].copy()
df_multi2 = all_joined[['Region', 'GIS_ACRES', target]].dropna()
df_multi2['impacted'] = 1

df_base2 = baseline[['Region', target]].dropna()
df_base2['GIS_ACRES'] = 0   # no fire = 0 acres
df_base2['impacted'] = 0

df_full = pd.concat([df_multi2, df_base2], ignore_index=True).dropna()

X = df_full[['GIS_ACRES', 'Region', 'impacted']]
y = df_full[target]

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)

model_multi = LinearRegression()
model_multi.fit(X_tr_s, y_tr)
preds_val = model_multi.predict(X_val_s)
r2_val = r2_score(y_val, preds_val)

print(f"\nPredicting: {target}")
print(f"Features: GIS_ACRES, Region, impacted")
print(f"R² on validation set: {r2_val:.4f}")
print(f"\nCoefficients:")
for feat, coef in zip(['GIS_ACRES', 'Region', 'impacted'], model_multi.coef_):
    print(f"  {feat}: {coef:.4f}")

# ============================================================
# PART 4: VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Water Quality: Impacted vs Baseline', fontsize=14)
axes = axes.flatten()

plot_analytes = analytes[:6]
for i, analyte in enumerate(plot_analytes):
    ax = axes[i]
    base_vals = baseline[analyte].dropna()
    imp_vals = impacted[analyte].dropna()
    ax.boxplot([base_vals, imp_vals], labels=['Baseline', 'Impacted'])
    ax.set_title(analyte.split(',')[0])
    ax.set_ylabel('Value')

plt.tight_layout()
plt.savefig("water_quality_comparison.png", dpi=150)
plt.show()
print("\nPlot saved to water_quality_comparison.png")

# ============================================================
# PART 5: REGRESSION PLOT — Fire Size vs DO
# ============================================================

df_plot = fire_data[['GIS_ACRES', 'Oxygen, Dissolved, Total']].dropna()
X_plot = df_plot[['GIS_ACRES']].values
y_plot = df_plot['Oxygen, Dissolved, Total'].values

model_plot = LinearRegression()
model_plot.fit(X_plot, y_plot)

plt.figure(figsize=(8, 5))
plt.scatter(X_plot, y_plot, alpha=0.3, label='Data points')
plt.plot(sorted(X_plot), model_plot.predict(sorted(X_plot)), color='red', label='Regression line')
plt.xlabel('Fire Size (GIS Acres)')
plt.ylabel('Dissolved Oxygen')
plt.title('Fire Size vs Dissolved Oxygen')
plt.legend()
plt.tight_layout()
plt.savefig("fire_size_vs_do.png", dpi=150)
plt.show()
print("Plot saved to fire_size_vs_do.png")