# PURPOSE:
#   Applies three statistical/ML modeling approaches to quantify the
#   relationship between wildfires and water quality. It answers:
#     - Which factors (fire, location, season, fire size) matter most in
#       predicting each analyte? (Random Forest Regressor)
#     - What does the non-linear shape of water quality change look like over
#       time after a fire? (GAM)
#     - Which analytes best distinguish pre-fire from post-fire readings?
#       (Random Forest Classifier)
#
# PREREQUISITE: Run pipeline.py first to generate the required CSVs.
#
# INPUTS (from csvs/):
#   - all_stations_fire_joined.csv  — All stations with fire info attached
#
# OUTPUTS:
#   Graphs (to graphs/):
#     - rf_feature_importance.png     — Heatmap of Random Forest feature importance
#     - gam_effects.png               — GAM smooth curves per analyte
#     - analyte_wildfire_impact.png   — RF Classifier: importance + % change
#
#   CSVs (to csvs/):
#     - random_forest_results.csv     — RF R² and post_fire importance per analyte
#     - gam_results.csv               — GAM R² and post-fire effect per analyte
#     - analyte_impact_ranking.csv    — RF Classifier: analyte importance ranking
import os
import traceback
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s

# PATH SETUP
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_CSVS = os.path.join(_ROOT, "csvs")
_GRAPHS = os.path.join(_ROOT, "graphs")

# ANALYTES
# All 9 analytes are used for the Random Forest and GAM models.
# The classifier uses the 6 core analytes (excludes heavy metals because they often have missing data).
ANALYTES = [
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

CLF_ANALYTES = [
    'Oxygen, Dissolved, Total',
    'Turbidity, Total',
    'Nitrogen, Total, Total',
    'pH',
    'Phosphorus as P, Total',
    'Total Organic Carbon, Total',
]

CLF_SHORT_NAMES = {
    'Oxygen, Dissolved, Total': 'Dissolved Oxygen',
    'Turbidity, Total': 'Turbidity',
    'Nitrogen, Total, Total': 'Nitrogen',
    'pH': 'pH',
    'Phosphorus as P, Total': 'Phosphorus',
    'Total Organic Carbon, Total': 'Total Organic Carbon',
}

# DATA LOADING
all_joined = pd.read_csv(os.path.join(_CSVS, "all_stations_fire_joined.csv"))
all_joined['SampleDate'] = pd.to_datetime(all_joined['SampleDate'])
all_joined['ALARM_DATE'] = pd.to_datetime(
    all_joined['ALARM_DATE'], utc=True
).dt.tz_localize(None)
all_joined['days_since_fire'] = (
    all_joined['SampleDate'] - all_joined['ALARM_DATE']
).dt.days


# SHARED DATA PREPARATION
df_model = all_joined[
    (all_joined['days_since_fire'] >= -365 * 5) &
    (all_joined['days_since_fire'] <= 365 * 5)
].copy()

df_model['post_fire'] = (df_model['days_since_fire'] >= 0).astype(int)
df_model['log_acres'] = np.log1p(df_model['GIS_ACRES'])
df_model['month'] = df_model['SampleDate'].dt.month
df_model['season_num'] = df_model['month'].map({
    12: 0, 1: 0, 2: 0,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 2,
    9: 3, 10: 3, 11: 3
})

#caps for the amount of rows that random forest and Gam can work with
RF_SAMPLE = 5000
GAM_SAMPLE = 3000

# MODEL 1: RANDOM FOREST REGRESSOR
# An ensemble of decision trees that can capture non-linear relationships and
# interactions. The main output is feature importance — how much does each
# predictor contribute to explaining an analyte's value?
#
# A high "post_fire" importance means fire timing is one of the best predictors
# of that analyte's level.

features = [
    'days_since_fire', # continuous: how many days before/after the fire date
    'log_acres',       # log-transformed fire size (log1p to handle zeros)
    'season_num',      # 0=winter, 1=spring, 2=summer, 3=fall
    'TargetLatitude',
    'TargetLongitude',
    'Region'
]

# collects R² and importance scores per analyte
rf_results = []          
# collects full importance vectors for the heatmap
feature_importance_all = {}

for analyte in ANALYTES:
    short = analyte.split(',')[0]
    # Drop rows where either a feature or the analyte value is missing
    df_a = df_model[features + [analyte]].dropna()
    # Skip analytes with too little data to train reliably
    if len(df_a) < 100:
        print(f"\n{short}: not enough data (n={len(df_a)})")
        continue
    X = df_a[features]
    y = df_a[analyte]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Cap training size to RF_SAMPLE rows so the model runs in reasonable time
    if len(X_train) > RF_SAMPLE:
        sample_idx = X_train.sample(RF_SAMPLE, random_state=42).index
        X_train = X_train.loc[sample_idx]
        y_train = y_train.loc[sample_idx]
    model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    # Evaluate on the held-out test set
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds) 
    # Feature importance: fraction of variance each predictor explains (sums to 1)
    importances = pd.Series(model.feature_importances_, index=features)
    top_feature = importances.idxmax()        # which predictor matters most
    feature_importance_all[short] = importances  # store for heatmap later
    print(f"\n{short}")
    print(f"R²:{r2:.4f}")
    print(f"top feature:{top_feature}")
    rf_results.append({
        'Analyte': short,
        'R²': round(r2, 4),
        'top_feature': top_feature,
        'n': len(df_a)
    })
# Sort by R² descending so the best-fitting analytes appear first
rf_df = pd.DataFrame(rf_results).sort_values('R²', ascending=False)

# --- Heatmap: feature importance across all analytes ---
if feature_importance_all:
    imp_df = pd.DataFrame(feature_importance_all).T 
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(imp_df.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df.index)
    plt.colorbar(im, ax=ax, label='Feature Importance')
    ax.set_title('Random Forest Feature Importance — All Analytes')
    plt.tight_layout()
    plt.savefig(os.path.join(_GRAPHS, "rf_feature_importance.png"), dpi=150)
    plt.show()

rf_df.to_csv(os.path.join(_CSVS, "random_forest_results.csv"), index=False)


# MODEL 2: GENERALIZED ADDITIVE MODEL (GAM)
# A GAM fits a smooth curve (spline) for each predictor rather than a straight
# line. The main output is a smooth curve showing how each analyte changes as
# a function of days since fire, with a 95% confidence band.
#
# GAM formula indices map to gam_features list order:
#   s(0) = days_since_fire, s(1) = log_acres, s(2) = season_num,
#   s(3) = TargetLatitude,  s(4) = TargetLongitude, s(5) = Region

gam_features = [
    'days_since_fire',  # term 0 — the curve we care about plotting
    'log_acres',        # term 1 — control for fire size
    'season_num',       # term 2 — control for seasonality
    'TargetLatitude',   # term 3 — control for geographic location
    'TargetLongitude',  # term 4
    'Region'            # term 5
]

# Each term gets its own flexible curve instead of a fixed slope.
gam_formula = s(0) + s(1) + s(2) + s(3) + s(4) + s(5)
gam_results = []

# Pre-build the 3×3 subplot grid — one panel per analyte in ANALYTES (9 total)
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle('GAM — Effect of Days Since Fire on Water Quality\n' '(shaded area = 95% confidence interval)', fontsize=13)
axes = axes.flatten()  # flatten to index axes[i] instead of axes[row][col]

for i, analyte in enumerate(ANALYTES):
    short = analyte.split(',')[0]
    # Drop rows missing any feature or the target analyte
    df_a = df_model[gam_features + [analyte]].dropna()
    if len(df_a) < 100:
        print(f"\n{short}: not enough data (n={len(df_a)})")
        axes[i].set_title(f"{short}\n(insufficient data)")
        continue
    # GAMs are slow to fit — cap to GAM_SAMPLE rows to keep runtime manageable
    if len(df_a) > GAM_SAMPLE:
        df_a = df_a.sample(GAM_SAMPLE, random_state=42)
    # GAM expects plain numpy arrays, not DataFrames
    X = df_a[gam_features].values
    y = df_a[analyte].values
    try:
        gam = LinearGAM(gam_formula).fit(X, y)
        # Values above 0.2 are considered reasonable for GAMs.
        r2_gam = gam.statistics_['pseudo_r2']['McFadden']
        # XX is a feature matrix
        # the other columns are held at their mean values.
        XX = gam.generate_X_grid(term=0, n=200)
        # partial_dependence isolates the effect of days_since_fire alone,
        pdep, confi = gam.partial_dependence(term=0, X=XX, width=0.95)

        ax = axes[i]
        ax.plot(XX[:, 0], pdep, color='steelblue', linewidth=2)
        ax.fill_between(XX[:, 0], confi[:, 0], confi[:, 1], alpha=0.3, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Fire date')  # day 0 = fire start
        ax.set_title(f"{short}  (R²={r2_gam:.3f})")
        ax.set_xlabel('Days since fire')
        ax.set_ylabel('Partial effect')
        ax.legend(fontsize=8)

        # Summarize the direction of change: average partial effect before vs after day 0
        days = XX[:, 0]
        before_effect = pdep[days < 0].mean() if any(days < 0) else np.nan
        after_effect = pdep[days > 0].mean() if any(days > 0) else np.nan
        change = after_effect - before_effect  # positive = analyte increases post-fire
        direction = '↑ increases' if change > 0 else '↓ decreases'

        print(f"\n{short}")
        print(f"  R²:               {r2_gam:.4f}")
        print(f"  Post-fire change: {change:+.4f}  {direction}")
        print(f"  n readings:       {len(df_a)}")

        gam_results.append({
            'Analyte': short,
            'R²': round(r2_gam, 4),
            'Post-fire change': round(change, 4),
            'Direction': direction,
            'n': len(df_a)
        })

    except Exception as e:
        # GAMs can fail if the data is too sparse or has near-zero variance.
        # Print the error and mark the subplot rather than crashing the whole script.
        traceback.print_exc()
        axes[i].set_title(f"{short}\n(model failed)")

plt.tight_layout()
plt.savefig(os.path.join(_GRAPHS, "gam_effects.png"), dpi=150)
plt.show()

# Sort by magnitude of post-fire change so the most-affected analytes are at the top
gam_df = pd.DataFrame(gam_results).sort_values('Post-fire change', ascending=False) if gam_results else pd.DataFrame()
if not gam_df.empty:
    gam_df.to_csv(os.path.join(_CSVS, "gam_results.csv"), index=False)


# --- Combined RF + GAM summary ---
print("\nRandom Forest — sorted by post_fire importance:")
print(rf_df[['Analyte', 'R²', 'top_feature']].to_string(index=False))
print("\nGAM — post-fire effect direction:")
if not gam_df.empty:
    print(gam_df[['Analyte', 'R²', 'Post-fire change', 'Direction']].to_string(index=False))
else:
    print("No GAM models succeeded — check errors above")


# MODEL 3: RANDOM FOREST CLASSIFIER
# This model flips the question: instead of predicting analyte values from fire
# features, it asks — can water quality measurements alone predict whether a
# reading was taken before or after a fire?
#
# Features: the 6 core analytes (dissolved oxygen, turbidity, nitrogen, etc.)
# Target: post_fire (0 = before fire, 1 = after fire)
#
# A high classification accuracy means fire leaves a measurable fingerprint in
# the water quality data. Feature importance tells you which analytes change
# most distinctly after a fire.

# Pull only the analyte columns and the target label from the 5-year window dataset.
df_clf = df_model[CLF_ANALYTES + ['post_fire']].copy()

# Require at least 3 analytes to be non-null per row.
df_clf = df_clf[df_clf[CLF_ANALYTES].notna().sum(axis=1) >= 3]

# Fill remaining missing values with each analyte's median so every row is complete.
for col in CLF_ANALYTES:
    df_clf[col] = df_clf[col].fillna(df_clf[col].median())

# Print class sizes — large imbalance between pre/post would bias the model
# toward the majority class without class_weight='balanced' below.
print(f"Total samples: {len(df_clf)}")
print(f"Pre-fire:  {(df_clf['post_fire'] == 0).sum()}")
print(f"Post-fire: {(df_clf['post_fire'] == 1).sum()}")
print(f"Class balance: {df_clf['post_fire'].value_counts(normalize=True).round(2).to_dict()}")

X = df_clf[CLF_ANALYTES]
y = df_clf['post_fire']  # binary target: 0 = pre-fire, 1 = post-fire

# StandardScaler normalizes each analyte to mean=0, std=1.
# Random forests don't technically require scaling, but it ensures analytes
# measured on very different scales (e.g. turbidity vs pH) are comparable
# when computing feature importance.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# stratify=y preserves the pre/post class ratio in both train and test sets,
# which matters when classes are imbalanced.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Cap training to 10,000 rows — the classifier already has 100 trees so
# training beyond this adds runtime with diminishing accuracy returns.
if len(X_train) > 10000:
    idx = np.random.RandomState(42).choice(len(X_train), 10000, replace=False)
    X_train = X_train[idx]
    y_train = y_train.iloc[idx]

# class_weight='balanced' automatically upweights the minority class during training,
# compensating for any imbalance between pre-fire and post-fire readings.
clf_model = RandomForestClassifier(
    n_estimators=100,     # 100 trees gives stable importance estimates
    random_state=42,
    n_jobs=-1,            # use all CPU cores
    class_weight='balanced'
)
clf_model.fit(X_train, y_train)

preds = clf_model.predict(X_test)

# classification_report shows precision, recall, and F1 for each class.
# Precision = of all predicted post-fire, how many actually were?
# Recall = of all actual post-fire, how many did we catch?
print(classification_report(y_test, preds, target_names=['Pre-fire', 'Post-fire']))

# Confusion matrix: rows = actual class, columns = predicted class.
# Top-left = correctly predicted pre-fire (true negatives)
# Bottom-right = correctly predicted post-fire (true positives)
# Off-diagonal = misclassifications
cm = confusion_matrix(y_test, preds)
print(f"                Predicted Pre  Predicted Post")
print(f"Actual Pre:     {cm[0,0]:>13}  {cm[0,1]:>13}")
print(f"Actual Post:    {cm[1,0]:>13}  {cm[1,1]:>13}")

# Feature importance: how much each analyte contributed to the classification decision.
# Higher = that analyte's value changed more distinctly between pre and post fire.
importances_clf = pd.Series(clf_model.feature_importances_, index=CLF_ANALYTES)
importances_clf.index = [CLF_SHORT_NAMES[a] for a in CLF_ANALYTES]  # use short display names
importances_clf = importances_clf.sort_values(ascending=False)

# Print a simple bar visualization of importance scores in the terminal
for analyte, imp in importances_clf.items():
    bar = '█' * int(imp * 200)
    print(f"  {analyte:25s}: {imp:.4f}  {bar}")

# For each analyte, compute the actual mean values before and after the fire
# and the % change. This is separate from importance — it tells us the direction
# and magnitude of the real-world change, not just how useful it was for classification.
clf_results = []
for analyte in CLF_ANALYTES:
    before_mean = df_clf[df_clf['post_fire'] == 0][analyte].mean()
    after_mean = df_clf[df_clf['post_fire'] == 1][analyte].mean()
    pct_change = ((after_mean - before_mean) / before_mean) * 100
    direction = '↑' if pct_change > 0 else '↓'
    short = CLF_SHORT_NAMES[analyte]
    imp = importances_clf[short]
    print(f"  {short:25s}: {before_mean:.3f} → {after_mean:.3f}  ({pct_change:+.1f}%)  {direction}  importance={imp:.4f}")
    clf_results.append({
        'Analyte': short,
        'Before': round(before_mean, 3),
        'After': round(after_mean, 3),
        '% Change': round(pct_change, 1),
        'Direction': direction,
        'Importance': round(imp, 4)
    })

clf_results_df = pd.DataFrame(clf_results).sort_values('Importance', ascending=False)

# --- Visualization: two side-by-side horizontal bar charts ---
# Left panel: feature importance (which analytes best distinguish pre vs post fire)
# Right panel: % change (how much each analyte actually changed, and in which direction)
# Blue = increased after fire, red = decreased after fire
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Which Water Quality Analytes Are Most Affected by Wildfires?', fontsize=13)

# Match bar color to the direction of change for that analyte
colors = [
    'steelblue' if clf_results_df.loc[clf_results_df['Analyte'] == a, 'Direction'].values[0] == '↑'
    else 'tomato' for a in importances_clf.index
]
# Reverse order so the most important analyte appears at the top of the chart
axes[0].barh(importances_clf.index[::-1], importances_clf.values[::-1], color=colors[::-1], alpha=0.8)
axes[0].set_xlabel('Feature Importance\n(higher = more affected by wildfire)')
axes[0].set_title('Analyte Importance\n(blue = increases, red = decreases after fire)')
axes[0].axvline(0, color='black', linewidth=0.8)

# Sort by % change so bars go from most-decreased (left) to most-increased (right)
res_sorted = clf_results_df.sort_values('% Change')
bar_colors = ['steelblue' if x > 0 else 'tomato' for x in res_sorted['% Change']]
axes[1].barh(res_sorted['Analyte'], res_sorted['% Change'], color=bar_colors, alpha=0.8)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_xlabel('% Change After Fire')
axes[1].set_title('Direction & Magnitude of Change\n(blue = increases, red = decreases)')

plt.tight_layout()
plt.savefig(os.path.join(_GRAPHS, "analyte_wildfire_impact.png"), dpi=150)
plt.show()
clf_results_df.to_csv(os.path.join(_CSVS, "analyte_impact_ranking.csv"), index=False)


