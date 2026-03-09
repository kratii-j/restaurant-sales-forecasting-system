"""Generate notebooks/10_risk_analysis.ipynb programmatically."""
import json, pathlib

cells = []

def _splitlines(text):
    """Split text into lines with \\n at end of each line except the last (std ipynb format)."""
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _splitlines(source)})

def code(source):
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": _splitlines(source.strip())
    })

# ── Title ───────────────────────────────────────────────────────
md("# 📊 Risk Analysis — Restaurant Sales Forecasting\n\n"
   "This notebook tests and validates the **Risk Scoring Engine** by:\n"
   "1. Loading data & generating predictions with confidence intervals\n"
   "2. Computing risk scores and classifications\n"
   "3. Analyzing risk distribution across **restaurants**, **days of week**, and **holidays vs regular days**\n"
   "4. Validating risk classifications against actual outcomes")

# ── Cell 1: Setup & Imports ─────────────────────────────────────
code("""import sys, os
sys.path.append(os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from src.models.risk_scoring_engine import calculate_risk_score, classify_risk
from src.models.confidence_intervals import validate_interval_coverage

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (12, 5)

print("✅ All imports successful")""")

# ── Cell 2: Load Data ──────────────────────────────────────────
md("## 1 · Load & Prepare Data")

code("""# Load featured orders (has lag/rolling/temporal features)
df = pd.read_csv("../data/processed/featured_orders.csv")
df["date"] = pd.to_datetime(df["date"])

# Load external context for holiday info
ext = pd.read_csv("../data/raw/external_context.csv")
ext["date"] = pd.to_datetime(ext["date"])

# Load restaurant ➜ city mapping
resto = pd.read_csv("../data/raw/restaurant_dataset.csv", encoding="utf-8-sig")
resto.columns = resto.columns.str.lower().str.strip().str.replace(" ", "_")

# Merge city onto orders, then merge holiday flag
df = df.merge(resto[["restaurant_id", "city"]].drop_duplicates(), on="restaurant_id", how="left")
holiday_lookup = ext[["date", "city", "is_holiday"]].drop_duplicates()
df = df.merge(holiday_lookup, on=["date", "city"], how="left")
df["is_holiday"] = df["is_holiday"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)

# Create day-of-week name for analysis
df["day_name"] = df["date"].dt.day_name()

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Restaurants: {df['restaurant_id'].nunique()}")
print(f"Holiday days: {df['is_holiday'].sum():,} rows ({df['is_holiday'].mean()*100:.1f}%)")
df.head()""")

# ── Cell 3: Train-Test Split ───────────────────────────────────
md("## 2 · Train / Test Split & Model Training\n\n"
   "We perform a **time-based split** (80/20) and train LightGBM quantile models to produce\n"
   "point predictions with lower (10th) and upper (90th) confidence bounds.")

code("""import lightgbm as lgb

# Time-based split
df = df.sort_values(["restaurant_id", "date"]).reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test  = df.iloc[split_idx:].copy()

# Features: drop non-feature columns
drop_cols = ["restaurant_id", "date", "total_orders", "total_revenue",
             "city", "day_name", "is_holiday"]
feature_cols = [c for c in train.columns if c not in drop_cols]

X_train = train[feature_cols].fillna(0)
y_train = train["total_orders"].values
X_test  = test[feature_cols].fillna(0)
y_test  = test["total_orders"].values

print(f"Train: {len(train):,} rows  |  Test: {len(test):,} rows")
print(f"Features: {len(feature_cols)}")""")

code("""# Train quantile models for confidence intervals
base_params = dict(n_estimators=200, learning_rate=0.1, num_leaves=50,
                   max_depth=-1, random_state=42, verbose=-1)

# Point prediction (median / q50)
model_point = lgb.LGBMRegressor(**base_params, objective="quantile", alpha=0.5)
model_point.fit(X_train, y_train)

# Lower bound (q10)
model_lower = lgb.LGBMRegressor(**base_params, objective="quantile", alpha=0.1)
model_lower.fit(X_train, y_train)

# Upper bound (q90)
model_upper = lgb.LGBMRegressor(**base_params, objective="quantile", alpha=0.9)
model_upper.fit(X_train, y_train)

# Generate predictions on test set
test["prediction"]  = model_point.predict(X_test)
test["lower_bound"] = model_lower.predict(X_test)
test["upper_bound"] = model_upper.predict(X_test)

# Ensure bounds are sensible (lower < point < upper)
test["lower_bound"] = np.minimum(test["lower_bound"], test["prediction"])
test["upper_bound"] = np.maximum(test["upper_bound"], test["prediction"])

print("✅ Models trained & predictions generated")
print(f"Coverage: {validate_interval_coverage(y_test, test['lower_bound'].values, test['upper_bound'].values):.1%}")""")

# ── Cell 4: Compute Risk ───────────────────────────────────────
md("## 3 · Compute Risk Scores & Classifications\n\n"
   "Using the **Risk Scoring Engine** — `risk_score = interval_width / (point_pred + 1)`")

code("""# Apply risk scoring engine
test["risk_score"] = calculate_risk_score(
    test["prediction"].values,
    test["lower_bound"].values,
    test["upper_bound"].values
)

test["risk_level"] = test["risk_score"].apply(classify_risk)

print("Risk Score Statistics:")
print(test["risk_score"].describe().round(4))
print(f"\\nRisk Level Distribution:")
print(test["risk_level"].value_counts())""")

# ── Cell 5: Overall Risk Distribution ─────────────────────────
md("## 4 · Risk Distribution Analysis")
md("### 4.1 Overall Risk Score Distribution")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of risk scores
axes[0].hist(test["risk_score"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].axvline(0.2, color="green", ls="--", lw=2, label="Low/Medium boundary (0.2)")
axes[0].axvline(0.5, color="red",   ls="--", lw=2, label="Medium/High boundary (0.5)")
axes[0].set_xlabel("Risk Score")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Risk Score Distribution")
axes[0].legend(fontsize=9)

# Pie chart of risk levels
level_counts = test["risk_level"].value_counts()
colors_map = {"Low Risk": "#55a868", "Medium Risk": "#f0ad4e", "High Risk": "#c44e52"}
colors = [colors_map.get(l, "#999") for l in level_counts.index]
axes[1].pie(level_counts, labels=level_counts.index, autopct="%1.1f%%",
            colors=colors, startangle=140, textprops={"fontsize": 11})
axes[1].set_title("Risk Level Proportions")

plt.tight_layout()
plt.savefig("../data/processed/risk_distribution_overall.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Saved: risk_distribution_overall.png")""")

# ── Cell 6: Risk by Restaurant ────────────────────────────────
md("### 4.2 Risk Distribution Across Restaurants")

code("""# Average risk per restaurant
restaurant_risk = (
    test.groupby("restaurant_id")["risk_score"]
    .agg(["mean", "median", "std", "count"])
    .rename(columns={"mean": "avg_risk", "median": "med_risk", "std": "risk_std", "count": "n_obs"})
    .sort_values("avg_risk", ascending=False)
    .reset_index()
)

print(f"Restaurants in test set: {len(restaurant_risk)}")
print("\\n— Top 10 Riskiest Restaurants —")
print(restaurant_risk.head(10).to_string(index=False))
print("\\n— Top 10 Safest Restaurants —")
print(restaurant_risk.tail(10).to_string(index=False))""")

code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top/Bottom 15 restaurants by avg risk
top15 = restaurant_risk.head(15)
bot15 = restaurant_risk.tail(15)

axes[0].barh(top15["restaurant_id"].astype(str), top15["avg_risk"], color="#c44e52", alpha=0.85)
axes[0].set_xlabel("Average Risk Score")
axes[0].set_title("Top 15 Riskiest Restaurants")
axes[0].invert_yaxis()

axes[1].barh(bot15["restaurant_id"].astype(str), bot15["avg_risk"], color="#55a868", alpha=0.85)
axes[1].set_xlabel("Average Risk Score")
axes[1].set_title("Top 15 Safest Restaurants")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("../data/processed/risk_by_restaurant.png", dpi=150, bbox_inches="tight")
plt.show()""")

code("""# Risk level proportions per restaurant (heatmap of top 30 restaurants)
top30_ids = restaurant_risk.head(30)["restaurant_id"].values
top30_data = test[test["restaurant_id"].isin(top30_ids)]

risk_pivot = pd.crosstab(
    top30_data["restaurant_id"], top30_data["risk_level"], normalize="index"
) * 100

# Reorder columns
for col in ["Low Risk", "Medium Risk", "High Risk"]:
    if col not in risk_pivot.columns:
        risk_pivot[col] = 0
risk_pivot = risk_pivot[["Low Risk", "Medium Risk", "High Risk"]]

fig, ax = plt.subplots(figsize=(10, 12))
risk_pivot.plot(kind="barh", stacked=True, ax=ax,
                color=["#55a868", "#f0ad4e", "#c44e52"], alpha=0.85)
ax.set_xlabel("Percentage (%)")
ax.set_title("Risk Level Breakdown — Top 30 Riskiest Restaurants")
ax.legend(title="Risk Level", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.show()""")

# ── Cell 7: Risk by Day of Week ────────────────────────────────
md("### 4.3 Risk Distribution by Day of Week")

code("""day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
test["day_name"] = pd.Categorical(test["day_name"], categories=day_order, ordered=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Box plot
sns.boxplot(x="day_name", y="risk_score", data=test, ax=axes[0],
            palette="coolwarm", order=day_order)
axes[0].set_xlabel("Day of Week")
axes[0].set_ylabel("Risk Score")
axes[0].set_title("Risk Score Distribution by Day of Week")
axes[0].tick_params(axis="x", rotation=45)

# Mean risk + risk level proportions
day_risk = test.groupby("day_name", observed=False)["risk_score"].agg(["mean", "median", "std"]).loc[day_order]
axes[1].bar(day_risk.index, day_risk["mean"], yerr=day_risk["std"], capsize=4,
            color="#4C72B0", alpha=0.8, edgecolor="white")
axes[1].set_xlabel("Day of Week")
axes[1].set_ylabel("Mean Risk Score (± std)")
axes[1].set_title("Average Risk Score by Day of Week")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("../data/processed/risk_by_day_of_week.png", dpi=150, bbox_inches="tight")
plt.show()""")

code("""# Risk level distribution per day of week
day_risk_levels = pd.crosstab(test["day_name"], test["risk_level"], normalize="index") * 100
for col in ["Low Risk", "Medium Risk", "High Risk"]:
    if col not in day_risk_levels.columns:
        day_risk_levels[col] = 0
day_risk_levels = day_risk_levels.loc[day_order, ["Low Risk", "Medium Risk", "High Risk"]]

print("Risk Level Distribution by Day of Week (%):")
print(day_risk_levels.round(1).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
day_risk_levels.plot(kind="bar", stacked=True, ax=ax,
                     color=["#55a868", "#f0ad4e", "#c44e52"], alpha=0.85)
ax.set_xlabel("Day of Week")
ax.set_ylabel("Percentage (%)")
ax.set_title("Risk Level Proportions by Day of Week")
ax.legend(title="Risk Level")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()""")

# ── Cell 8: Holidays vs Regular Days ──────────────────────────
md("### 4.4 Holidays vs Regular Days")

code("""test["day_type"] = test["is_holiday"].map({1: "Holiday", 0: "Regular Day"})

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Box plot comparison
sns.boxplot(x="day_type", y="risk_score", data=test, ax=axes[0],
            palette={"Holiday": "#c44e52", "Regular Day": "#4C72B0"})
axes[0].set_title("Risk Score: Holiday vs Regular Day")
axes[0].set_xlabel("")
axes[0].set_ylabel("Risk Score")

# Violin plot for deeper distribution view
sns.violinplot(x="day_type", y="risk_score", data=test, ax=axes[1],
               palette={"Holiday": "#c44e52", "Regular Day": "#4C72B0"}, inner="quartile")
axes[1].set_title("Risk Distribution (Violin)")
axes[1].set_xlabel("")
axes[1].set_ylabel("Risk Score")

# Risk level stacked bar
holiday_risk = pd.crosstab(test["day_type"], test["risk_level"], normalize="index") * 100
for col in ["Low Risk", "Medium Risk", "High Risk"]:
    if col not in holiday_risk.columns:
        holiday_risk[col] = 0
holiday_risk = holiday_risk[["Low Risk", "Medium Risk", "High Risk"]]
holiday_risk.plot(kind="bar", stacked=True, ax=axes[2],
                  color=["#55a868", "#f0ad4e", "#c44e52"], alpha=0.85)
axes[2].set_xlabel("")
axes[2].set_ylabel("Percentage (%)")
axes[2].set_title("Risk Level Proportions")
axes[2].legend(title="Risk Level")
axes[2].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("../data/processed/risk_holiday_vs_regular.png", dpi=150, bbox_inches="tight")
plt.show()""")

code("""# Statistical summary
print("=" * 60)
print("HOLIDAY vs REGULAR DAY — Risk Summary")
print("=" * 60)

for dtype in ["Holiday", "Regular Day"]:
    subset = test[test["day_type"] == dtype]
    print(f"\\n  {dtype} ({len(subset):,} observations):")
    print(f"    Mean risk score : {subset['risk_score'].mean():.4f}")
    print(f"    Median risk     : {subset['risk_score'].median():.4f}")
    print(f"    Std dev         : {subset['risk_score'].std():.4f}")
    for level in ["Low Risk", "Medium Risk", "High Risk"]:
        pct = (subset["risk_level"] == level).mean() * 100
        print(f"    {level:12s}    : {pct:5.1f}%")

# Statistical test
from scipy import stats
holiday_scores = test[test["is_holiday"] == 1]["risk_score"]
regular_scores = test[test["is_holiday"] == 0]["risk_score"]

if len(holiday_scores) > 0 and len(regular_scores) > 0:
    stat, pval = stats.mannwhitneyu(holiday_scores, regular_scores, alternative="two-sided")
    print(f"\\n  Mann-Whitney U test: U={stat:.0f}, p-value={pval:.4e}")
    print(f"  {'⚠️ Significant difference' if pval < 0.05 else '✅ No significant difference'} (α=0.05)")""")

# ── Cell 9: Validate Risk Classifications ──────────────────────
md("## 5 · Validate Risk Classifications\n\n"
   "We validate that the risk engine classifications are **meaningful** by checking:\n"
   "1. Higher risk → larger prediction errors (MAE)\n"
   "2. Higher risk → wider confidence intervals\n"
   "3. Coverage varies logically across risk levels\n"
   "4. Monotonicity of risk thresholds")

code("""# Validation 1: Error by risk level — high risk should have higher forecast error
test["abs_error"] = np.abs(test["total_orders"] - test["prediction"])
test["interval_width"] = test["upper_bound"] - test["lower_bound"]
test["within_interval"] = ((test["total_orders"] >= test["lower_bound"]) &
                           (test["total_orders"] <= test["upper_bound"])).astype(int)

print("=" * 70)
print("RISK CLASSIFICATION VALIDATION")
print("=" * 70)

validation_summary = test.groupby("risk_level").agg(
    count=("risk_score", "size"),
    mean_risk_score=("risk_score", "mean"),
    mean_abs_error=("abs_error", "mean"),
    median_abs_error=("abs_error", "median"),
    mean_interval_width=("interval_width", "mean"),
    coverage=("within_interval", "mean"),
    mean_prediction=("prediction", "mean"),
    mean_actual=("total_orders", "mean"),
).round(4)

# Reorder
for level in ["Low Risk", "Medium Risk", "High Risk"]:
    if level in validation_summary.index:
        pass
validation_summary = validation_summary.reindex(["Low Risk", "Medium Risk", "High Risk"]).dropna()

print("\\n📋 Validation Summary by Risk Level:")
print(validation_summary.to_string())""")

code("""# Validation 2: Visual — MAE and interval width should increase with risk
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

risk_order = ["Low Risk", "Medium Risk", "High Risk"]
colors = ["#55a868", "#f0ad4e", "#c44e52"]

# 2a. MAE by risk level
mae_data = [test[test["risk_level"] == r]["abs_error"] for r in risk_order if r in test["risk_level"].values]
risk_labels = [r for r in risk_order if r in test["risk_level"].values]
bp1 = axes[0, 0].boxplot(mae_data, labels=risk_labels, patch_artist=True, showfliers=False)
for patch, color in zip(bp1["boxes"], colors[:len(mae_data)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0, 0].set_ylabel("Absolute Error")
axes[0, 0].set_title("✅ Check 1: Higher Risk → Larger Forecast Error?")

# 2b. Interval width by risk level
iw_data = [test[test["risk_level"] == r]["interval_width"] for r in risk_order if r in test["risk_level"].values]
bp2 = axes[0, 1].boxplot(iw_data, labels=risk_labels, patch_artist=True, showfliers=False)
for patch, color in zip(bp2["boxes"], colors[:len(iw_data)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0, 1].set_ylabel("Interval Width")
axes[0, 1].set_title("✅ Check 2: Higher Risk → Wider Intervals?")

# 2c. Coverage by risk level
coverage_by_level = test.groupby("risk_level")["within_interval"].mean().reindex(risk_order).dropna()
axes[1, 0].bar(coverage_by_level.index, coverage_by_level.values,
               color=colors[:len(coverage_by_level)], alpha=0.8, edgecolor="white")
axes[1, 0].axhline(0.8, color="gray", ls="--", label="80% target")
axes[1, 0].set_ylabel("Coverage Rate")
axes[1, 0].set_title("✅ Check 3: Confidence Interval Coverage by Risk Level")
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 1.05)

# 2d. Risk score vs absolute error scatter (sampled)
sample = test.sample(min(5000, len(test)), random_state=42)
axes[1, 1].scatter(sample["risk_score"], sample["abs_error"],
                   alpha=0.15, s=8, c="#4C72B0")
axes[1, 1].set_xlabel("Risk Score")
axes[1, 1].set_ylabel("Absolute Error")
axes[1, 1].set_title("Risk Score vs Forecast Error (correlation)")

# Correlation
corr = test["risk_score"].corr(test["abs_error"])
axes[1, 1].annotate(f"Pearson r = {corr:.3f}", xy=(0.05, 0.95),
                    xycoords="axes fraction", fontsize=12, fontweight="bold",
                    bbox=dict(boxstyle="round", fc="lightyellow"))

plt.tight_layout()
plt.savefig("../data/processed/risk_validation.png", dpi=150, bbox_inches="tight")
plt.show()""")

code("""# Validation 3: Monotonicity checks
print("=" * 70)
print("MONOTONICITY VALIDATION")
print("=" * 70)

checks_passed = 0
total_checks = 0

# Check: MAE increases with risk level
mae_by_level = validation_summary["mean_abs_error"]
if len(mae_by_level) >= 2:
    total_checks += 1
    if mae_by_level.is_monotonic_increasing:
        print("✅ PASS: Mean Absolute Error increases Low → Medium → High")
        checks_passed += 1
    else:
        print("⚠️ PARTIAL: MAE does not strictly increase across risk levels")
        print(f"   Values: {dict(mae_by_level)}")

# Check: Interval width increases with risk level
iw_by_level = validation_summary["mean_interval_width"]
if len(iw_by_level) >= 2:
    total_checks += 1
    if iw_by_level.is_monotonic_increasing:
        print("✅ PASS: Interval width increases Low → Medium → High")
        checks_passed += 1
    else:
        print("⚠️ PARTIAL: Interval width does not strictly increase across risk levels")
        print(f"   Values: {dict(iw_by_level)}")

# Check: Risk score positively correlates with error
total_checks += 1
corr = test["risk_score"].corr(test["abs_error"])
if corr > 0:
    print(f"✅ PASS: Risk score positively correlates with error (r={corr:.3f})")
    checks_passed += 1
else:
    print(f"⚠️ FAIL: Risk score negatively correlates with error (r={corr:.3f})")

# Check: Boundary thresholds are correct
total_checks += 1
low_max  = test[test["risk_level"] == "Low Risk"]["risk_score"].max() if "Low Risk" in test["risk_level"].values else 0
med_min  = test[test["risk_level"] == "Medium Risk"]["risk_score"].min() if "Medium Risk" in test["risk_level"].values else 0
med_max  = test[test["risk_level"] == "Medium Risk"]["risk_score"].max() if "Medium Risk" in test["risk_level"].values else 0
high_min = test[test["risk_level"] == "High Risk"]["risk_score"].min() if "High Risk" in test["risk_level"].values else 0

boundaries_ok = low_max < 0.2 and med_min >= 0.2 and med_max < 0.5 and high_min >= 0.5
if boundaries_ok:
    print("✅ PASS: Risk boundaries correctly applied (Low<0.2, Med 0.2-0.5, High≥0.5)")
    checks_passed += 1
else:
    print(f"⚠️ FAIL: Boundary check — Low max={low_max:.4f}, Med=[{med_min:.4f},{med_max:.4f}], High min={high_min:.4f}")

print(f"\\n📊 Validation Result: {checks_passed}/{total_checks} checks passed")""")

# ── Cell 10: Summary ──────────────────────────────────────────
md("## 6 · Summary & Conclusions")

code("""print("=" * 70)
print("RISK ANALYSIS SUMMARY")
print("=" * 70)

total = len(test)
print(f"\\n  Total test observations : {total:,}")
print(f"  Unique restaurants      : {test['restaurant_id'].nunique()}")
print(f"  Date range (test)       : {test['date'].min().date()} → {test['date'].max().date()}")

print(f"\\n  Risk Score Statistics:")
print(f"    Mean   : {test['risk_score'].mean():.4f}")
print(f"    Median : {test['risk_score'].median():.4f}")
print(f"    Std    : {test['risk_score'].std():.4f}")
print(f"    Min    : {test['risk_score'].min():.4f}")
print(f"    Max    : {test['risk_score'].max():.4f}")

print(f"\\n  Risk Level Distribution:")
for level in ["Low Risk", "Medium Risk", "High Risk"]:
    n = (test["risk_level"] == level).sum()
    pct = n / total * 100
    print(f"    {level:12s}: {n:>7,} ({pct:5.1f}%)")

print(f"\\n  Overall Coverage (80% CI) : {test['within_interval'].mean():.1%}")
print(f"  Risk-Error Correlation    : {test['risk_score'].corr(test['abs_error']):.3f}")
print("\\n" + "=" * 70)
print("✅ Risk Engine validation complete")""")

# ── Build Notebook ─────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = pathlib.Path("../data/processed")
out.mkdir(parents=True, exist_ok=True)

nb_path = pathlib.Path("10_risk_analysis.ipynb")
nb_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"✅ Notebook written to {nb_path} ({len(cells)} cells)")
