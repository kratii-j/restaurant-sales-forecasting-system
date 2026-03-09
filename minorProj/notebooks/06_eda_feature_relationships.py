"""
Step 12: EDA — Feature Relationships
=====================================
Explores inter-feature correlations, feature-target associations,
and cross-dataset relationship patterns for the restaurant sales
forecasting system.

Outputs: all figures saved to notebooks/figures/step12/
"""

import warnings, os, sys
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# ── paths ────────────────────────────────────────────────────
BASE = Path("/Users/akshatsoni/Desktop/restaurant-sales-forecasting-system/minorProj")
RAW  = BASE / "data" / "raw"
FIG  = BASE / "notebooks" / "figures" / "step12"
FIG.mkdir(parents=True, exist_ok=True)

# ── style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
    "axes.titleweight": "bold",
})

# ══════════════════════════════════════════════════════════════
# 1.  Load datasets
# ══════════════════════════════════════════════════════════════
print("Loading datasets...")
daily    = pd.read_csv(RAW / "daily_orders.csv")
external = pd.read_csv(RAW / "external_context.csv")
resto    = pd.read_csv(RAW / "restaurant_dataset.csv", encoding="utf-8-sig")
featured = pd.read_csv(RAW / "featured_orders.csv")

# normalise col names
daily.columns    = daily.columns.str.lower().str.strip().str.replace(" ", "_")
external.columns = external.columns.str.lower().str.strip().str.replace(" ", "_")
resto.columns    = resto.columns.str.lower().str.strip().str.replace(" ", "_")
featured.columns = featured.columns.str.lower().str.strip().str.replace(" ", "_")
# strip any remaining BOM chars
resto.columns = resto.columns.str.replace('\ufeff', '')

daily["date"]    = pd.to_datetime(daily["date"])
external["date"] = pd.to_datetime(external["date"])
featured["date"] = pd.to_datetime(featured["date"])

print(f"  daily_orders : {daily.shape}")
print(f"  external_ctx : {external.shape}")
print(f"  restaurants  : {resto.shape}")
print(f"  featured     : {featured.shape}")

# ══════════════════════════════════════════════════════════════
# 2.  CORRELATION HEATMAP  — numeric features in daily_orders
# ══════════════════════════════════════════════════════════════
print("\n[1/10] Correlation heatmap — daily_orders numeric features")
num_cols_daily = daily.select_dtypes(include="number").columns.tolist()
corr_daily = daily[num_cols_daily].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_daily, dtype=bool), k=1)
sns.heatmap(corr_daily, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, linewidths=0.5,
            vmin=-1, vmax=1, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"})
ax.set_title("Correlation Matrix — Daily Orders (Numeric Features)")
fig.savefig(FIG / "01_daily_orders_correlation_heatmap.png")
plt.close(fig)
print("  ✓ saved 01_daily_orders_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════
# 3.  CORRELATION HEATMAP  — featured_orders (including lags)
# ══════════════════════════════════════════════════════════════
print("\n[2/10] Correlation heatmap — featured_orders (with engineered features)")
num_cols_feat = featured.select_dtypes(include="number").columns.tolist()
corr_feat = featured[num_cols_feat].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask2 = np.triu(np.ones_like(corr_feat, dtype=bool), k=1)
sns.heatmap(corr_feat, mask=mask2, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, linewidths=0.4,
            vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 8},
            cbar_kws={"shrink": 0.7, "label": "Pearson r"})
ax.set_title("Correlation Matrix — Featured Orders (All Numeric)")
fig.savefig(FIG / "02_featured_orders_correlation_heatmap.png")
plt.close(fig)
print("  ✓ saved 02_featured_orders_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════
# 4.  SCATTERPLOT MATRIX — key features vs target
# ══════════════════════════════════════════════════════════════
print("\n[3/10] Pairplot — key features vs total_orders")
# sample for speed
sample = daily.sample(n=min(5000, len(daily)), random_state=42).copy()
pair_cols = ["total_orders", "total_revenue", "avg_discount",
             "cancellation_rate", "avg_delivery_time"]
pair_cols = [c for c in pair_cols if c in sample.columns]

g = sns.pairplot(sample[pair_cols], diag_kind="kde",
                 plot_kws={"alpha": 0.35, "s": 12},
                 diag_kws={"fill": True})
g.figure.suptitle("Pairplot — Key Operating Metrics", y=1.02, fontweight="bold")
g.savefig(FIG / "03_key_features_pairplot.png")
plt.close(g.figure)
print("  ✓ saved 03_key_features_pairplot.png")

# ══════════════════════════════════════════════════════════════
# 5.  ORDERS vs REVENUE — colored by promotion_flag
# ══════════════════════════════════════════════════════════════
print("\n[4/10] Orders vs Revenue — promotion effect")
sample["promo"] = sample["promotion_flag"].map({True: "Promotion", False: "No Promotion",
                                                  "True": "Promotion", "False": "No Promotion"})
fig, ax = plt.subplots(figsize=(11, 7))
for label, color in [("No Promotion", "#4C72B0"), ("Promotion", "#DD8452")]:
    sub = sample[sample["promo"] == label]
    ax.scatter(sub["total_orders"], sub["total_revenue"],
               alpha=0.35, s=18, label=label, color=color)
ax.set_xlabel("Total Orders")
ax.set_ylabel("Total Revenue")
ax.set_title("Orders vs Revenue — Promotion Effect")
ax.legend(title="Promotion")
fig.savefig(FIG / "04_orders_vs_revenue_promo.png")
plt.close(fig)
print("  ✓ saved 04_orders_vs_revenue_promo.png")

# ══════════════════════════════════════════════════════════════
# 6.  DISCOUNT → ORDERS  (binned + box)
# ══════════════════════════════════════════════════════════════
print("\n[5/10] Discount impact on orders (binned)")
daily_tmp = daily.copy()
daily_tmp["discount_bin"] = pd.cut(daily_tmp["avg_discount"],
                                    bins=[-0.01, 0, 5, 10, 20, 100],
                                    labels=["0%", "0–5%", "5–10%", "10–20%", "20%+"])
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=daily_tmp, x="discount_bin", y="total_orders",
            palette="viridis", ax=ax, showfliers=False)
ax.set_title("Total Orders by Discount Bracket")
ax.set_xlabel("Average Discount (%)")
ax.set_ylabel("Total Orders")
fig.savefig(FIG / "05_discount_vs_orders_boxplot.png")
plt.close(fig)
print("  ✓ saved 05_discount_vs_orders_boxplot.png")

# ══════════════════════════════════════════════════════════════
# 7.  DELIVERY TIME → ORDERS (scatter + regression)
# ══════════════════════════════════════════════════════════════
print("\n[6/10] Delivery time vs orders (joint / regression)")
sample_clean = sample.dropna(subset=["avg_delivery_time", "total_orders"])
g = sns.jointplot(data=sample_clean, x="avg_delivery_time", y="total_orders",
                  kind="reg", height=7,
                  scatter_kws={"alpha": 0.25, "s": 12},
                  line_kws={"color": "crimson"})
g.figure.suptitle("Delivery Time vs Total Orders", y=1.02, fontweight="bold")
g.savefig(FIG / "06_delivery_time_vs_orders.png")
plt.close(g.figure)
print("  ✓ saved 06_delivery_time_vs_orders.png")

# ══════════════════════════════════════════════════════════════
# 8.  CANCELLATION RATE → ORDERS
# ══════════════════════════════════════════════════════════════
print("\n[7/10] Cancellation rate vs orders")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# scatter
axes[0].scatter(sample_clean["cancellation_rate"],
                sample_clean["total_orders"],
                alpha=0.25, s=12, color="#6A5ACD")
axes[0].set_xlabel("Cancellation Rate")
axes[0].set_ylabel("Total Orders")
axes[0].set_title("Scatter: Cancellation Rate vs Orders")

# binned bar
daily_tmp["cancel_bin"] = pd.cut(daily_tmp["cancellation_rate"],
                                  bins=[-0.01, 0.02, 0.05, 0.10, 1.0],
                                  labels=["<2%", "2-5%", "5-10%", "10%+"])
cancel_agg = daily_tmp.groupby("cancel_bin", observed=True)["total_orders"].mean()
cancel_agg.plot.bar(ax=axes[1], color=sns.color_palette("mako", n_colors=4),
                    edgecolor="white")
axes[1].set_title("Avg Orders by Cancellation Rate Bracket")
axes[1].set_xlabel("Cancellation Rate")
axes[1].set_ylabel("Mean Total Orders")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)

fig.tight_layout()
fig.savefig(FIG / "07_cancellation_vs_orders.png")
plt.close(fig)
print("  ✓ saved 07_cancellation_vs_orders.png")

# ══════════════════════════════════════════════════════════════
# 9.  WEATHER  → ORDERS (external context merged)
# ══════════════════════════════════════════════════════════════
print("\n[8/10] Weather impact on orders (merged dataset)")

# We need to merge daily orders with restaurant city info, then with external context
# Join restaurant → daily on restaurant_id, then external on date+city
rest_city = resto[["restaurant_id", "city"]].copy()
rest_city.columns = rest_city.columns.str.lower().str.strip().str.replace(" ", "_")

daily_city = daily.merge(rest_city, on="restaurant_id", how="left")
daily_ext = daily_city.merge(external, on=["date", "city"], how="inner")

print(f"  Merged daily+external: {daily_ext.shape[0]:,} rows")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Weather boxplot
weather_order = daily_ext.groupby("weather")["total_orders"].median().sort_values().index
sns.boxplot(data=daily_ext, x="weather", y="total_orders",
            order=weather_order, palette="coolwarm",
            showfliers=False, ax=axes[0])
axes[0].set_title("Total Orders by Weather Condition")
axes[0].set_xlabel("Weather")
axes[0].set_ylabel("Total Orders")

# Holiday / Event impact
cats = []
vals = []
for flag_col, label_yes, label_no in [
    ("is_holiday", "Holiday", "Non-Holiday"),
    ("event_flag", "Event Day", "No Event"),
]:
    if flag_col in daily_ext.columns:
        for flag_val, label in [(True, label_yes), (False, label_no),
                                 ("True", label_yes), ("False", label_no)]:
            sub = daily_ext[daily_ext[flag_col] == flag_val]
            if len(sub) > 0:
                cats.append(label)
                vals.append(sub["total_orders"].mean())

# deduplicate
seen = set()
cats_dedup = []
vals_dedup = []
for c, v in zip(cats, vals):
    if c not in seen:
        cats_dedup.append(c)
        vals_dedup.append(v)
        seen.add(c)

axes[1].barh(cats_dedup, vals_dedup,
             color=sns.color_palette("Set2", n_colors=len(cats_dedup)),
             edgecolor="white", height=0.5)
axes[1].set_xlabel("Mean Total Orders")
axes[1].set_title("Holiday / Event Impact on Orders")
for i, v in enumerate(vals_dedup):
    axes[1].text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=10)

fig.tight_layout()
fig.savefig(FIG / "08_weather_holiday_event_impact.png")
plt.close(fig)
print("  ✓ saved 08_weather_holiday_event_impact.png")

# ══════════════════════════════════════════════════════════════
# 10. RESTAURANT ATTRIBUTES → AVG DAILY ORDERS
# ══════════════════════════════════════════════════════════════
print("\n[9/10] Restaurant attributes vs average daily orders")

avg_orders = daily.groupby("restaurant_id")["total_orders"].mean().reset_index()
avg_orders.columns = ["restaurant_id", "avg_daily_orders"]

resto_merged = resto.merge(avg_orders, on="restaurant_id", how="inner")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Price range
sns.boxplot(data=resto_merged, x="price_range", y="avg_daily_orders",
            palette="magma", showfliers=False, ax=axes[0, 0])
axes[0, 0].set_title("Avg Daily Orders by Price Range")
axes[0, 0].set_xlabel("Price Range (1=Budget → 4=Premium)")

# Aggregate rating
axes[0, 1].scatter(resto_merged["aggregate_rating"],
                   resto_merged["avg_daily_orders"],
                   alpha=0.3, s=15, color="#E07A5F")
# lowess-ish via binned means
rating_bins = pd.cut(resto_merged["aggregate_rating"],
                     bins=np.arange(0, 5.5, 0.5))
rating_means = resto_merged.groupby(rating_bins, observed=True)["avg_daily_orders"].mean()
axes[0, 1].plot(rating_means.index.map(lambda x: x.mid),
               rating_means.values, color="navy", linewidth=2, marker="o")
axes[0, 1].set_title("Avg Daily Orders vs Aggregate Rating")
axes[0, 1].set_xlabel("Aggregate Rating")
axes[0, 1].set_ylabel("Avg Daily Orders")

# Votes
axes[1, 0].scatter(resto_merged["votes"],
                   resto_merged["avg_daily_orders"],
                   alpha=0.3, s=12, color="#81B29A")
axes[1, 0].set_title("Avg Daily Orders vs Votes")
axes[1, 0].set_xlabel("Restaurant Votes (Popularity)")
axes[1, 0].set_ylabel("Avg Daily Orders")

# Online delivery
for col_name, label in [("has_online_delivery", "Online Delivery"),
                         ("has_table_booking", "Table Booking")]:
    if col_name in resto_merged.columns:
        grp = resto_merged.groupby(col_name)["avg_daily_orders"].mean()
        axes[1, 1].bar([f"{label}: Yes", f"{label}: No"],
                      [grp.get("Yes", grp.get(True, 0)),
                       grp.get("No", grp.get(False, 0))],
                      alpha=0.7, edgecolor="white")
axes[1, 1].set_title("Avg Daily Orders — Delivery & Booking")
axes[1, 1].set_ylabel("Mean Daily Orders")

fig.suptitle("Restaurant Attributes vs Daily Order Volume",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG / "09_restaurant_attributes_vs_orders.png")
plt.close(fig)
print("  ✓ saved 09_restaurant_attributes_vs_orders.png")

# ══════════════════════════════════════════════════════════════
# 11. LAG / ROLLING FEATURES  → TARGET
# ══════════════════════════════════════════════════════════════
print("\n[10/10] Lag & rolling features vs target")
lag_cols = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7"]
lag_cols = [c for c in lag_cols if c in featured.columns]

# drop NaN rows for lag analysis
feat_lag = featured.dropna(subset=lag_cols + ["total_orders"])
# sample for plotting speed
feat_sample = feat_lag.sample(n=min(8000, len(feat_lag)), random_state=42)

n_plots = len(lag_cols)
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()

for i, col in enumerate(lag_cols):
    ax = axes[i]
    ax.scatter(feat_sample[col], feat_sample["total_orders"],
               alpha=0.2, s=8, color="#7B68EE")
    # Pearson r
    r, p = stats.pearsonr(feat_sample[col].dropna(),
                          feat_sample.loc[feat_sample[col].notna(), "total_orders"])
    ax.set_title(f"{col} vs total_orders\nr = {r:.3f}  (p = {p:.1e})")
    ax.set_xlabel(col)
    ax.set_ylabel("total_orders")

# hide unused axes
for j in range(len(lag_cols), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Lag / Rolling Features vs Total Orders",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG / "10_lag_rolling_vs_orders.png")
plt.close(fig)
print("  ✓ saved 10_lag_rolling_vs_orders.png")

# ══════════════════════════════════════════════════════════════
# 12.  SUMMARY STATISTICS — Top correlations with target
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FEATURE–TARGET CORRELATION SUMMARY")
print("=" * 60)

target = "total_orders"
# From featured_orders (includes lags)
corr_target = featured[num_cols_feat].corr()[target].drop(target).sort_values(key=abs, ascending=False)
print(f"\nTop correlations with '{target}' (featured_orders):")
print(corr_target.head(12).to_string())

# Spearman rank correlation for non-linear relationships
print("\nSpearman rank correlations with target:")
spearman_corrs = {}
for col in num_cols_feat:
    if col == target:
        continue
    clean = featured[[col, target]].dropna()
    if len(clean) > 100:
        rho, p = stats.spearmanr(clean[col], clean[target])
        spearman_corrs[col] = rho
spearman_series = pd.Series(spearman_corrs).sort_values(key=abs, ascending=False)
print(spearman_series.head(12).to_string())

print("\n" + "=" * 60)
print("WEATHER IMPACT SUMMARY")
print("=" * 60)
weather_stats = daily_ext.groupby("weather")["total_orders"].agg(["mean", "median", "std", "count"])
print(weather_stats.sort_values("mean", ascending=False).to_string())

print("\n" + "=" * 60)
print("PROMOTION IMPACT SUMMARY")
print("=" * 60)
promo_stats = daily.groupby("promotion_flag")["total_orders"].agg(["mean", "median", "std", "count"])
print(promo_stats.to_string())

print("\n✅ Step 12 complete! All 10 figures saved to:")
print(f"   {FIG}")
print(f"   Files: {sorted([f.name for f in FIG.glob('*.png')])}")
