"""
src/eda.py
Exploratory Data Analysis — generates 8 publication-quality charts
Run: python src/eda.py
Output: saves 8 PNG files to images/eda/ folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── Style settings (makes charts look professional) ───────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "grid.linestyle":   "--",
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
PALETTE = ["#4f46e5","#10b981","#f59e0b","#ef4444","#06b6d4"]

def load_data():
    df = pd.read_csv("data/processed/clean_sales.csv", parse_dates=["date"])
    print(f"Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════
# CHART 1 — Daily Revenue Trend (Line Chart)
# ══════════════════════════════════════════════════════════════
def chart1_sales_trend(df):
    daily = df.groupby("date")["revenue"].sum().reset_index()
    rolling = daily["revenue"].rolling(30).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["date"], daily["revenue"], color="#c7d2fe",
            linewidth=0.8, label="Daily Revenue", alpha=0.7)
    ax.plot(daily["date"], rolling, color="#4f46e5",
            linewidth=2.2, label="30-Day Rolling Average")
    ax.fill_between(daily["date"], daily["revenue"], alpha=0.08, color="#4f46e5")
    ax.set_title("Daily Revenue Trend — 2 Years", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1e6:.1f}M"))
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("images/eda/01_sales_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart 1 saved: 01_sales_trend.png")


# ══════════════════════════════════════════════════════════════
# CHART 2 — Category-wise Revenue (Horizontal Bar)
# ══════════════════════════════════════════════════════════════
def chart2_category_revenue(df):
    cat = df.groupby("category")["revenue"].sum().sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cat.index, cat.values,
                   color=PALETTE[:len(cat)], edgecolor="white", height=0.6)
    # Add value labels on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.01, bar.get_y() + bar.get_height()/2,
                f"₹{w/1e6:.1f}M", va="center", fontsize=10, color="#374151")
    ax.set_title("Revenue by Product Category", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Total Revenue (₹)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1e6:.0f}M"))
    plt.tight_layout()
    plt.savefig("images/eda/02_category_revenue.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart 2 saved: 02_category_revenue.png")


# ══════════════════════════════════════════════════════════════
# CHART 3 — Monthly Sales Heatmap (Seasonality)
# ══════════════════════════════════════════════════════════════
def chart3_seasonality_heatmap(df):
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")

    pivot = df.pivot_table(values="units_sold", index="year",
                           columns="month", aggfunc="sum")

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(pivot, annot=True, fmt=",", cmap="YlOrRd",
                linewidths=0.5, ax=ax,
                cbar_kws={"label":"Units Sold"},
                annot_kws={"size": 9})
    ax.set_title("Monthly Sales Heatmap — Seasonality Patterns",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticklabels(month_names, rotation=0)
    plt.tight_layout()
    plt.savefig("images/eda/03_seasonality_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart 3 saved: 03_seasonality_heatmap.png")


# ══════════════════════════════════════════════════════════════
# CHART 4 — Store Performance Comparison (Grouped Bar)
# ══════════════════════════════════════════════════════════════
def chart4_store_comparison(df):
    store_cat = df.groupby(["store_id","category"])["revenue"].sum().unstack()

    fig, ax = plt.subplots(figsize=(12, 6))
    store_cat.plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white",
                   width=0.75)
    ax.set_title("Store Performance by Category", fontsize=15,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Store")
    ax.set_ylabel("Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1e6:.1f}M"))
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig("images/eda/04_store_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart 4 saved: 04_store_comparison.png")


# ══════════════════════════════════════════════════════════════
# CHART 5 — Top 10 Products by Revenue (Bar Chart)
# ══════════════════════════════════════════════════════════════
def chart5_top_products(df):
    top10 = (df.groupby("product_id")["revenue"]
               .sum().sort_values(ascending=False).head(10))

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#4f46e5" if i == 0 else "#a5b4fc" for i in range(10)]
    ax.bar(range(len(top10)), top10.values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels(top10.index, rotation=30, ha="right", fontsize=9)
    ax.set_title("Top 10 Products by Revenue", fontsize=15,
                 fontweight="bold", pad=12)
    ax.set_ylabel("Total Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1e6:.1f}M"))
    for i, v in enumerate(top10.values):
        ax.text(i, v * 1.01, f"₹{v/1e6:.1f}M",
                ha="center", fontsize=8, color="#374151")
    plt.tight_layout()
    plt.savefig("images/eda/05_top_products.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart 5 saved: 05_top_products.png")


# ══════════════════════════════════════════════════════════════
# CHART 6 — Day-of-Week Sales Pattern (Box Plot)
# ══════════════════════════════════════════════════════════════
def chart6_weekday_pattern(df):
    dow_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    df["day_name"] = df["date"].dt.dayofweek.map(dow_names)
    order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df, x="day_name", y="units_sold",
                order=order, palette="Blues", ax=ax,
                fliersize=2, linewidth=0.8)
    ax.set_title("Units Sold by Day of Week — Weekend vs Weekday Pattern",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Units Sold")
    # Highlight weekends
    ax.axvspan(4.5, 6.5, alpha=0.1, color="#f59e0b", label="Weekend")
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/eda/06_weekday_pattern.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart 6 saved: 06_weekday_pattern.png")


# ══════════════════════════════════════════════════════════════
# CHART 7 — Revenue vs Units Sold Correlation (Scatter)
# ══════════════════════════════════════════════════════════════
def chart7_correlation(df):
    sample = df.sample(min(5000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(sample["units_sold"], sample["revenue"],
                         c=pd.Categorical(sample["category"]).codes,
                         cmap="tab10", alpha=0.4, s=15, edgecolors="none")
    # Add trend line
    z = np.polyfit(sample["units_sold"], sample["revenue"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample["units_sold"].min(),
                          sample["units_sold"].max(), 200)
    ax.plot(x_line, p(x_line), color="#ef4444", linewidth=2,
            linestyle="--", label="Trend")
    ax.set_title("Units Sold vs Revenue — Correlation by Category",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Units Sold")
    ax.set_ylabel("Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1e3:.0f}K"))
    categories = df["category"].unique()
    handles = [plt.scatter([],[], c=f"C{i}", label=c, s=30)
               for i, c in enumerate(categories)]
    ax.legend(handles=handles, title="Category", fontsize=9)
    plt.tight_layout()
    plt.savefig("images/eda/07_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart 7 saved: 07_correlation.png")


# ══════════════════════════════════════════════════════════════
# CHART 8 — Monthly Revenue Growth (with % change labels)
# ══════════════════════════════════════════════════════════════
def chart8_monthly_growth(df):
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("year_month")["revenue"].sum()
    pct_change = monthly.pct_change() * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios":[2,1]})
    # Top: Revenue bars
    colors = ["#4f46e5" if v >= 0 else "#ef4444"
              for v in pct_change.fillna(0)]
    ax1.bar(range(len(monthly)), monthly.values,
            color="#c7d2fe", edgecolor="white")
    ax1.plot(range(len(monthly)), monthly.rolling(3).mean().values,
             color="#4f46e5", linewidth=2, label="3-month avg")
    ax1.set_title("Monthly Revenue & Month-over-Month Growth",
                  fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Revenue (₹)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1e6:.1f}M"))
    ax1.legend()

    # Bottom: % change
    pct_vals = pct_change.fillna(0).values
    bar_colors = ["#10b981" if v >= 0 else "#ef4444" for v in pct_vals]
    ax2.bar(range(len(pct_vals)), pct_vals, color=bar_colors, edgecolor="white")
    ax2.axhline(0, color="#374151", linewidth=0.8)
    ax2.set_ylabel("MoM Change (%)")
    ax2.set_xlabel("Month")
    periods = [str(p) for p in monthly.index]
    ax2.set_xticks(range(0, len(periods), 3))
    ax2.set_xticklabels(periods[::3], rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig("images/eda/08_monthly_growth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart 8 saved: 08_monthly_growth.png")


# ══════════════════════════════════════════════════════════════
# MAIN — run all charts
# ══════════════════════════════════════════════════════════════
def run_eda():
    print("\n" + "="*50)
    print("STARTING EDA — Generating 8 charts")
    print("="*50 + "\n")
    df = load_data()
    chart1_sales_trend(df)
    chart2_category_revenue(df)
    chart3_seasonality_heatmap(df)
    chart4_store_comparison(df)
    chart5_top_products(df)
    chart6_weekday_pattern(df)
    chart7_correlation(df)
    chart8_monthly_growth(df)
    print("\n" + "="*50)
    print("EDA COMPLETE — 8 charts saved to images/eda/")
    print("="*50)

if __name__ == "__main__":
    run_eda()