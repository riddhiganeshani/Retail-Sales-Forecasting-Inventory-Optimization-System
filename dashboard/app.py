import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")

# ─── TITLE ─────────────────────────
st.markdown("<h1 style='color:#4f46e5;'>📊 Retail Intelligence Dashboard</h1>", unsafe_allow_html=True)

# ─── LOAD DATA ─────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/clean_sales.csv", parse_dates=["date"])

df = load_data()

# ─── SIDEBAR FILTERS ─────────────────────────
st.sidebar.header("🔍 Filters")

store = st.sidebar.multiselect("Store", df["store_id"].unique(), default=df["store_id"].unique())
product = st.sidebar.multiselect("Product", df["product_id"].unique(), default=df["product_id"].unique())

date_range = st.sidebar.date_input("Date Range",
                                  [df["date"].min(), df["date"].max()])

filtered_df = df[
    (df["store_id"].isin(store)) &
    (df["product_id"].isin(product)) &
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1]))
]

# ─── KPI SECTION ─────────────────────────
st.subheader("📌 Business KPIs")

col1, col2, col3 = st.columns(3)
col1.metric("💰 Revenue", f"₹{filtered_df['revenue'].sum():,.0f}")
col2.metric("📦 Units Sold", f"{filtered_df['units_sold'].sum():,.0f}")
col3.metric("📊 Avg Daily", f"{filtered_df.groupby('date')['revenue'].sum().mean():,.0f}")

# ─── SALES TREND ─────────────────────────
st.subheader("📈 Sales Trend")

daily = filtered_df.groupby("date")["revenue"].sum().reset_index()
rolling = daily["revenue"].rolling(30).mean()

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(daily["date"], daily["revenue"], color="#c7d2fe")
ax.plot(daily["date"], rolling, color="#4f46e5", linewidth=2)
ax.fill_between(daily["date"], daily["revenue"], alpha=0.1)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
st.pyplot(fig)

# ─── CATEGORY + HEATMAP ─────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Category Revenue")
    cat = filtered_df.groupby("category")["revenue"].sum()
    fig, ax = plt.subplots()
    cat.plot(kind="barh", color="#4f46e5", ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("🔥 Seasonality")
    filtered_df["month"] = filtered_df["date"].dt.month
    filtered_df["year"] = filtered_df["date"].dt.year
    pivot = filtered_df.pivot_table(values="units_sold", index="year", columns="month")
    fig, ax = plt.subplots()
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
    st.pyplot(fig)

# ─── STORE + TOP PRODUCTS ─────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏬 Store Comparison")
    store_cat = filtered_df.groupby(["store_id","category"])["revenue"].sum().unstack()
    fig, ax = plt.subplots()
    store_cat.plot(kind="bar", ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("🏆 Top Products")
    top10 = filtered_df.groupby("product_id")["revenue"].sum().nlargest(10)
    fig, ax = plt.subplots()
    top10.plot(kind="bar", color="#10b981", ax=ax)
    st.pyplot(fig)

# ─── WEEKDAY + CORRELATION ─────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Weekday Pattern")
    filtered_df["day"] = filtered_df["date"].dt.day_name()
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x="day", y="units_sold", ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("🔗 Correlation")
    fig, ax = plt.subplots()
    ax.scatter(filtered_df["units_sold"], filtered_df["revenue"], alpha=0.3)
    st.pyplot(fig)

# ─── MONTHLY GROWTH ─────────────────────────
st.subheader("📉 Monthly Growth")

filtered_df["year_month"] = filtered_df["date"].dt.to_period("M")
monthly = filtered_df.groupby("year_month")["revenue"].sum()

fig, ax = plt.subplots()
monthly.plot(color="#ef4444", ax=ax)
st.pyplot(fig)

# ─── ML FORECAST ─────────────────────────
st.subheader("🔮 ML Forecasting")

forecast_days = st.slider("Forecast Days", 7, 60, 30)

# Feature Engineering
daily_sales = daily.copy()
daily_sales["day"] = daily_sales["date"].dt.day
daily_sales["month"] = daily_sales["date"].dt.month
daily_sales["weekday"] = daily_sales["date"].dt.weekday

X = daily_sales[["day", "month", "weekday"]]
y = daily_sales["revenue"]

model = RandomForestRegressor()
model.fit(X, y)

future_dates = pd.date_range(daily_sales["date"].max(), periods=forecast_days)

future_df = pd.DataFrame({"date": future_dates})
future_df["day"] = future_df["date"].dt.day
future_df["month"] = future_df["date"].dt.month
future_df["weekday"] = future_df["date"].dt.weekday

future_df["forecast"] = model.predict(future_df[["day","month","weekday"]])

fig, ax = plt.subplots()
ax.plot(daily_sales["date"], y, label="Actual")
ax.plot(future_df["date"], future_df["forecast"], linestyle="--", label="Forecast")
ax.legend()
st.pyplot(fig)

# ─── INVENTORY ─────────────────────────
st.subheader("📦 Inventory Optimization")

lead_time = st.slider("Lead Time", 1, 30, 7)

avg_demand = daily["revenue"].mean()
std_demand = daily["revenue"].std()

safety_stock = 1.65 * std_demand * np.sqrt(lead_time)
reorder_point = avg_demand * lead_time + safety_stock

col1, col2 = st.columns(2)
col1.metric("Safety Stock", f"₹{safety_stock:,.0f}")
col2.metric("Reorder Point", f"₹{reorder_point:,.0f}")

if avg_demand < reorder_point:
    st.error("⚠️ Reorder Required")
else:
    st.success("✅ Stock OK")

# ─── DOWNLOAD REPORT ─────────────────────────
st.subheader("📥 Download Report")

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="retail_report.csv",
    mime="text/csv"
)