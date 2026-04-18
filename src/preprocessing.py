import pandas as pd

df = pd.read_csv("data/raw/sales_data.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()

# Sort data
df = df.sort_values("date")

# Save clean data
df.to_csv("data/processed/clean_sales.csv", index=False)

print("✅ Clean data saved: data/processed/clean_sales.csv")