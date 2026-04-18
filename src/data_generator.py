import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range(start="2022-01-01", end="2023-12-31")

data = []

stores = [1, 2, 3]
products = [101, 102, 103, 104, 105]
categories = ["Electronics", "Clothing", "Grocery"]

for date in dates:
    for store in stores:
        for product in products:
            units = np.random.randint(5, 50)
            price = np.random.randint(100, 1000)
            revenue = units * price

            data.append([
                date,
                store,
                product,
                np.random.choice(categories),
                units,
                revenue
            ])

df = pd.DataFrame(data, columns=[
    "date", "store_id", "product_id", "category", "units_sold", "revenue"
])

df.to_csv("data/raw/sales_data.csv", index=False)

print("✅ Raw data generated: data/raw/sales_data.csv")