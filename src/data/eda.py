import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df_raw = pd.read_csv('./data/raw/purchase_history.csv')
df_cleaned = pd.read_csv('./data/cleaned/purchase_history_cleaned.csv')

print("\n### Differences between raw and cleaned dataframes ###\n")

print(f"\nTotal rows in raw data: {df_raw.shape[0]}")
print(f"Total rows in cleaned data: {df_cleaned.shape[0]}")

# 1. Check for missing values
missing_values_raw = df_raw.isnull().sum()
missing_values_cleaned = df_cleaned.isnull().sum()

print("\nMissing values in raw data (only showing columns with missing values):")
print(missing_values_raw[missing_values_raw > 0])

print("\nMissing values in cleaned data (should ideally be none or significantly reduced):")
print(missing_values_cleaned[missing_values_cleaned > 0])

# 2. Check for duplicate rows
duplicate_rows_raw = df_raw.duplicated().sum()
duplicate_rows_cleaned = df_cleaned.duplicated().sum()

print(f"\nNumber of duplicate rows in raw data: {duplicate_rows_raw}")
print(f"Number of duplicate rows in cleaned data: {duplicate_rows_cleaned}")

# 3. Check for differences in column names
columns_raw = set(df_raw.columns)
columns_cleaned = set(df_cleaned.columns)

print("\nColumns present only in raw data (possibly removed during cleaning):")
print(columns_raw - columns_cleaned)

print("\nColumns present only in cleaned data (possibly new columns added during cleaning):")
print(columns_cleaned - columns_raw)

# 4. Check for differences in data types
print("\nData type differences between raw and cleaned data:")
for column in columns_raw.intersection(columns_cleaned):  # Only check common columns
    if df_raw[column].dtype != df_cleaned[column].dtype:
        print(f"Column '{column}': raw type = {df_raw[column].dtype}, cleaned type = {df_cleaned[column].dtype}")

print("\n[INFO] Data comparison complete.\n")

df_products = df_cleaned[['ProductID', 'ProductName', 'Category']].drop_duplicates()
df_products.to_csv('./data/cleaned/products.csv', index=False)
##################Top-Selling Products##################
#By Revenue
revenue_by_product = (
    df_cleaned.groupby(['ProductID', 'ProductName'])['PurchaseAmount']
      .sum()
      .sort_values(ascending=False)
)

print("\nTop 5 products by total revenue:")
print(revenue_by_product.head(5))

#By Purchase Count
# Count how many times each product was purchased
count_by_product = (
    df_cleaned.groupby(['ProductID', 'ProductName'])
      .size()
      .sort_values(ascending=False)
)

print("\nTop 5 products by purchase count:")
print(count_by_product.head(5))
#########################################################

##################Top-Selling Categories#################
#By Total Revenue
revenue_by_category = (
    df_cleaned.groupby('Category')['PurchaseAmount']
      .sum()
      .sort_values(ascending=False)
)

print("\nCategories by total revenue (descending):")
print(revenue_by_category)

#By Purchase Count
count_by_category = (
    df_cleaned.groupby('Category')
      .size()
      .sort_values(ascending=False)
)

print("\nCategories by purchase count (descending):")
print(count_by_category)

##################Average Spending per Customer##################
total_spend_per_customer = df_cleaned.groupby('CustomerID')['PurchaseAmount'].sum()
print("\nTotal spending by each customer (first 10 shown):")
print(total_spend_per_customer.head(10))

avg_spend_per_customer_txn = df_cleaned.groupby('CustomerID')['PurchaseAmount'].mean()
print("\nAverage spend per transaction for each customer (first 10):")
print(avg_spend_per_customer_txn.head(10))

# If you'd like an overall average of that metric:
overall_avg_per_transaction = avg_spend_per_customer_txn.mean()
print(f"\nOverall average spend per transaction (across all customers): ${overall_avg_per_transaction:.2f}")
##################################################################