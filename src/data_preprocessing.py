import os
import pandas as pd
import numpy as np


# 1. Load the original/synthetic dataset
df = pd.read_csv('./data/raw/purchase_history.csv')

# 2. Remove unnecessary personal info columns (if they exist)
columns_to_drop = ['CustomerName', 'Email', 'Phone', 'Address']
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
if existing_cols_to_drop:
    df.drop(columns=existing_cols_to_drop, inplace=True)

# 3. Check for missing data and fill or drop as needed
#    Example: fill text columns with 'Unknown'
text_columns = ['ProductName', 'Category']
for col in text_columns:
    if col in df.columns:
        df[col].fillna('Unknown', inplace=True)

# Fill missing PurchaseAmount with the median for the specific product
if 'PurchaseAmount' in df.columns:
    df['PurchaseAmount'] = df.groupby('ProductID')['PurchaseAmount'].transform(lambda x: x.fillna(x.median()))

# If a product has all missing values, fallback to the category median
if 'PurchaseAmount' in df.columns and 'Category' in df.columns:
    df['PurchaseAmount'] = df.groupby('Category')['PurchaseAmount'].transform(lambda x: x.fillna(x.median()))

# 4. Convert PurchaseDate to datetime (drop invalid dates)
if 'PurchaseDate' in df.columns:
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
    df = df[df['PurchaseDate'].notnull()]

# 5. Drop rows missing critical IDs
critical_cols = ['CustomerID', 'ProductID']
for col in critical_cols:
    if col in df.columns:
        df = df[df[col].notnull()]

# 6. Optional: clip outliers in PurchaseAmount using the 1.5 * IQR rule
if 'PurchaseAmount' in df.columns:
    q1 = df['PurchaseAmount'].quantile(0.25)
    q3 = df['PurchaseAmount'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df['PurchaseAmount'] = df['PurchaseAmount'].clip(lower=lower_bound, upper=upper_bound)

# 7. Save the cleaned DataFrame to 'data/cleaned'
df.to_csv('./data/cleaned/purchase_history_cleaned.csv', index=False)
