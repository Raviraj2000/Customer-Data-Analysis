import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load cleaned purchase history
def load_data(file_path='data/cleaned/purchase_history_cleaned.csv'):
    return pd.read_csv(file_path)

# Extract features for clustering
def extract_features(df):
    customer_data = df.groupby('CustomerID').agg(
        total_spent=('PurchaseAmount', 'sum'),
        purchase_count=('PurchaseAmount', 'count'),
        avg_purchase_amount=('PurchaseAmount', 'mean'),
        unique_categories=('Category', 'nunique')
    ).reset_index()
    
    return customer_data

# Standardize features
def preprocess_data(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.iloc[:, 1:])  # Exclude CustomerID
    return df, scaled_features, scaler
