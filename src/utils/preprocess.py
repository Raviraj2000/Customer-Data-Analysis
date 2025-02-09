import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load cleaned purchase history
def load_data(file_path):
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
    customer_ids = df[['CustomerID']]
    scaled_features = scaler.fit_transform(df.iloc[:, 1:])  # Exclude CustomerID
    scaled_df = pd.DataFrame(scaled_features, columns=df.columns[1:])
    scaled_df.insert(0, 'CustomerID', customer_ids.values)
    return df, scaled_df, scaler
