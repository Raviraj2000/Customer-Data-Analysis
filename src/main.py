from utils.preprocess import load_data, extract_features
from clustering.kmeans import apply_kmeans_with_elbow
from recommendation.hybrid import hybrid_recommendation
from utils.preprocess import load_data

# Load dataset
df_purchases = load_data("data/cleaned/purchase_history_cleaned.csv")
df_products = load_data("data/cleaned/products.csv")
df_customer = extract_features(df_purchases)

# Run K-Means Clustering
print("\n[INFO] Running K-Means Clustering...")
customer_data = apply_kmeans_with_elbow(df_customer, min_k=2, max_clusters=10)
print("[INFO] K-Means clustering completed and saved.")

customer_id = 37  # Change this to test different users
hybrid_recs = hybrid_recommendation(customer_id, df_purchases, df_products, top_n=5, alpha=0.4)

# Print recommendations with product names and categories
print(f"\nHybrid Model Recommendations for Customer {customer_id}:")
for product_id in hybrid_recs:
    product_info = df_products[df_products["ProductID"] == product_id]
    if not product_info.empty:
        product_name = product_info["ProductName"].values[0]
        category = product_info["Category"].values[0]
        print(f" - {product_name} ({category})")
    else:
        print(f" - Product ID {product_id} (Not Found)")

# Count how many times the customer bought from each category
customer_purchases = df_purchases[df_purchases["CustomerID"] == customer_id]
category_counts = customer_purchases.groupby("Category")["ProductID"].count().reset_index()
category_counts.columns = ["Category", "Purchase Count"]

print(f"\n Customer {customer_id} Purchase History by Category:")
for _, row in category_counts.iterrows():
    print(f" - {row['Category']}: {row['Purchase Count']} purchases")