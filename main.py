import os
import sys

# Dynamically add 'src/' to Python's path (Works on Windows, Mac, Linux)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Now import clustering modules
from utils.preprocess import load_data, extract_features
from src.clustering.kmeans import apply_kmeans_with_silhouette
from src.clustering.dbscan import apply_dbscan
from src.clustering.hierarchical import apply_hierarchical, plot_dendrogram

# Load dataset
df = load_data()
customer_data = extract_features(df)

# Run K-Means Clustering
print("\n[INFO] Running K-Means Clustering...")
customer_data = apply_kmeans_with_silhouette(customer_data, min_k=2, max_clusters=10)
customer_data.to_csv('data/cleaned/customer_clusters_kmeans.csv', index=False)
print("[INFO] K-Means clustering completed and saved.")

# Run DBSCAN Clustering
print("\n[INFO] Running DBSCAN Clustering...")
customer_data = apply_dbscan(customer_data, min_eps=0.1, max_eps=2.0, step=0.1, k=5)
customer_data_dbscan.to_csv('data/cleaned/customer_clusters_dbscan.csv', index=False)
print("[INFO] DBSCAN clustering completed and saved.")

# Run Hierarchical Clustering
print("\n[INFO] Running Hierarchical Clustering...")
plot_dendrogram(customer_data)  # Optional: Dendrogram for clusters
customer_data = apply_hierarchical(customer_data, max_clusters=10)
customer_data_hierarchical.to_csv('data/cleaned/customer_clusters_hierarchical.csv', index=False)
print("[INFO] Hierarchical clustering completed and saved.")

print("\n[INFO] All clustering models executed successfully!")