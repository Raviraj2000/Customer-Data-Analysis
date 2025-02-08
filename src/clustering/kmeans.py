import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.utils.preprocess import extract_features, preprocess_data

def find_best_k_silhouette(customer_data, min_k=3, max_clusters=10):
    """
    Finds the best number of clusters using the Silhouette Score.
    Ensures k is at least `min_k` to prevent trivial clustering.
    """
    _, scaled_data, _ = preprocess_data(customer_data)

    best_k = min_k
    best_score = -1
    wcss = []
    silhouette_scores = []

    for k in range(min_k, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, cluster_labels)

        wcss.append(kmeans.inertia_)
        silhouette_scores.append(score)

        print(f"[INFO] k={k}, Silhouette Score={score:.2f}")

        if score > best_score:
            best_score = score
            best_k = k

    # Plot Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(range(min_k, max_clusters + 1), wcss, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal K')
    plt.show()

    # Plot Silhouette Score Trend
    plt.figure(figsize=(8, 5))
    plt.plot(range(min_k, max_clusters + 1), silhouette_scores, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different k')
    plt.show()

    print(f"[INFO] Best k chosen based on Silhouette Score: {best_k}")
    return best_k

def apply_kmeans_with_silhouette(customer_data, min_k=3, max_clusters=10):
    """
    Uses the Silhouette Score to determine the best k, applies K-Means, and visualizes clusters.
    """
    customer_data, scaled_data, _ = preprocess_data(customer_data)

    # Step 1: Find the best number of clusters using silhouette score
    optimal_k = find_best_k_silhouette(customer_data, min_k=min_k, max_clusters=max_clusters)

    # Step 2: Apply K-Means with the best k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    customer_data['KMeans_Cluster'] = kmeans.fit_predict(scaled_data)

    # Step 3: Plot K-Means clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        customer_data['total_spent'], 
        customer_data['purchase_count'], 
        c=customer_data['KMeans_Cluster'], cmap='viridis', alpha=0.7
    )
    plt.xlabel('Total Spending')
    plt.ylabel('Purchase Count')
    plt.title(f'K-Means Clustering (k={optimal_k})')
    plt.colorbar(scatter, label='Cluster')
    plt.show()

    return customer_data
