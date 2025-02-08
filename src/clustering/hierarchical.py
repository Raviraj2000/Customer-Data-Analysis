import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from src.utils.preprocess import extract_features, preprocess_data

def plot_dendrogram(customer_data, max_d=20):
    """
    Plots the dendrogram to determine the best number of clusters for Hierarchical Clustering.
    """
    _, scaled_data, _ = preprocess_data(customer_data)
    linkage_matrix = linkage(scaled_data, method='ward')

    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=6)
    plt.axhline(y=max_d, color='r', linestyle='--')  # Draw a horizontal line to indicate cluster cutoff
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Customer Index")
    plt.ylabel("Distance")
    plt.show()

def find_best_k_hierarchical(customer_data, max_clusters=10):
    """
    Uses the silhouette score to determine the best number of clusters for Hierarchical Clustering.
    """
    _, scaled_data, _ = preprocess_data(customer_data)
    linkage_matrix = linkage(scaled_data, method='ward')

    best_k = 2
    best_score = -1
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
        score = silhouette_score(scaled_data, cluster_labels)
        silhouette_scores.append((k, score))

        print(f"[INFO] k={k}, Silhouette Score={score:.2f}")

        if score > best_score:
            best_score = score
            best_k = k

    # Plot silhouette scores for different cluster counts
    k_values, scores = zip(*silhouette_scores)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters (Hierarchical Clustering)')
    plt.show()

    print(f"[INFO] Best number of clusters chosen: k={best_k} with Silhouette Score={best_score:.2f}")
    return best_k

def apply_hierarchical(customer_data, max_clusters=10):
    """
    Finds the best number of clusters, applies Hierarchical Clustering, and visualizes the results.
    """
    customer_data, scaled_data, _ = preprocess_data(customer_data)

    # Step 1: Find the best number of clusters
    best_k = find_best_k_hierarchical(customer_data, max_clusters)

    # Step 2: Apply Hierarchical Clustering
    model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    customer_data['Hierarchical_Cluster'] = model.fit_predict(scaled_data)

    # Step 3: Visualize Clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        customer_data['total_spent'], 
        customer_data['purchase_count'], 
        c=customer_data['Hierarchical_Cluster'], cmap='viridis', alpha=0.7
    )
    plt.xlabel('Total Spending')
    plt.ylabel('Purchase Count')
    plt.title(f'Hierarchical Clustering (k={best_k})')
    plt.colorbar(scatter, label='Cluster')
    plt.show()

    return customer_data
