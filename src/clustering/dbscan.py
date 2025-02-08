import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from src.utils.preprocess import extract_features, preprocess_data

def find_optimal_eps(customer_data, min_eps=0.1, max_eps=2.0, step=0.1, k=5):
    """
    Finds the best epsilon value for DBSCAN using silhouette score.
    Plots k-nearest neighbor distances to visualize potential eps values.
    """
    _, scaled_data, _ = preprocess_data(customer_data)

    # Step 1: Nearest Neighbor Distance Plot (Elbow Method for eps)
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(scaled_data)
    distances, _ = neighbors_fit.kneighbors(scaled_data)
    distances = np.sort(distances[:, k-1], axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.title('Optimal eps for DBSCAN (Look for the elbow point)')
    plt.show()

    # Step 2: Evaluate Silhouette Score Across Different eps Values
    best_eps = min_eps
    best_score = -1
    eps_values = np.arange(min_eps, max_eps, step)
    scores = []

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=k)
        cluster_labels = dbscan.fit_predict(scaled_data)

        # Ignore single-cluster results (-1 means all points are noise)
        if len(set(cluster_labels)) > 1:
            score = silhouette_score(scaled_data, cluster_labels)
            scores.append((eps, score))

            print(f"[INFO] eps={eps:.2f}, Silhouette Score={score:.2f}")

            if score > best_score:
                best_score = score
                best_eps = eps

    # Step 3: Plot Silhouette Scores vs eps values
    if scores:
        eps_list, silhouette_list = zip(*scores)
        plt.figure(figsize=(8, 5))
        plt.plot(eps_list, silhouette_list, marker='o', linestyle='-')
        plt.xlabel('Epsilon (eps)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Epsilon for DBSCAN')
        plt.show()

    print(f"[INFO] Best epsilon chosen: eps={best_eps:.2f} with Silhouette Score={best_score:.2f}")
    return best_eps

def apply_dbscan(customer_data, min_eps=0.1, max_eps=2.0, step=0.1, k=5):
    """
    Optimizes eps, applies DBSCAN clustering, and visualizes the clusters.
    """
    customer_data, scaled_data, _ = preprocess_data(customer_data)

    # Step 1: Find the best eps
    best_eps = find_optimal_eps(customer_data, min_eps, max_eps, step, k)

    # Step 2: Apply DBSCAN
    dbscan = DBSCAN(eps=best_eps, min_samples=k)
    customer_data['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_data)

    # Step 3: Visualize DBSCAN Clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        customer_data['total_spent'], 
        customer_data['purchase_count'], 
        c=customer_data['DBSCAN_Cluster'], cmap='viridis', alpha=0.7
    )
    plt.xlabel('Total Spending')
    plt.ylabel('Purchase Count')
    plt.title(f'DBSCAN Clustering (eps={best_eps:.2f})')
    plt.colorbar(scatter, label='Cluster')
    plt.show()

    return customer_data