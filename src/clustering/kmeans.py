import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from src.utils.preprocess import extract_features, preprocess_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from src.utils.preprocess import preprocess_data

def find_best_k_elbow(customer_data, min_k=3, max_clusters=10):
    """
    Finds the best number of clusters using the Elbow Method with automatic knee detection.
    """
    _, scaled_data, _ = preprocess_data(customer_data)

    wcss = []
    k_values = list(range(min_k, max_clusters + 1))

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
        print(f"[INFO] k={k}, WCSS={wcss[-1]:.2f}")

    # Plot Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal K')
    plt.show()

    # Use KneeLocator to find the optimal k dynamically
    elbow_finder = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")
    optimal_k = elbow_finder.elbow

    if optimal_k is None:
        print("[WARNING] KneeLocator could not detect an elbow point! Defaulting to min_k.")
        optimal_k = min_k  # Prevents NoneType errors

    print(f"[INFO] Best k chosen based on Elbow Method: {optimal_k}")
    return optimal_k

def apply_kmeans_with_elbow(customer_data, min_k=3, max_clusters=10):
    """
    Uses the Elbow Method to determine the best k, applies K-Means, and visualizes clusters.
    """
    customer_data, scaled_data, _ = preprocess_data(customer_data)

    # Step 1: Find the best number of clusters using Elbow Method
    optimal_k = find_best_k_elbow(customer_data, min_k=min_k, max_clusters=max_clusters)

    # Step 2: Apply K-Means with the best k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    customer_data['KMeans_Cluster'] = kmeans.fit_predict(scaled_data.iloc[:, 1:])

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
