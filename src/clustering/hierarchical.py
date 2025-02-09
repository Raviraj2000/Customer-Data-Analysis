import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
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

def determine_k_from_dendrogram(customer_data, max_d=20):
    """
    Determines the number of clusters by cutting the dendrogram at a fixed distance.
    """
    _, scaled_data, _ = preprocess_data(customer_data)
    linkage_matrix = linkage(scaled_data, method='ward')
    cluster_labels = fcluster(linkage_matrix, max_d, criterion='distance')
    
    best_k = len(set(cluster_labels))  # Count the unique cluster labels
    print(f"[INFO] Best estimated number of clusters based on dendrogram cutoff: k={best_k}")
    return best_k

def label_clusters(cluster_assignments, customer_data):
    """
    Labels clusters based on spending behavior.
    """
    cluster_means = customer_data.groupby(cluster_assignments)['total_spent'].mean().sort_values()
    cluster_labels = {}

    # Assign labels based on spending levels
    for i, cluster in enumerate(cluster_means.index):
        if i == 0:
            cluster_labels[cluster] = "Budget Shoppers"
        elif i == len(cluster_means) - 1:
            cluster_labels[cluster] = "High Spenders"
        else:
            cluster_labels[cluster] = "Regular Shoppers"

    # Map the numeric cluster labels to named clusters
    return cluster_assignments.map(cluster_labels)

def apply_hierarchical(customer_data, max_d=20):
    """
    Finds the best number of clusters using dendrogram analysis, applies Hierarchical Clustering, 
    labels the clusters, and visualizes the results.
    """
    customer_data, scaled_data, _ = preprocess_data(customer_data)

    # Step 1: Determine the best number of clusters using dendrogram cutoff
    best_k = determine_k_from_dendrogram(customer_data, max_d)

    # Step 2: Apply Hierarchical Clustering
    model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    customer_data['Cluster_ID'] = model.fit_predict(scaled_data)

    # Step 3: Assign Cluster Labels
    customer_data['Cluster_Label'] = label_clusters(customer_data['Cluster_ID'], customer_data)

    # Step 4: Visualize Clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        customer_data['total_spent'], 
        customer_data['purchase_count'], 
        c=customer_data['Cluster_ID'], cmap='viridis', alpha=0.7
    )
    plt.xlabel('Total Spending')
    plt.ylabel('Purchase Count')
    plt.title(f'Hierarchical Clustering (k={best_k})')
    plt.colorbar(scatter, label='Cluster')
    plt.show()

    return customer_data
