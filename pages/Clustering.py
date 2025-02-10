import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.mixture import GaussianMixture
from utils.preprocess import extract_features, preprocess_data
from kneed import KneeLocator

# ---------------------------- #
# **Function: Find Best k Using BIC for GMM**
# ---------------------------- #
def find_best_k_gmm(scaled_data, min_k=2, max_clusters=10, bic_drop_threshold=1.5, knee_sensitivity=1.0):
    """
    Finds the best number of clusters using BIC and selects the first major drop in BIC score.
    This version attempts to detect when additional clusters yield diminishing returns.
    """
    bic_scores = []
    k_values = list(range(min_k, max_clusters + 1))
    
    # Calculate BIC scores for each cluster number
    for k in k_values:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(scaled_data)
        bic_scores.append(gmm.bic(scaled_data))
    
    # Compute differences in BIC scores between consecutive k values
    bic_drops = np.diff(bic_scores)
    # Compute drop ratios between consecutive drops
    bic_drop_ratios = np.abs(bic_drops[:-1] / bic_drops[1:])
    
    # Check for a significant drop using the provided threshold.
    # Here, we detect cases where the drop ratio is high,
    # meaning that adding an extra cluster significantly improves BIC relative to the next step.
    significant_drop_indices = np.where(bic_drop_ratios > bic_drop_threshold)[0]

    if len(significant_drop_indices) > 0:
        # Take the first occurrence where the drop is significant
        optimal_k = k_values[significant_drop_indices[0] + 1]
    else:
        # Use KneeLocator as a fallback, with an adjustable sensitivity
        knee_locator = KneeLocator(
            k_values, 
            bic_scores, 
            curve="convex", 
            direction="decreasing",
            S=knee_sensitivity
        )
        optimal_k = knee_locator.elbow if knee_locator.elbow else k_values[np.argmin(bic_scores)]

    return optimal_k, k_values, bic_scores

# ---------------------------- #
# **Function: Apply GMM Clustering**
# ---------------------------- #
def apply_gmm(customer_data, k):
    """
    Applies Gaussian Mixture Model (GMM) clustering on customer data.
    """
    customer_data, scaled_data, _ = preprocess_data(customer_data)
    gmm = GaussianMixture(n_components=k, random_state=42)
    customer_data['Cluster'] = gmm.fit_predict(scaled_data)
    return customer_data

# ---------------------------- #
# **Function: Assign Cluster Labels**
# ---------------------------- #
def assign_cluster_labels(customer_data):
    """
    Dynamically assigns cluster labels based on the ratio of total spending to purchase count.
    Clusters with a lower spend-per-purchase ratio (i.e. low total_spent but high purchase_count)
    are labeled as "Budget Spenders", while clusters with a higher ratio are labeled as "Premium Customers".
    """
    # Compute cluster statistics
    cluster_stats = customer_data.groupby("Cluster").agg(
        avg_spent=("total_spent", "mean"),
        avg_purchases=("purchase_count", "mean")
    )
    # Calculate spend per purchase ratio; lower values indicate budget spending habits.
    cluster_stats["spend_per_purchase"] = cluster_stats["avg_spent"] / cluster_stats["avg_purchases"]
    # Sort clusters by the spend per purchase ratio in ascending order.
    cluster_stats = cluster_stats.sort_values(by="spend_per_purchase", ascending=True)
    
    n_clusters = len(cluster_stats)

    labels = ["Budget Spenders", "Value Shoppers", "Moderate Shoppers", "High Spenders"]

    # Assign the labels in order of increasing spend_per_purchase ratio.
    cluster_stats["Cluster Label"] = labels[:n_clusters]
    
    # Create a mapping from original cluster index to the new label.
    # Note: cluster_stats index is the cluster identifier from GMM.
    cluster_label_mapping = cluster_stats["Cluster Label"].to_dict()
    
    # Map the new labels back to the original dataset.
    customer_data["Cluster Label"] = customer_data["Cluster"].map(cluster_label_mapping)
    
    return customer_data

# ---------------------------- #
# **Streamlit UI**
# ---------------------------- #
st.set_page_config(page_title="Customer Clustering with GMM", layout="wide")

st.markdown("## 🎯 Customer Clustering with Gaussian Mixture Model (GMM)")

# Load and preprocess data
df_cleaned = pd.read_csv("./data/raw/purchase_history.csv")
customer_data = extract_features(df_cleaned)
_, scaled_data, _ = preprocess_data(customer_data)

# Step 1: Determine Optimal k Using BIC
optimal_k, k_values, bic_scores = find_best_k_gmm(
    scaled_data, 
    min_k=2, 
    max_clusters=10, 
    bic_drop_threshold=1.5, 
    knee_sensitivity=1.0
)

# Step 2: Apply GMM Clustering with optimal k
customer_data = apply_gmm(customer_data, optimal_k)

# Step 3: Assign labels to clusters based on spend-per-purchase ratio.
customer_data = assign_cluster_labels(customer_data)

st.markdown("### 📌 Clustering Results")
st.write(f"🔢 **Optimal Number of Clusters (k): {optimal_k}**")

# ---------------------------- #
# **BIC Score Plot**
# ---------------------------- #
fig_bic = px.line(
    x=k_values, 
    y=bic_scores, 
    markers=True, 
    title="📉 Optimal K Selection using BIC (Bayesian Information Criterion)",
    labels={"x": "Number of Clusters (k)", "y": "BIC Score"}
)
fig_bic.add_vline(x=optimal_k, line_dash="dash", line_color="red", annotation_text=f"Optimal k={optimal_k}")
st.plotly_chart(fig_bic, use_container_width=True)

# ---------------------------- #
# **Cluster Visualization**
# ---------------------------- #
st.markdown("### 📌 GMM Clustering Visualization")
fig_clusters = px.scatter(
    customer_data, 
    x="total_spent", 
    y="purchase_count",
    color="Cluster Label",  
    title="🛒 Customer Segments",
    labels={"total_spent": "Total Spending", "purchase_count": "Purchase Count", "Cluster": "Cluster"},
    color_discrete_map={
        "Budget Spenders": "green", 
        "Value Shoppers": "teal", 
        "Moderate Shoppers": "blue", 
        "Mid-range Shoppers": "orange", 
        "High-end Shoppers": "magenta", 
        "Premium Customers": "purple"
    },
    opacity=0.7,
    size_max=6
)
st.plotly_chart(fig_clusters, use_container_width=True)