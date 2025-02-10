import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
from scipy.cluster.hierarchy import dendrogram, linkage
from utils.preprocess import extract_features, preprocess_data

# ---------------------------- #
# **Function: Find Best k Using BIC for GMM**
# ---------------------------- #
def find_best_k_gmm(scaled_data, min_k=2, max_clusters=10):
    """
    Finds the best number of clusters using the Bayesian Information Criterion (BIC) for GMM.
    """
    bic_scores = []
    k_values = list(range(min_k, max_clusters + 1))

    for k in k_values:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(scaled_data)
        bic_scores.append(gmm.bic(scaled_data))

    optimal_k = k_values[np.argmin(bic_scores)]
    return optimal_k, k_values, bic_scores

# ---------------------------- #
# **Function: Find Best k Using Elbow Method for K-Means**
# ---------------------------- #
def find_best_k_elbow(scaled_data, min_k=2, max_clusters=10):
    """
    Finds the best number of clusters using the Elbow Method for K-Means.
    """
    wcss = []
    k_values = list(range(min_k, max_clusters + 1))

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    elbow_finder = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")
    optimal_k = elbow_finder.elbow if elbow_finder.elbow else min_k

    return optimal_k, k_values, wcss

# ---------------------------- #
# **Function: Plot Dendrogram for Hierarchical**
# ---------------------------- #
def plot_dendrogram(scaled_data):
    """
    Plots a dendrogram to visualize hierarchical clustering.
    """
    linkage_matrix = linkage(scaled_data, method='ward')
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=6)
    plt.title("Hierarchical Clustering - Dendrogram")
    plt.xlabel("Customer Index")
    plt.ylabel("Distance")
    st.pyplot(fig)

# ---------------------------- #
# **Clustering Algorithms**
# ---------------------------- #
def apply_gmm(customer_data, k):
    customer_data, scaled_data, _ = preprocess_data(customer_data)
    gmm = GaussianMixture(n_components=k, random_state=42)
    customer_data['Cluster'] = gmm.fit_predict(scaled_data)
    return customer_data

def apply_kmeans(customer_data, k):
    customer_data, scaled_data, _ = preprocess_data(customer_data)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    customer_data['Cluster'] = kmeans.fit_predict(scaled_data)
    return customer_data

def apply_hierarchical(customer_data, k):
    customer_data, scaled_data, _ = preprocess_data(customer_data)
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    customer_data['Cluster'] = model.fit_predict(scaled_data)
    return customer_data

def apply_dbscan(customer_data, eps, min_samples):
    customer_data, scaled_data, _ = preprocess_data(customer_data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    customer_data['Cluster'] = dbscan.fit_predict(scaled_data)
    return customer_data

# ---------------------------- #
# **Streamlit UI**
# ---------------------------- #
st.set_page_config(page_title="Customer Clustering", layout="wide")

st.title("🎯 Customer Clustering")

# Load and preprocess data
df_cleaned = pd.read_csv("./data/cleaned/purchase_history_cleaned.csv")
customer_data = extract_features(df_cleaned)
_, scaled_data, _ = preprocess_data(customer_data)

# Step 1: Clustering Algorithm Selection
clustering_method = st.sidebar.selectbox(
    "Select Clustering Algorithm", 
    ["Gaussian Mixture Model (GMM)", "K-Means", "Hierarchical Clustering", "DBSCAN"]
)

# Step 2: Determine optimal k or hyperparameters
if clustering_method == "Gaussian Mixture Model (GMM)":
    optimal_k, k_values, scores = find_best_k_gmm(scaled_data, min_k=2, max_clusters=10)
    score_label = "BIC (Bayesian Information Criterion)"
elif clustering_method == "K-Means":
    optimal_k, k_values, scores = find_best_k_elbow(scaled_data, min_k=2, max_clusters=10)
    score_label = "WCSS (Within-Cluster Sum of Squares)"
else:
    optimal_k = None  # DBSCAN doesn't use k

# Step 3: Parameter selection
if clustering_method in ["Gaussian Mixture Model (GMM)", "K-Means"]:
    k_selected = st.sidebar.slider(
        "Select Number of Clusters (k)", 
        min_value=2, 
        max_value=10, 
        value=optimal_k
    )
elif clustering_method == "Hierarchical Clustering":
    st.subheader("📊 Hierarchical Clustering - Dendrogram")
    plot_dendrogram(scaled_data)
    k_selected = st.sidebar.slider(
        "Select Number of Clusters (k)", 
        min_value=2, 
        max_value=10, 
        value=3
    )
elif clustering_method == "DBSCAN":
    eps = st.sidebar.slider("Epsilon (eps)", min_value=0.1, max_value=5.0, step=0.1, value=1.0)
    min_samples = st.sidebar.slider("Min Samples", min_value=2, max_value=10, value=5)

st.write(f"🔢 **Selected Clusters (k): {k_selected if clustering_method != 'DBSCAN' else 'N/A'}**")

# Step 4: Apply selected clustering method
if clustering_method == "Gaussian Mixture Model (GMM)":
    customer_data = apply_gmm(customer_data, k_selected)
elif clustering_method == "K-Means":
    customer_data = apply_kmeans(customer_data, k_selected)
elif clustering_method == "Hierarchical Clustering":
    customer_data = apply_hierarchical(customer_data, k_selected)
else:
    customer_data = apply_dbscan(customer_data, eps, min_samples)

# ---------------------------- #
# **Elbow Method / BIC Plot**
# ---------------------------- #
if clustering_method in ["Gaussian Mixture Model (GMM)", "K-Means"]:
    fig_scores = px.line(
        x=k_values, 
        y=scores, 
        markers=True, 
        title=f"{clustering_method} - Optimal K Selection",
        labels={"x": "Number of Clusters (k)", "y": score_label}
    )
    fig_scores.add_vline(x=optimal_k, line_dash="dash", line_color="red", annotation_text=f"Optimal k={optimal_k}")
    st.plotly_chart(fig_scores, use_container_width=True)

# ---------------------------- #
# **Cluster Visualization**
# ---------------------------- #
fig_clusters = px.scatter(
    customer_data, 
    x="total_spent", 
    y="purchase_count",
    color=customer_data["Cluster"].astype(str),  
    title=f"{clustering_method} Clustering",
    labels={"total_spent": "Total Spending", "purchase_count": "Purchase Count", "Cluster": "Cluster"},
    color_discrete_sequence=px.colors.qualitative.Bold,
    opacity=0.7,
    size_max=6
)

st.plotly_chart(fig_clusters, use_container_width=True)
