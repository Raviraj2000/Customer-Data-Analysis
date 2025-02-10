import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------- #
# **Function: Pearson Correlation for Collaborative Filtering**
# ---------------------------- #
def pearson_correlation(matrix):
    """ Computes the Pearson correlation coefficient between users for collaborative filtering. """
    return np.corrcoef(matrix)

# ---------------------------- #
# **Function: Hybrid Recommendation System**
# ---------------------------- #
def hybrid_recommendation(customer_id, purchase_data, product_data, top_n=5, alpha=0.5):
    """
    Hybrid recommendation system: Combines collaborative filtering and content-based filtering.

    :param customer_id: The target customer for recommendations.
    :param purchase_data: DataFrame containing purchase history.
    :param product_data: DataFrame containing product details.
    :param top_n: Number of recommendations to generate.
    :param alpha: Weighting factor between collaborative and content-based filtering.
    :return: Recommended products with names & categories, similar users, past transactions.
    """

    # Step 1: User-Based Collaborative Filtering (Pearson Correlation)
    user_product_matrix = purchase_data.pivot_table(index='CustomerID', columns='ProductID', values='PurchaseAmount', fill_value=0)
    user_similarity = pearson_correlation(user_product_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_product_matrix.index, columns=user_product_matrix.index)

    collaborative_recommendations = set()
    similar_users = []
    
    if customer_id in user_similarity_df.index:
        similar_users = user_similarity_df[customer_id].sort_values(ascending=False).iloc[1:6].index.tolist()  # Top 5 similar users

        for similar_user in similar_users:
            user_purchases = purchase_data[purchase_data['CustomerID'] == similar_user]['ProductID'].tolist()
            collaborative_recommendations.update(user_purchases)
            if len(collaborative_recommendations) >= top_n:
                break

    # Step 2: Content-Based Filtering (TF-IDF on Product Category)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(product_data['Category'])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Get user purchases
    user_purchases = purchase_data[purchase_data['CustomerID'] == customer_id]['ProductID'].tolist()

    # Generate content-based recommendations
    purchased_indices = [product_data[product_data['ProductID'] == pid].index[0] for pid in user_purchases if pid in product_data['ProductID'].values]
    
    content_recommendations = []
    if purchased_indices:
        similar_products = similarity_matrix[purchased_indices].mean(axis=0)
        content_recommendations = product_data.iloc[similar_products.argsort()[-top_n:][::-1]]['ProductID'].tolist()

    # Step 3: Merge Recommendations Using Alpha
    num_collab = round(top_n * alpha)
    num_content = top_n - num_collab

    collab_recommendations = list(collaborative_recommendations)[:top_n]
    content_recommendations = content_recommendations[:top_n]

    hybrid_recommendations = list(set(
        collab_recommendations[:num_collab] +
        content_recommendations[:num_content]
    ))

    # Ensure exactly `top_n` recommendations
    while len(hybrid_recommendations) < top_n:
        if len(collab_recommendations) > num_collab:
            hybrid_recommendations.append(collab_recommendations[num_collab])
            num_collab += 1
        elif len(content_recommendations) > num_content:
            hybrid_recommendations.append(content_recommendations[num_content])
            num_content += 1
        else:
            break  # No more recommendations available

    # Fetch product names & categories for display
    recommended_products = product_data[product_data['ProductID'].isin(hybrid_recommendations)][['ProductID', 'ProductName', 'Category']]

    # Fetch past transactions of the customer
    past_transactions = purchase_data[purchase_data['CustomerID'] == customer_id][['ProductID', 'PurchaseAmount']]
    past_transactions = past_transactions.merge(product_data, on='ProductID', how='left')

    return recommended_products, similar_users, past_transactions

# ---------------------------- #
# **Streamlit UI**
# ---------------------------- #
st.set_page_config(page_title="Hybrid Recommendation System", layout="wide")
st.title(f"🛍️ Hybrid Recommendation System(Collaborative + Content-Based)")
# Load datasets
df_purchases = pd.read_csv("./data/cleaned/purchase_history_cleaned.csv")
df_products = pd.read_csv("./data/cleaned/products.csv")

# Sidebar - Recommendation Settings
st.sidebar.header("Recommendation Settings")

# Display Customer ID Range
customer_id_min = df_purchases["CustomerID"].min()
customer_id_max = df_purchases["CustomerID"].max()
st.sidebar.write(f"Customer ID Range: **{customer_id_min} - {customer_id_max}**")

# Select Existing Customer ID
existing_customers = df_purchases["CustomerID"].unique()
customer_id = st.sidebar.selectbox("Select Customer ID", existing_customers)

# Alpha Weighting Slider
alpha = st.sidebar.slider("Collaborative vs Content-Based Weighting (α)", 0.0, 1.0, 0.5)

# Top-N Recommendations
top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# Generate Recommendations
if st.sidebar.button("Get Recommendations"):
    recommendations, similar_users, past_transactions = hybrid_recommendation(customer_id, df_purchases, df_products, top_n, alpha)

    # **Update Title Dynamically**
    st.subheader(f"Recommendations for Customer: {customer_id}")

    # **Display Past Transactions & Recommendations Side by Side**
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛒 Past Transactions of Selected Customer")
        if not past_transactions.empty:
            st.dataframe(past_transactions)
        else:
            st.write("⚠️ No past transactions found for this customer.")

    with col2:
        st.subheader("📌 Recommended Products")
        if not recommendations.empty:
            st.dataframe(recommendations)
        else:
            st.write("⚠️ No recommendations found. Try adjusting the settings or selecting a different customer.")

    # **Display Similar Users with Individual Dropdowns**
    st.subheader("👥 Similar Customers Identified in Collaborative Filtering")
    if similar_users:
        st.write("🔍 **Click on each dropdown to view purchase history of similar users**")

        for similar_user in similar_users:
            with st.expander(f"📌 Similar Customer {similar_user}"):
                similar_user_transactions = df_purchases[df_purchases['CustomerID'] == similar_user][['ProductID', 'PurchaseAmount']]
                similar_user_transactions = similar_user_transactions.merge(df_products, on='ProductID', how='left')
                if not similar_user_transactions.empty:
                    st.dataframe(similar_user_transactions)
                else:
                    st.write("⚠️ No purchase history found for this similar customer.")

    else:
        st.write("⚠️ No similar users found.")
