import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def pearson_correlation(matrix):
    """
    Computes the Pearson correlation coefficient between users for collaborative filtering.
    """
    return np.corrcoef(matrix)

def hybrid_recommendation(customer_id, purchase_data, product_data, top_n=5, alpha=0.5):
    """
    Hybrid recommendation system: Combines collaborative filtering and content-based filtering.
    
    :param customer_id: The target customer for recommendations.
    :param purchase_data: DataFrame containing purchase history.
    :param product_data: DataFrame containing product details.
    :param top_n: Number of recommendations to generate.
    :param alpha: Weighting factor between collaborative and content-based filtering.
    :return: List of recommended Product IDs.
    """
    
    # Step 1: User-Based Collaborative Filtering (Pearson Correlation)
    user_product_matrix = purchase_data.pivot_table(index='CustomerID', columns='ProductID', values='PurchaseAmount', fill_value=0)
    user_similarity = pearson_correlation(user_product_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_product_matrix.index, columns=user_product_matrix.index)
    
    similar_users = user_similarity_df[customer_id].sort_values(ascending=False).iloc[1:].index
    collaborative_recommendations = set()

    for similar_user in similar_users:
        user_purchases = purchase_data[purchase_data['CustomerID'] == similar_user]['ProductID'].tolist()
        collaborative_recommendations.update(user_purchases)
        if len(collaborative_recommendations) >= top_n:
            break

    # Step 2: Content-Based Filtering (TF-IDF on Product Category)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(product_data['Category'])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    user_purchases = purchase_data[purchase_data['CustomerID'] == customer_id]['ProductID'].tolist()
    purchased_indices = [product_data[product_data['ProductID'] == pid].index[0] for pid in user_purchases]
    similar_products = similarity_matrix[purchased_indices].mean(axis=0)
    content_recommendations = product_data.iloc[similar_products.argsort()[-top_n:][::-1]]['ProductID'].tolist()

    # Step 3: Merge Recommendations Using Alpha
    collab_weight = alpha
    content_weight = 1 - alpha

    collab_recommendations = list(collaborative_recommendations)[:top_n]
    content_recommendations = content_recommendations[:top_n]

    hybrid_recommendations = list(set(
        collab_recommendations[:int(top_n * collab_weight)] +
        content_recommendations[:int(top_n * content_weight)]
    ))

    return hybrid_recommendations
