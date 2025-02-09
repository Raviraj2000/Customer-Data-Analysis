import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def user_based_recommendation(customer_id, purchase_data, top_n=5):
    """
    User-based collaborative filtering: Finds similar users and recommends products.
    """
    # Create a user-product matrix
    user_product_matrix = purchase_data.pivot_table(index='CustomerID', columns='ProductID', values='PurchaseAmount', fill_value=0)

    # Compute similarity between users
    user_similarity = cosine_similarity(user_product_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_product_matrix.index, columns=user_product_matrix.index)

    # Get the most similar user (excluding the customer themselves)
    similar_users = user_similarity_df[customer_id].sort_values(ascending=False).iloc[1:].index

    recommended_products = set()
    for similar_user in similar_users:
        user_purchases = purchase_data[purchase_data['CustomerID'] == similar_user]['ProductID'].tolist()
        recommended_products.update(user_purchases)
        if len(recommended_products) >= top_n:
            break

    return list(recommended_products)[:top_n]

def item_based_recommendation(customer_id, purchase_data, top_n=5):
    """
    Item-based collaborative filtering: Finds similar products based on purchase patterns.
    """
    # Create a user-product matrix
    user_product_matrix = purchase_data.pivot_table(index='CustomerID', columns='ProductID', values='PurchaseAmount', fill_value=0)

    # Compute similarity between products
    item_similarity = cosine_similarity(user_product_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_product_matrix.columns, columns=user_product_matrix.columns)

    # Get products purchased by the user
    user_purchases = purchase_data[purchase_data['CustomerID'] == customer_id]['ProductID'].tolist()

    recommended_products = set()
    for product in user_purchases:
        similar_products = item_similarity_df[product].sort_values(ascending=False).iloc[1:top_n+1].index.tolist()
        recommended_products.update(similar_products)

    return list(recommended_products)[:top_n]
