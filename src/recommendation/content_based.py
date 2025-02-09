from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_products_content(customer_id, purchase_data, product_data, top_n=5):
    """
    Content-based filtering: Finds similar products based on product descriptions and categories.
    """
    user_purchases = purchase_data[purchase_data['CustomerID'] == customer_id]
    merged_data = user_purchases.merge(product_data, on='ProductID')

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(product_data['Category'])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    purchased_indices = [product_data[product_data['ProductID'] == pid].index[0] for pid in user_purchases['ProductID']]
    similar_products = similarity_matrix[purchased_indices].mean(axis=0)

    recommended_indices = similar_products.argsort()[-top_n:][::-1]
    return product_data.iloc[recommended_indices]['ProductID'].tolist()
