import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Data Analysis", layout="wide")

st.title("📊 Data Analysis")
st.write("Explore key insights on product sales, customer spending, and category trends.")

# Load cleaned data
df_cleaned = pd.read_csv("./data/cleaned/purchase_history_cleaned.csv")

# =================== Top-Selling Products & Categories =================== #
st.subheader("🔥 Top-Selling Products & Categories")

# Sidebar sliders for adjusting number of rows in tables
top_n_products = st.sidebar.slider("Select Top N Products", 5, 50, 5)
top_n_categories = st.sidebar.slider("Select Top N Categories", 5, 20, 5)
top_n_customers = st.sidebar.slider("Select Top N Customers", 5, 50, 5)

col1, col2 = st.columns(2)

# Top Products by Revenue
revenue_by_product = (
    df_cleaned.groupby(['ProductID', 'ProductName'])['PurchaseAmount']
    .sum()
    .sort_values(ascending=False)
    .head(top_n_products)
)
with col1:
    st.write(f"💰 **Top {top_n_products} Products by Revenue**")
    st.dataframe(revenue_by_product.rename("Total Revenue ($)"))

# Top Products by Purchase Count
count_by_product = (
    df_cleaned.groupby(['ProductID', 'ProductName'])
    .size()
    .sort_values(ascending=False)
    .head(top_n_products)
)
with col2:
    st.write(f"🛒 **Top {top_n_products} Products by Purchase Count**")
    st.dataframe(count_by_product.rename("Total Purchases"))

# Top Categories by Revenue
revenue_by_category = (
    df_cleaned.groupby('Category')['PurchaseAmount']
    .sum()
    .sort_values(ascending=False)
    .head(top_n_categories)
)

# Top Categories by Purchase Count
count_by_category = (
    df_cleaned.groupby('Category')
    .size()
    .sort_values(ascending=False)
    .head(top_n_categories)
)

col3, col4 = st.columns(2)
with col3:
    st.write(f"📦 **Top {top_n_categories} Categories by Revenue**")
    st.dataframe(revenue_by_category.rename("Total Revenue ($)"))

with col4:
    st.write(f"📦 **Top {top_n_categories} Categories by Purchase Count**")
    st.dataframe(count_by_category.rename("Total Purchases"))

# =================== Customer Spending Insights =================== #
st.subheader("👤 Customer Spending Insights")

col5, col6 = st.columns(2)

# Calculate total spending per customer
total_spend_per_customer = df_cleaned.groupby('CustomerID')['PurchaseAmount'].sum()

# ✅ Calculate the number of transactions per customer
num_transactions_per_customer = df_cleaned.groupby('CustomerID')['PurchaseAmount'].count()

# ✅ Compute the average spend per transaction per customer
avg_spend_per_transaction_per_customer = total_spend_per_customer / num_transactions_per_customer

with col5:
    st.write(f"💵 **Top {top_n_customers} Customers by Avg Transaction Spend**")
    st.dataframe(
        avg_spend_per_transaction_per_customer.sort_values(ascending=False)
        .head(top_n_customers)
        .rename("Avg Spend per Transaction ($)")
    )

# ✅ Compute overall average of these values (i.e., mean of all avg transactions per customer)
overall_avg_spend_per_transaction = avg_spend_per_transaction_per_customer.mean()
# =================== Revenue Visualization =================== #
st.subheader("📈 Revenue Trends")

# Bar chart for revenue by category
fig_category_revenue = px.bar(
    revenue_by_category.reset_index(),
    x="Category", 
    y="PurchaseAmount",
    title="Total Revenue by Category",
    color="Category",
    text_auto=".2s"
)
st.plotly_chart(fig_category_revenue, use_container_width=True)

# Bar chart for revenue by top-selling products
fig_product_revenue = px.bar(
    revenue_by_product.reset_index(),
    x="ProductName", 
    y="PurchaseAmount",
    title="Top Products by Revenue",
    color="ProductName",
    text_auto=".2s"
)
st.plotly_chart(fig_product_revenue, use_container_width=True)