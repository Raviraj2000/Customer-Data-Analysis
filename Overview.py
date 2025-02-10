import streamlit as st
import pandas as pd
import plotly.express as px
from utils.preprocess import load_data, extract_features, preprocess_data

# ---------------------------- #
# **Streamlit Page Configuration**
# ---------------------------- #
st.set_page_config(
    page_title="Customer Insights Dashboard",
    layout="wide",
)

st.title("📊 Customer Insights Dashboard")
st.write("Navigate to different sections using the sidebar.")

# ---------------------------- #
# **Load Cleaned Data**
# ---------------------------- #
df_cleaned = load_data("./data/raw/purchase_history.csv")

# ---------------------------- #
# **Dataset Overview KPIs**
# ---------------------------- #
st.subheader("📂 Dataset Overview")

# Compute dataset insights
total_rows = df_cleaned.shape[0]
total_columns = df_cleaned.shape[1]
total_missing_values = df_cleaned.isnull().sum().sum()
duplicate_rows = df_cleaned.duplicated().sum()

# Unique counts for key entities
total_customers_cleaned = df_cleaned["CustomerID"].nunique()
total_products_cleaned = df_cleaned["ProductID"].nunique()
total_transactions_cleaned = len(df_cleaned)
total_revenue_cleaned = df_cleaned["PurchaseAmount"].sum()

# Compute additional insights
avg_purchase_per_customer = total_revenue_cleaned / total_customers_cleaned
avg_transaction_value = total_revenue_cleaned / total_transactions_cleaned
max_transaction_value = df_cleaned["PurchaseAmount"].max()
min_transaction_value = df_cleaned["PurchaseAmount"].min()

# ---------------------------- #
# **Display Key Metrics in Blocks**
# ---------------------------- #
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("🗂️ **Total Rows**")
    st.metric(label=" ", value=f"{total_rows:,}")

with col2:
    st.markdown("📊 **Total Columns**")
    st.metric(label=" ", value=f"{total_columns:,}")

with col3:
    st.markdown("⚠️ **Missing Values**")
    st.metric(label=" ", value=f"{total_missing_values:,}")

with col4: 
    st.markdown("❌ **Duplicate Rows Removed**")
    st.metric(label=" ", value=f"{duplicate_rows:,}")

# ---------------------------- #
# **Additional Business Insights**
# ---------------------------- #
st.subheader("📊 Business Insights")

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.markdown("🛍️ **Unique Customers**")
    st.metric(label=" ", value=f"{total_customers_cleaned:,}")

with col6:
    st.markdown("📦 **Unique Products Sold**")
    st.metric(label=" ", value=f"{total_products_cleaned:,}")

with col7:
    st.markdown("📝 **Total Transactions**")
    st.metric(label=" ", value=f"{total_transactions_cleaned:,}")

with col8:
    st.markdown("💰 **Total Revenue**")
    st.metric(label=" ", value=f"${total_revenue_cleaned:,.2f}")

col9, col10, col11, col12 = st.columns(4)
with col9:
    st.markdown("💵 **Avg Purchase per Customer**")
    st.metric(label=" ", value=f"${avg_purchase_per_customer:,.2f}")

with col10:
    st.markdown("📊 **Avg Transaction Value**")
    st.metric(label=" ", value=f"${avg_transaction_value:,.2f}")

with col11:
    st.markdown("🔝 **Max Transaction Value**")
    st.metric(label=" ", value=f"${max_transaction_value:,.2f}")

with col12:
    st.markdown("🔻 **Min Transaction Value**")
    st.metric(label=" ", value=f"${min_transaction_value:,.2f}")

# ---------------------------- #
# **Scatter Plot: Total Spend vs. Purchase Count**
# ---------------------------- #
st.subheader("📈 Customer Spending vs. Purchase Frequency")

# Aggregate total spent and purchase count per customer
customer_spending = extract_features(df_cleaned)

# Scatter plot
fig_scatter = px.scatter(
    customer_spending,
    x="total_spent",
    y="purchase_count",
    title="Interactive Scatter Plot: Total Spending vs. Purchase Count",
    labels={"total_spent": "Total Spending", "purchase_count": "Purchase Count"},
    opacity=0.6
)

st.plotly_chart(fig_scatter, use_container_width=True)