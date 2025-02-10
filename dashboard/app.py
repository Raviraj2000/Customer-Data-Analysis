import streamlit as st
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Customer Insights Dashboard",
    layout="wide",
)

st.title("📊 Customer Insights Dashboard")
st.write("Navigate to different sections using the sidebar.")

# Load raw and cleaned data
df_raw = pd.read_csv("./data/raw/purchase_history.csv")
df_cleaned = pd.read_csv("./data/cleaned/purchase_history_cleaned.csv")

# **Comparison Metrics**
st.subheader("📈 Dataset Overview")

# Show dataset metrics
total_customers_cleaned = df_cleaned["CustomerID"].nunique()
total_products_cleaned = df_cleaned["ProductID"].nunique()
total_transactions_cleaned = len(df_cleaned)
total_revenue_cleaned = df_cleaned["PurchaseAmount"].sum()

# Display as columns for better UI
col1, col2, col3, col4 = st.columns(4)
col1.metric("🛍️ Customers", f"{total_customers_cleaned:,}")
col2.metric("📦 Products", f"{total_products_cleaned:,}")
col3.metric("📝 Transactions", f"{total_transactions_cleaned:,}")
col4.metric("💰 Revenue ($)", f"${total_revenue_cleaned:,.2f}")

# **Show Raw and Cleaned Data**
st.subheader("📄 Data Before and After Cleaning")

col5, col6 = st.columns(2)
with col5:
    st.write("🔴 **Raw Data Sample**")
    st.dataframe(df_raw.head())

with col6:
    st.write("✅ **Cleaned Data Sample**")
    st.dataframe(df_cleaned.head())

# ---------------------------- #
# **Segmented Aggregation Approach**
# ---------------------------- #
st.subheader("📊 Aggregated Insights by Spending Behavior")

# Select numeric columns (excluding IDs)
numeric_columns = [col for col in df_cleaned.select_dtypes(include=['int64', 'float64']).columns if "ID" not in col]

# Ensure dropdowns have valid defaults
if len(numeric_columns) < 2:
    st.error("Not enough numeric columns for analysis.")
else:
    # User selects the variable for binning (e.g., 'Total Spending')
    bin_column = st.selectbox("Select Variable for Segmentation", numeric_columns, index=0)

    # User selects the variable to aggregate
    agg_column = st.selectbox("Select Aggregation Metric", numeric_columns, index=1)

    # Number of bins for segmentation
    num_bins = st.slider("Select Number of Bins", min_value=3, max_value=20, value=10)

    # Create bins (Convert to string to avoid serialization issues)
    df_cleaned["Segment"] = pd.cut(df_cleaned[bin_column], bins=num_bins, precision=0).astype(str)

    # Aggregation metrics
    agg_df = df_cleaned.groupby("Segment").agg(
        total_customers=("CustomerID", "nunique"),
        avg_purchase_amount=(agg_column, "mean"),
        total_revenue=(agg_column, "sum")
    ).reset_index()

    # Scatter plot of aggregated insights
    fig = px.scatter(
        agg_df,
        x="Segment",
        y="total_revenue",
        size="total_customers",
        color="avg_purchase_amount",
        title=f"Segmented Analysis: {bin_column} vs {agg_column}",
        labels={"Segment": f"{bin_column} Range", "total_revenue": "Total Revenue", "avg_purchase_amount": "Avg Purchase Amount"},
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show aggregated data
    st.write("📊 **Aggregated Data**")
    st.dataframe(agg_df.style.format({
        "total_customers": "{:,}",
        "avg_purchase_amount": "${:,.2f}",
        "total_revenue": "${:,.2f}"
    }))
