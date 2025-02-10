# 📊 Customer Insights Dashboard

A comprehensive customer data analysis dashboard built with Python and Streamlit, providing interactive visualizations and detailed insights into customer behavior, purchase patterns, and business metrics.

## 🚀 Features

- **Dataset Overview**: Track key metrics including total records, data quality indicators, and duplicates
- **Business Insights**: Monitor critical KPIs such as:
  - Total revenue and unique customers
  - Average purchase per customer
  - Transaction value analysis
  - Product performance metrics
- **Interactive Dashboard**: Built with Streamlit for real-time data exploration
- **Data Quality Monitoring**: Track missing values and data integrity

## 🛠️ Technology Stack

- Python 3.10
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- Seaborn
- YData Profiling

## 📁 Project Structure

```plaintext
Customer-Data-Analysis/
├── Overview.py          # Main dashboard application
├── data/               # Data directory
│   └── raw/           # Raw data files
├── pages/             # Additional dashboard pages
├── utils/             # Utility functions
└── requirements.txt   # Project dependencies
```

## ⚡️ Quick Start

### 1. Clone the Repository
```bash
# Clone this repository
git clone https://github.com/Raviraj2000/Customer-Data-Analysis.git

# Navigate to project directory
cd Customer-Data-Analysis
```
### 2. Run Streamlit Application
```bash
# Run the app
streamlit run Overview.py
```