import csv
import random
from datetime import datetime, timedelta
from faker import Faker
import numpy as np
import pandas as pd

def generate_synthetic_data_no_patterns(
    num_customers=1000,
    num_products=50,
    num_records=10000,
    output_file='data/raw/purchase_history.csv'
):
    """
    Generates synthetic purchase data without predefined patterns,
    ensuring distinct clusters emerge based on spending behavior.
    
    :param num_customers: Number of unique customers.
    :param num_products: Number of unique products.
    :param num_records: Total number of purchase records.
    :param output_file: CSV file to save the output.
    """
    fake = Faker('en_US')

    # -------------------------------------------------------
    # 1. Generate Customers with Random Spending Habits
    # -------------------------------------------------------
    customers = {}
    for cid in range(1, num_customers + 1):
        customers[cid] = {
            'spend_range': random.choice([(5, 100), (75, 200), (200, 500), (400, 700)])
        }

    # -------------------------------------------------------
    # 2. Generate Products
    # -------------------------------------------------------
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Sports & Outdoors', 
                  'Books', 'Health & Beauty', 'Toys & Games', 'Grocery', 'Automotive', 'Pet Supplies']

    products = {}
    for pid in range(1, num_products + 1):
        category = random.choice(categories)  # No predefined category bias
        product_name = f"{fake.word().capitalize()} {category.split()[-1]}"
        products[pid] = {
            'Name': product_name,
            'Category': category
        }

    # -------------------------------------------------------
    # 3. Generate Purchase Records
    # -------------------------------------------------------
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 1, 1)
    date_range_days = (end_date - start_date).days

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'CustomerID','ProductID', 'ProductName', 'Category', 'PurchaseAmount', 'PurchaseDate'
        ])

        for _ in range(num_records):
            # 1. Pick a random customer
            cid = random.randint(1, num_customers)
            spend_range = customers[cid]['spend_range']

            # 2. Generate a random date
            random_offset = random.randint(0, date_range_days)
            purchase_date = start_date + timedelta(days=random_offset)
            purchase_date_str = purchase_date.strftime('%Y-%m-%d')

            # 3. Pick a random product
            chosen_pid = random.randint(1, num_products)
            product_info = products[chosen_pid]

            # 4. Assign a random purchase amount within the customer's spending range
            purchase_amount = round(random.uniform(spend_range[0], spend_range[1]), 2)

            # 5. Write to CSV
            writer.writerow([
                cid,
                chosen_pid,
                product_info['Name'],
                product_info['Category'],
                purchase_amount,
                purchase_date_str
            ])

    df = pd.read_csv(output_file)
    df_products = df[['ProductID', 'ProductName', 'Category']].drop_duplicates()
    df_products.to_csv('./data/raw/products.csv', index=False)

    print(f"[INFO] Synthetic data saved to {output_file}")


if __name__ == '__main__':
    generate_synthetic_data_no_patterns()
