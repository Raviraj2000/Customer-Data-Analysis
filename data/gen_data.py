import csv
import random
from datetime import datetime, timedelta
from faker import Faker
import numpy as np

def generate_synthetic_data_structured(
    num_customers=500,
    num_products=50,
    num_records=8000,
    output_file='structured_purchase_history.csv'
):
    """
    Generates synthetic purchase data with built-in patterns to enable:
    - Customer segmentation
    - Product recommendations
    - Meaningful data analysis
    
    :param num_customers: Number of unique customers.
    :param num_products: Number of unique products.
    :param num_records: Total number of purchase records.
    :param output_file: CSV file to save the output.
    """
    fake = Faker('en_US')

    # -------------------------------------------------------
    # 1. Define Segments (customer archetypes)
    # -------------------------------------------------------
    # For demonstration, let's define four main segments:
    #   1. Tech Enthusiasts: High spend on Electronics, moderate on others.
    #   2. Bargain Hunters: Low spend across categories, but frequent small orders.
    #   3. Fashion Lovers: High spend on Clothing, Health & Beauty.
    #   4. General Shoppers: Balanced approach across categories.
    #
    # We also define each segment's typical spending range and
    # category preferences (expressed as weights).

    segments_definition = {
        'Tech Enthusiast': {
            'spend_range': (80, 500),     # High-value range
            'category_weights': {
                'Electronics': 5, 'Clothing': 1, 'Home & Kitchen': 1,
                'Sports & Outdoors': 1, 'Books': 1, 'Health & Beauty': 1,
                'Toys & Games': 1, 'Grocery': 1, 'Automotive': 1, 'Pet Supplies': 1
            }
        },
        'Bargain Hunter': {
            'spend_range': (5, 50),       # Lower spend range
            'category_weights': {
                'Electronics': 1, 'Clothing': 2, 'Home & Kitchen': 2,
                'Sports & Outdoors': 1, 'Books': 2, 'Health & Beauty': 1,
                'Toys & Games': 1, 'Grocery': 3, 'Automotive': 1, 'Pet Supplies': 2
            }
        },
        'Fashion Lover': {
            'spend_range': (20, 300),     # Medium-High spend
            'category_weights': {
                'Electronics': 1, 'Clothing': 4, 'Home & Kitchen': 1,
                'Sports & Outdoors': 1, 'Books': 1, 'Health & Beauty': 3,
                'Toys & Games': 1, 'Grocery': 2, 'Automotive': 1, 'Pet Supplies': 1
            }
        },
        'General Shopper': {
            'spend_range': (10, 200),     # Medium range
            'category_weights': {
                'Electronics': 2, 'Clothing': 2, 'Home & Kitchen': 2,
                'Sports & Outdoors': 2, 'Books': 2, 'Health & Beauty': 2,
                'Toys & Games': 2, 'Grocery': 2, 'Automotive': 1, 'Pet Supplies': 2
            }
        }
    }

    # We want to roughly distribute customers among these 4 segments.
    segment_distribution = {
        'Tech Enthusiast': 0.2,  # 20%
        'Bargain Hunter': 0.3,   # 30%
        'Fashion Lover': 0.2,    # 20%
        'General Shopper': 0.3   # 30%
    }
    
    # -------------------------------------------------------
    # 2. Assign each customer to a segment
    # -------------------------------------------------------
    # We'll create a list of segments for the 500 customers based on the distribution.
    # E.g. 20% Tech Enthusiasts, 30% Bargain Hunters, etc.
    segments_list = []
    for seg, frac in segment_distribution.items():
        count = int(frac * num_customers)
        segments_list.extend([seg] * count)
    # If rounding doesn't add to 500, fill the gap with 'General Shopper'
    while len(segments_list) < num_customers:
        segments_list.append('General Shopper')
    # Shuffle to randomize which customer gets which segment
    random.shuffle(segments_list)

    # Create a dictionary mapping customer -> segment
    # Also store some realistic demographic info with Faker
    customers = {}
    for cid in range(1, num_customers+1):
        seg_type = segments_list[cid-1]
        customers[cid] = {
            'Segment': seg_type,
            'Name': fake.name(),
            'Email': fake.email(),
            'Phone': fake.numerify('###-###-####'),
            'Address': fake.address().replace('\n', ', ')
        }

    # -------------------------------------------------------
    # 3. Define Product Categories & Build Weighted Pools
    # -------------------------------------------------------
    categories = list(segments_definition['Tech Enthusiast']['category_weights'].keys())
    # We'll create 50 products distributed among these 10 categories
    # We can do a simple approach: each product is assigned a random category
    # (or weighted by some global popularity)
    products = {}
    for pid in range(1, num_products+1):
        cat = random.choice(categories)
        # Generate a simple product name based on category
        prod_name = f"{fake.word().capitalize()} {cat.split()[-1]}"
        products[pid] = {
            'Name': prod_name,
            'Category': cat
        }

    # -------------------------------------------------------
    # 4. Generate Purchase Records
    # -------------------------------------------------------
    # We'll create 8,000 records (num_records).
    # We'll also incorporate a simple *time-based bias*:
    #   e.g., "Toys & Games" more popular in Nov/Dec,
    #         "Sports & Outdoors" more popular in summer (Jun-Aug).
    
    start_date = datetime(2023, 1, 1)
    end_date   = datetime(2025, 1, 1)
    date_range_days = (end_date - start_date).days

    def get_category_weight_by_month(category, purchase_month):
        """
        Increase or decrease category weight based on month
        for seasonality. Adjust to taste.
        Example:
          - Toys & Games: +50% in Nov/Dec
          - Sports & Outdoors: +50% in Jun/Jul/Aug
        """
        # Base multiplier
        multiplier = 1.0
        
        if category == 'Toys & Games':
            if purchase_month in [11, 12]:  # Nov, Dec
                multiplier = 1.5
        elif category == 'Sports & Outdoors':
            if purchase_month in [6, 7, 8]: # Jun, Jul, Aug
                multiplier = 1.5
        
        return multiplier

    # Prepare CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'CustomerID', 'CustomerName', 'Email', 'Phone', 'Address',
            'Segment', 'ProductID', 'ProductName', 'Category',
            'PurchaseAmount', 'PurchaseDate'
        ])

        for _ in range(num_records):
            # 1. Pick a random customer
            cid = random.randint(1, num_customers)
            seg_type = customers[cid]['Segment']
            
            # 2. Generate a random date (with uniform distribution across the range)
            random_offset = random.randint(0, date_range_days)
            purchase_date = start_date + timedelta(days=random_offset)
            purchase_month = purchase_date.month
            purchase_date_str = purchase_date.strftime('%Y-%m-%d')

            # 3. Choose a category based on segment’s category weights,
            #    but also factor in the month-based multiplier.
            seg_info = segments_definition[seg_type]
            base_weights = seg_info['category_weights']  # dict of category->weight
            # Build a new weighted list for categories
            weighted_cats = []
            for cat, w in base_weights.items():
                # Adjust with seasonality
                season_mult = get_category_weight_by_month(cat, purchase_month)
                w_adjusted = int(w * season_mult)
                weighted_cats.extend([cat] * w_adjusted)
            chosen_category = random.choice(weighted_cats)

            # 4. Within that chosen category, pick a random product 
            #    from the subset that belongs to that category.
            #    If none exist, fallback to random.
            cat_products = [pid for pid, pinfo in products.items() 
                            if pinfo['Category'] == chosen_category]
            if cat_products:
                chosen_pid = random.choice(cat_products)
            else:
                chosen_pid = random.randint(1, num_products)

            # 5. Spending range depends on segment
            low, high = seg_info['spend_range']
            purchase_amount = round(random.uniform(low, high), 2)

            # 6. Write to CSV
            prod_info = products[chosen_pid]
            writer.writerow([
                cid,
                customers[cid]['Name'],
                customers[cid]['Email'],
                customers[cid]['Phone'],
                customers[cid]['Address'],
                seg_type,
                chosen_pid,
                prod_info['Name'],
                prod_info['Category'],
                purchase_amount,
                purchase_date_str
            ])

    print(f"[INFO] Synthetic structured data saved to {output_file}")


if __name__ == '__main__':
    generate_synthetic_data_structured()
