"""
Persona-based Customer Data Generator V2

Generates synthetic customer profiles based on clustering results.
Reads dynamic persona names from labeled_reviews.csv (not hardcoded).
"""

import json
import os
import pandas as pd
import numpy as np


# Category Mapping
CATEGORIES = {
    97: "Skincare",
    98: "Makeup",
    99: "Perfume", 
    100: "Living",
    101: "Accessories",
    102: "Food",
    103: "Men",
    104: "Baby",
    105: "Device",
    106: "Pet"
}


def parse_profile_info(profile_str):
    """Parse profile string like '40대/남성/복합성/주름' into structured data."""
    if not profile_str or pd.isna(profile_str):
        return {'age': 30, 'gender': 'F', 'skin_type': 'normal', 'skin_concern': 'moisture'}
    
    parts = str(profile_str).split('/')
    
    # Age
    age = 30
    for p in parts:
        if '대' in p:
            try:
                age = int(p.replace('대', '')) + np.random.randint(0, 9)
            except:
                age = 30
            break
    
    # Gender
    gender = 'F'
    for p in parts:
        if '남성' in p:
            gender = 'M'
            break
    
    # Skin Type
    skin_type = 'normal'
    skin_type_kr_map = {
        '건성': 'dry', '지성': 'oily', '복합성': 'combination',
        '민감성': 'sensitive', '수분부족지성': 'oily', '중성': 'normal'
    }
    for p in parts:
        for kr, en in skin_type_kr_map.items():
            if kr in p:
                skin_type = en
                break
    
    # Skin Concern
    skin_concern = 'moisture'
    concern_kr_map = {
        '보습': 'moisture', '주름': 'wrinkle', '선물': 'gift',
        '재구매': 'repurchase', '진정': 'calming', '각질': 'texture'
    }
    for p in parts:
        for kr, en in concern_kr_map.items():
            if kr in p:
                skin_concern = en
                break
    
    return {'age': age, 'gender': gender, 'skin_type': skin_type, 'skin_concern': skin_concern}


def generate_customer_data():
    """
    Creates customers.json based on labeled_reviews.csv from clustering.
    Uses dynamic persona names instead of hardcoded ones.
    """
    
    # Load labeled reviews
    possible_paths = ['labeled_reviews.csv', '../labeled_reviews.csv', 'neural_recsys/labeled_reviews.csv']
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded: {path}")
            break
    if df is None:
        raise FileNotFoundError("labeled_reviews.csv not found! Run clustering.py first.")
    
    print(f"Total reviews: {len(df)}")
    
    customers = []
    
    for idx, row in df.iterrows():
        # Parse profile info for demographics
        profile = parse_profile_info(row.get('profile_info', ''))
        
        # Use persona_name from clustering if available
        persona_name = row.get('persona_name', f'Cluster_{row.get("cluster", 0)}')
        cluster_id = row.get('cluster', 0)
        
        customer = {
            "user_id": f"user_{idx:05d}",
            "persona_id": int(cluster_id),
            "persona_name": persona_name,
            "age": int(profile['age']),
            "gender": profile['gender'],
            "skin_type": profile['skin_type'],
            "skin_concern": profile['skin_concern'],
            "purchased_product_id": str(row.get('product_id', '')),
            "product_category": int(row.get('category', 97)),
            "product_category_name": CATEGORIES.get(int(row.get('category', 97)), "Unknown"),
            "review_content": str(row.get('content', ''))[:200]
        }
        customers.append(customer)
    
    # Ensure output directory exists
    os.makedirs("../data", exist_ok=True)
    
    # Save
    with open("../data/customers.json", "w", encoding="utf-8") as f:
        json.dump(customers, f, ensure_ascii=False, indent=2)
    
    print(f"\nGenerated {len(customers)} customer profiles → ../data/customers.json")
    
    # Summary
    df_cust = pd.DataFrame(customers)
    print("\n=== Persona Distribution ===")
    print(df_cust['persona_name'].value_counts())
    print("\n=== Skin Type Distribution ===")
    print(df_cust['skin_type'].value_counts())
    print("\n=== Category Distribution ===")
    print(df_cust['product_category_name'].value_counts())
    
    return customers


if __name__ == "__main__":
    generate_customer_data()
