"""
Trainer V2: Structured Feature Training Pipeline for New Data Schema

Adaptations:
1. Loads data/products.json instead of amoremall_reviews.json
2. Handles numeric/string Price values robustly
3. Maps Korean category names to integer codes
4. Uses tqdm for progress tracking
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import re
from tqdm import tqdm

from model import CustomerProductModel, SKIN_TYPE_MAP, SKIN_CONCERN_MAP, GENDER_MAP, CATEGORY_MAP

# Extended Category Map for Korean names
CATEGORY_MAP_KR = {
    "스킨케어": 97, "메이크업": 98, "향수": 99, "생활용품": 100,
    "소품&도구": 101, "뷰티푸드": 102, "남성": 103, "베이비": 104,
    "뷰티디바이스": 105, "반려동물용품": 106,
}

def parse_price(price_val):
    if isinstance(price_val, (int, float)):
        return float(price_val)
    if isinstance(price_val, str):
        # Remove "원", ",", " " etc.
        clean = re.sub(r'[^\d]', '', price_val)
        if clean:
            return float(clean)
    return 50000.0  # Default


class CustomerProductDataset(Dataset):
    def __init__(self, customers, products, brand_map):
        self.customers = customers
        self.products = products
        self.brand_map = brand_map
        self.pid_to_prod = {p['product_id']: p for p in self.products}
        
    def __len__(self):
        return len(self.customers)
    
    def _encode_customer(self, cust):
        return {
            'skin_type': SKIN_TYPE_MAP.get(cust.get('skin_type', 'normal'), 4),
            'skin_concern': SKIN_CONCERN_MAP.get(cust.get('skin_concern', 'moisture'), 0),
            'gender': GENDER_MAP.get(cust.get('gender', 'F'), 0),
            'age': cust.get('age', 30) / 50.0
        }
    
    def _encode_product(self, prod):
        cat_val = prod.get('category', 97)
        # Handle string category
        if isinstance(cat_val, str):
            cat_code = CATEGORY_MAP_KR.get(cat_val, 97)
        else:
            cat_code = int(cat_val)
            
        return {
            'category': CATEGORY_MAP.get(cat_code, 0),
            'brand': self.brand_map.get(prod.get('brand_name', ''), 0),
            'price': min(parse_price(prod.get('price', 0)) / 200000.0, 1.0)
        }
    
    def __getitem__(self, idx):
        cust = self.customers[idx]
        pos_pid = cust.get('purchased_product_id', '')
        pos_prod = self.pid_to_prod.get(pos_pid, random.choice(self.products))
        
        neg_prod = random.choice(self.products)
        while neg_prod.get('product_id') == pos_prod.get('product_id'):
            neg_prod = random.choice(self.products)
        
        cust_enc = self._encode_customer(cust)
        pos_enc = self._encode_product(pos_prod)
        neg_enc = self._encode_product(neg_prod)
        
        return {
            'cust_skin_type': cust_enc['skin_type'],
            'cust_skin_concern': cust_enc['skin_concern'],
            'cust_gender': cust_enc['gender'],
            'cust_age': cust_enc['age'],
            'pos_category': pos_enc['category'],
            'pos_brand': pos_enc['brand'],
            'pos_price': pos_enc['price'],
            'neg_category': neg_enc['category'],
            'neg_brand': neg_enc['brand'],
            'neg_price': neg_enc['price'],
        }


def collate_fn(batch):
    return {
        'customer': {
            'skin_type': torch.tensor([b['cust_skin_type'] for b in batch], dtype=torch.long),
            'skin_concern': torch.tensor([b['cust_skin_concern'] for b in batch], dtype=torch.long),
            'gender': torch.tensor([b['cust_gender'] for b in batch], dtype=torch.long),
            'age': torch.tensor([b['cust_age'] for b in batch], dtype=torch.float32),
        },
        'positive': {
            'category': torch.tensor([b['pos_category'] for b in batch], dtype=torch.long),
            'brand': torch.tensor([b['pos_brand'] for b in batch], dtype=torch.long),
            'price': torch.tensor([b['pos_price'] for b in batch], dtype=torch.float32),
        },
        'negative': {
            'category': torch.tensor([b['neg_category'] for b in batch], dtype=torch.long),
            'brand': torch.tensor([b['neg_brand'] for b in batch], dtype=torch.long),
            'price': torch.tensor([b['neg_price'] for b in batch], dtype=torch.float32),
        }
    }


def train_model():
    print("="*50)
    print("TRAINING V2: Structured Feature Model")
    print("="*50)
    
    # 1. Load Data
    # Updated paths for the new pipeline structure
    data_path_customers = "../data/customers.json"
    data_path_products = "../data/products.json"
    
    # Fallback to local if running inside data folder (should not happen in this setup but safe to check)
    import os
    if not os.path.exists(data_path_customers):
         data_path_customers = "data/customers.json"
    if not os.path.exists(data_path_products):
         data_path_products = "data/products.json"

    print(f"Loading customers from: {data_path_customers}")
    with open(data_path_customers, "r", encoding="utf-8") as f:
        customers = json.load(f)
    print(f"Loaded {len(customers)} customers")
    
    print(f"Loading products from: {data_path_products}")
    with open(data_path_products, "r", encoding="utf-8") as f:
        products = json.load(f)
    print(f"Loaded {len(products)} products")
    
    # 2. Prepare Maps
    brands = set(p.get('brand_name', '') for p in products)
    brand_map = {b: i for i, b in enumerate(sorted(list(brands)))}
    print(f"Found {len(brand_map)} unique brands")
    
    # 3. Setup Training
    dataset = CustomerProductDataset(customers, products, brand_map)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    model = CustomerProductModel(num_brands=len(brand_map))
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    print("\nTraining (500 Epochs)...")
    
    epochs = 500
    pbar = tqdm(range(epochs))
    
    for epoch in pbar:
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            cust_emb = model.encode_customer(
                batch['customer']['skin_type'],
                batch['customer']['skin_concern'],
                batch['customer']['gender'],
                batch['customer']['age']
            )
            pos_emb = model.encode_product(
                batch['positive']['category'],
                batch['positive']['brand'],
                batch['positive']['price']
            )
            neg_emb = model.encode_product(
                batch['negative']['category'],
                batch['negative']['brand'],
                batch['negative']['price']
            )
            loss = criterion(cust_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        pbar.set_description(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    
    # 4. Save
    torch.save({'model_state_dict': model.state_dict(), 'brand_map': brand_map}, "model.pth")
    print("\nSaved: model.pth")
    
    # Also save to results dir if it exists
    # Find latest results dir
    try:
        results_base = "../results"
        dirs = [d for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d))]
        if dirs:
            latest_dir = os.path.join(results_base, sorted(dirs)[-1])
            torch.save({'model_state_dict': model.state_dict(), 'brand_map': brand_map}, os.path.join(latest_dir, "model.pth"))
            print(f"Saved copy to: {latest_dir}/model.pth")
    except Exception as e:
        print(f"Could not save copy to results dir: {e}")
        
    return model, brand_map


if __name__ == "__main__":
    train_model()
