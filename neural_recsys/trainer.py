"""
Trainer: Structured Feature Training Pipeline
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random

from model import CustomerProductModel, SKIN_TYPE_MAP, SKIN_CONCERN_MAP, GENDER_MAP, CATEGORY_MAP


class CustomerProductDataset(Dataset):
    def __init__(self, customers, products_df, brand_map):
        self.customers = customers
        self.products = products_df.to_dict('records')
        self.brand_map = brand_map
        self.pid_to_prod = {p['product_id']: p for p in self.products}
        
    def __len__(self):
        return len(self.customers)
    
    def _encode_customer(self, cust):
        return {
            'skin_type': SKIN_TYPE_MAP.get(cust['skin_type'], 4),
            'skin_concern': SKIN_CONCERN_MAP.get(cust['skin_concern'], 0),
            'gender': GENDER_MAP.get(cust['gender'], 0),
            'age': cust['age'] / 50.0
        }
    
    def _encode_product(self, prod):
        return {
            'category': CATEGORY_MAP.get(prod['category'], 0),
            'brand': self.brand_map.get(prod['brand_name'], 0),
            'price': min(prod['price'] / 200000.0, 1.0)
        }
    
    def __getitem__(self, idx):
        cust = self.customers[idx]
        pos_prod = self.pid_to_prod.get(cust['purchased_product_id'], random.choice(self.products))
        neg_prod = random.choice(self.products)
        while neg_prod['product_id'] == pos_prod['product_id']:
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
    print("TRAINING: Structured Feature Model")
    print("="*50)
    
    with open("data/customers.json", "r", encoding="utf-8") as f:
        customers = json.load(f)
    print(f"Loaded {len(customers)} customers")
    
    with open("data/amoremall_reviews.json", "r", encoding="utf-8") as f:
        products_raw = json.load(f)
    products_df = pd.DataFrame(products_raw)
    print(f"Loaded {len(products_df)} products")
    
    unique_brands = products_df['brand_name'].unique()
    brand_map = {b: i for i, b in enumerate(unique_brands)}
    
    dataset = CustomerProductDataset(customers, products_df, brand_map)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    model = CustomerProductModel(num_brands=len(brand_map))
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    print("\nTraining (500 Epochs)...")
    
    for epoch in range(500):
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
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/500 | Loss: {total_loss/len(dataloader):.4f}")
    
    torch.save({'model_state_dict': model.state_dict(), 'brand_map': brand_map}, "model.pth")
    print("\nSaved: model.pth")
    return model, brand_map


if __name__ == "__main__":
    train_model()
