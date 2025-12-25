"""
Two-Tower Model with Structured Features

Customer Tower: age, gender, skin_type, skin_concern
Product Tower: category, brand, price
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomerProductModel(nn.Module):
    def __init__(self, 
                 num_skin_types=5,
                 num_skin_concerns=6,
                 num_genders=2,
                 num_categories=5,
                 num_brands=20,
                 embed_dim=16,
                 hidden_dim=64,
                 output_dim=32):
        
        super(CustomerProductModel, self).__init__()
        
        # Customer Tower
        self.skin_type_emb = nn.Embedding(num_skin_types, embed_dim)
        self.skin_concern_emb = nn.Embedding(num_skin_concerns, embed_dim)
        self.gender_emb = nn.Embedding(num_genders, embed_dim)
        
        customer_input_dim = embed_dim * 3 + 1
        self.customer_mlp = nn.Sequential(
            nn.Linear(customer_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Product Tower
        self.category_emb = nn.Embedding(num_categories, embed_dim)
        self.brand_emb = nn.Embedding(num_brands, embed_dim)
        
        product_input_dim = embed_dim * 2 + 1
        self.product_mlp = nn.Sequential(
            nn.Linear(product_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def encode_customer(self, skin_type, skin_concern, gender, age):
        st_emb = self.skin_type_emb(skin_type)
        sc_emb = self.skin_concern_emb(skin_concern)
        g_emb = self.gender_emb(gender)
        x = torch.cat([st_emb, sc_emb, g_emb, age.unsqueeze(1)], dim=1)
        x = self.customer_mlp(x)
        return F.normalize(x, p=2, dim=1)
    
    def encode_product(self, category, brand, price):
        cat_emb = self.category_emb(category)
        brand_emb = self.brand_emb(brand)
        x = torch.cat([cat_emb, brand_emb, price.unsqueeze(1)], dim=1)
        x = self.product_mlp(x)
        return F.normalize(x, p=2, dim=1)
    
    def forward(self, customer_data, product_data):
        cust_emb = self.encode_customer(
            customer_data['skin_type'],
            customer_data['skin_concern'],
            customer_data['gender'],
            customer_data['age']
        )
        prod_emb = self.encode_product(
            product_data['category'],
            product_data['brand'],
            product_data['price']
        )
        similarity = torch.sum(cust_emb * prod_emb, dim=1)
        return similarity, cust_emb, prod_emb


# Feature Mappings
SKIN_TYPE_MAP = {'dry': 0, 'oily': 1, 'combination': 2, 'sensitive': 3, 'normal': 4, 'mature': 3, 'unknown': 4}
SKIN_CONCERN_MAP = {'moisture': 0, 'wrinkle': 1, 'gift': 2, 'repurchase': 3, 'calming': 4, 'texture': 5}
GENDER_MAP = {'F': 0, 'M': 1}
CATEGORY_MAP = {97: 0, 98: 1, 99: 2, 101: 3, 103: 4}
