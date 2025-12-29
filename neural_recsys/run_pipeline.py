"""
Run Pipeline: Full Recommendation System Training on New Data

This script:
1. Transforms the new products.json to a compatible format
2. Generates customer profiles from review data
3. Trains the Two-Tower model
4. Evaluates performance (Hit@K)
5. Visualizes results (embedding space, persona distribution)
6. Saves all outputs to a versioned folder

Usage:
    cd neural_recsys
    python run_pipeline.py
"""

import json
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import Counter
from tqdm import tqdm

# ====================== CONFIGURATION ======================
DATA_PATH = "../data/products.json"  # New scraped data
OUTPUT_BASE = "../results"

# Category mapping (Korean -> Code)
CATEGORY_MAP_KR = {
    "스킨케어": 97,
    "메이크업": 98,
    "향수": 99,
    "생활용품": 100,
    "소품&도구": 101,
    "뷰티푸드": 102,
    "남성": 103,
    "베이비": 104,
    "뷰티디바이스": 105,
    "반려동물용품": 106,
}

# Inverse for display
CATEGORY_NAMES = {v: k for k, v in CATEGORY_MAP_KR.items()}

# Existing model mappings (from model.py)
SKIN_TYPE_MAP = {'dry': 0, 'oily': 1, 'combination': 2, 'sensitive': 3, 'normal': 4}
SKIN_CONCERN_MAP = {'moisture': 0, 'wrinkle': 1, 'gift': 2, 'repurchase': 3, 'calming': 4, 'texture': 5}
GENDER_MAP = {'F': 0, 'M': 1}
CATEGORY_MAP = {97: 0, 98: 1, 99: 2, 100: 3, 101: 4, 102: 5, 103: 6, 104: 7, 105: 8, 106: 9}


# ====================== MODEL DEFINITION ======================
class CustomerProductModel(nn.Module):
    def __init__(self, num_brands, embed_dim=16, hidden_dim=64, output_dim=32):
        super().__init__()
        self.skin_type_emb = nn.Embedding(5, embed_dim)
        self.skin_concern_emb = nn.Embedding(6, embed_dim)
        self.gender_emb = nn.Embedding(2, embed_dim)
        self.category_emb = nn.Embedding(10, embed_dim)
        self.brand_emb = nn.Embedding(num_brands + 1, embed_dim)
        
        # Customer MLP: 3 embeddings (16*3) + age (1) = 49
        self.customer_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3 + 1, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Product MLP: 2 embeddings (16*2) + price (1) = 33
        self.product_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def encode_customer(self, skin_type, skin_concern, gender, age):
        st = self.skin_type_emb(skin_type)
        sc = self.skin_concern_emb(skin_concern)
        g = self.gender_emb(gender)
        a = age.unsqueeze(-1)
        x = torch.cat([st, sc, g, a], dim=-1)
        x = self.customer_mlp(x)
        return nn.functional.normalize(x, p=2, dim=-1)
    
    def encode_product(self, category, brand, price):
        c = self.category_emb(category)
        b = self.brand_emb(brand)
        p = price.unsqueeze(-1)
        x = torch.cat([c, b, p], dim=-1)
        x = self.product_mlp(x)
        return nn.functional.normalize(x, p=2, dim=-1)


# ====================== DATASET ======================
class CustomerProductDataset(Dataset):
    def __init__(self, customers, products, brand_map):
        self.customers = customers
        self.products = products
        self.brand_map = brand_map
        self.pid_to_prod = {p['product_id']: p for p in products}
        
    def __len__(self):
        return len(self.customers)
    
    def __getitem__(self, idx):
        cust = self.customers[idx]
        pos_prod = self.pid_to_prod.get(cust['purchased_product_id'], random.choice(self.products))
        neg_prod = random.choice(self.products)
        while neg_prod['product_id'] == pos_prod['product_id']:
            neg_prod = random.choice(self.products)
        
        return {
            'cust_skin_type': SKIN_TYPE_MAP.get(cust.get('skin_type', 'normal'), 4),
            'cust_skin_concern': SKIN_CONCERN_MAP.get(cust.get('skin_concern', 'moisture'), 0),
            'cust_gender': GENDER_MAP.get(cust.get('gender', 'F'), 0),
            'cust_age': cust.get('age', 30) / 50.0,
            'pos_category': CATEGORY_MAP.get(pos_prod.get('category', 97), 0),
            'pos_brand': self.brand_map.get(pos_prod.get('brand_name', ''), 0),
            'pos_price': min(pos_prod.get('price', 50000) / 200000.0, 1.0),
            'neg_category': CATEGORY_MAP.get(neg_prod.get('category', 97), 0),
            'neg_brand': self.brand_map.get(neg_prod.get('brand_name', ''), 0),
            'neg_price': min(neg_prod.get('price', 50000) / 200000.0, 1.0),
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


# ====================== DATA TRANSFORMATION ======================
def parse_profile_info(profile_str):
    """
    Parse profile string like "40대/남성/복합성/주름" into structured data.
    """
    if not profile_str:
        return {'age': 30, 'gender': 'F', 'skin_type': 'normal', 'skin_concern': 'moisture'}
    
    parts = profile_str.split('/')
    
    # Age
    age = 30
    for p in parts:
        if '대' in p:
            try:
                age = int(p.replace('대', '')) + random.randint(0, 9)
            except:
                age = 30
            break
    
    # Gender
    gender = 'F'
    for p in parts:
        if '남성' in p:
            gender = 'M'
            break
        elif '여성' in p:
            gender = 'F'
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


def transform_data(products_raw):
    """
    Transform new products.json format to compatible format.
    Also generate customer data from reviews.
    """
    products = []
    customers = []
    
    # Extract brand names for encoding
    brand_names = set()
    for prod in products_raw:
        # Extract brand from name (first word often)
        name = prod.get('name', '')
        # Try to find brand in brackets like [설화수]
        if '[' in name and ']' in name:
            brand = name.split('[')[1].split(']')[0]
        else:
            brand = name.split()[0] if name else 'Unknown'
        brand_names.add(brand)
    
    brand_map = {b: i for i, b in enumerate(brand_names)}
    
    customer_idx = 0
    for prod in products_raw:
        # Transform product
        name = prod.get('name', '')
        if '[' in name and ']' in name:
            brand = name.split('[')[1].split(']')[0]
        else:
            brand = name.split()[0] if name else 'Unknown'
        
        category_kr = prod.get('category', '스킨케어')
        category_code = CATEGORY_MAP_KR.get(category_kr, 97)
        
        price = prod.get('price', '50000')
        if isinstance(price, str):
            price = int(price.replace(',', '')) if price else 50000
        
        transformed_prod = {
            'product_id': prod.get('product_id', ''),
            'product_name': name,
            'price': price,
            'brand_name': brand,
            'category': category_code,
            'sub_category': prod.get('sub_category', ''),
        }
        products.append(transformed_prod)
        
        # Generate customers from reviews
        for review in prod.get('reviews', []):
            profile = parse_profile_info(review.get('profile_info', ''))
            
            customer = {
                'user_id': f'user_{customer_idx:05d}',
                'age': profile['age'],
                'gender': profile['gender'],
                'skin_type': profile['skin_type'],
                'skin_concern': profile['skin_concern'],
                'purchased_product_id': prod.get('product_id', ''),
                'product_category': category_code,
                'review_text': review.get('text', '')[:200],
            }
            customers.append(customer)
            customer_idx += 1
    
    return products, customers, brand_map


# ====================== TRAINING ======================
def train_model(customers, products, brand_map, epochs=500):
    print("=" * 50)
    print("TRAINING: Two-Tower Model")
    print("=" * 50)
    print(f"Customers: {len(customers)}")
    print(f"Products: {len(products)}")
    print(f"Brands: {len(brand_map)}")
    
    dataset = CustomerProductDataset(customers, products, brand_map)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    
    model = CustomerProductModel(num_brands=len(brand_map))
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    losses = []
    
    pbar = tqdm(range(epochs), desc="Training", unit="epoch")
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
        losses.append(avg_loss)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return model, brand_map, losses


# ====================== EVALUATION ======================
def evaluate_model(model, customers, products, brand_map, top_k=3):
    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)
    
    model.eval()
    
    # Pre-compute product embeddings
    prod_embeddings = []
    prod_ids = []
    prod_categories = []
    
    with torch.no_grad():
        for prod in products:
            cat = CATEGORY_MAP.get(prod.get('category', 97), 0)
            brand = brand_map.get(prod.get('brand_name', ''), 0)
            price = min(prod.get('price', 50000) / 200000.0, 1.0)
            
            emb = model.encode_product(
                torch.tensor([cat]),
                torch.tensor([brand]),
                torch.tensor([price])
            ).numpy()[0]
            
            prod_embeddings.append(emb)
            prod_ids.append(prod['product_id'])
            prod_categories.append(prod.get('category', 97))
    
    prod_embeddings = np.array(prod_embeddings)
    
    # Evaluate each customer
    hits = 0
    category_hits = 0
    results = []
    
    for cust in customers:
        st = SKIN_TYPE_MAP.get(cust.get('skin_type', 'normal'), 4)
        sc = SKIN_CONCERN_MAP.get(cust.get('skin_concern', 'moisture'), 0)
        g = GENDER_MAP.get(cust.get('gender', 'F'), 0)
        age = cust.get('age', 30) / 50.0
        
        with torch.no_grad():
            cust_emb = model.encode_customer(
                torch.tensor([st]),
                torch.tensor([sc]),
                torch.tensor([g]),
                torch.tensor([age])
            ).numpy()[0]
        
        # Compute similarities
        scores = prod_embeddings @ cust_emb
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_pids = [prod_ids[i] for i in top_indices]
        top_cats = [prod_categories[i] for i in top_indices]
        
        purchased_pid = cust.get('purchased_product_id', '')
        purchased_cat = cust.get('product_category', 97)
        
        hit = purchased_pid in top_pids
        cat_hit = purchased_cat in top_cats
        
        if hit:
            hits += 1
        if cat_hit:
            category_hits += 1
        
        results.append({
            'user_id': cust.get('user_id', ''),
            'skin_type': cust.get('skin_type', ''),
            'purchased_product': purchased_pid,
            'hit': hit,
            'category_hit': cat_hit,
        })
    
    hit_rate = hits / len(customers) * 100
    cat_hit_rate = category_hits / len(customers) * 100
    
    print(f"Hit@{top_k}: {hit_rate:.1f}%")
    print(f"Category Hit@{top_k}: {cat_hit_rate:.1f}%")
    
    metrics = {
        'total_customers': len(customers),
        'total_products': len(products),
        f'hit_at_{top_k}': hit_rate,
        f'category_hit_at_{top_k}': cat_hit_rate,
    }
    
    return metrics, results


# ====================== VISUALIZATION ======================
def visualize_results(model, customers, products, brand_map, output_dir):
    print("\n" + "=" * 50)
    print("VISUALIZATION")
    print("=" * 50)
    
    model.eval()
    
    # Pre-compute product embeddings
    print("Computing product embeddings...")
    prod_embeddings = []
    prod_categories = []
    
    with torch.no_grad():
        for prod in products:
            cat = CATEGORY_MAP.get(prod.get('category', 97), 0)
            brand = brand_map.get(prod.get('brand_name', ''), 0)
            price = min(prod.get('price', 50000) / 200000.0, 1.0)
            emb = model.encode_product(
                torch.tensor([cat]), torch.tensor([brand]), torch.tensor([price])
            ).numpy()[0]
            prod_embeddings.append(emb)
            prod_categories.append(CATEGORY_NAMES.get(prod.get('category', 97), 'Unknown'))
    
    prod_embeddings = np.array(prod_embeddings)
    
    # 1. IMPROVED Embedding Space (Customers + Products)
    print("Creating combined embedding space visualization...")
    all_embeddings = []
    all_labels = []
    all_types = []  # 'customer' or 'product'
    
    # Add customer embeddings (sample for speed)
    with torch.no_grad():
        for cust in customers[:300]:
            st = SKIN_TYPE_MAP.get(cust.get('skin_type', 'normal'), 4)
            sc = SKIN_CONCERN_MAP.get(cust.get('skin_concern', 'moisture'), 0)
            g = GENDER_MAP.get(cust.get('gender', 'F'), 0)
            age = cust.get('age', 30) / 50.0
            emb = model.encode_customer(
                torch.tensor([st]), torch.tensor([sc]), torch.tensor([g]), torch.tensor([age])
            ).numpy()[0]
            all_embeddings.append(emb)
            all_labels.append(cust.get('skin_type', 'normal'))
            all_types.append('customer')
    
    # Add product embeddings (sample for visibility)
    for i, prod in enumerate(products[:200]):
        all_embeddings.append(prod_embeddings[i])
        all_labels.append(prod_categories[i])
        all_types.append('product')
    
    all_embeddings = np.array(all_embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) - 1))
    embedded = tsne.fit_transform(all_embeddings)
    
    # Create plot with different markers
    plt.figure(figsize=(14, 10))
    
    # Plot customers (circles)
    customer_mask = np.array([t == 'customer' for t in all_types])
    customer_labels = [all_labels[i] for i in range(len(all_labels)) if all_types[i] == 'customer']
    unique_skin_types = list(set(customer_labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_skin_types)))
    
    for i, skin_type in enumerate(unique_skin_types):
        mask = np.array([customer_mask[j] and all_labels[j] == skin_type for j in range(len(all_labels))])
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                   c=[colors[i]], label=f'Customer: {skin_type}', alpha=0.5, s=30, marker='o')
    
    # Plot products (stars)
    product_mask = np.array([t == 'product' for t in all_types])
    product_labels = [all_labels[i] for i in range(len(all_labels)) if all_types[i] == 'product']
    unique_categories = list(set(product_labels))
    cat_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
    
    for i, cat in enumerate(unique_categories):
        mask = np.array([product_mask[j] and all_labels[j] == cat for j in range(len(all_labels))])
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                   c=[cat_colors[i]], label=f'Product: {cat}', alpha=0.8, s=100, marker='*')
    
    plt.title('Customer & Product Embedding Space (t-SNE)', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_space.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: embedding_space.png")
    
    # 2. Persona (Skin Type) Distribution
    print("Creating persona distribution...")
    skin_types = [c.get('skin_type', 'normal') for c in customers]
    counts = Counter(skin_types)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(counts.keys(), counts.values(), color=plt.cm.Pastel2.colors[:len(counts)])
    plt.title('Customer Skin Type Distribution', fontsize=14)
    plt.xlabel('Skin Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{int(bar.get_height())}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'persona_distribution.png'), dpi=150)
    plt.close()
    print(f"Saved: persona_distribution.png")
    
    # 3. Results by Persona (Skin Type)
    print("Creating results by persona...")
    prod_ids = [p['product_id'] for p in products]
    
    persona_hits = {}
    persona_cat_hits = {}
    for cust in customers:
        st = SKIN_TYPE_MAP.get(cust.get('skin_type', 'normal'), 4)
        sc = SKIN_CONCERN_MAP.get(cust.get('skin_concern', 'moisture'), 0)
        g = GENDER_MAP.get(cust.get('gender', 'F'), 0)
        age = cust.get('age', 30) / 50.0
        skin_type = cust.get('skin_type', 'normal')
        
        with torch.no_grad():
            cust_emb = model.encode_customer(
                torch.tensor([st]), torch.tensor([sc]), torch.tensor([g]), torch.tensor([age])
            ).numpy()[0]
        
        scores = prod_embeddings @ cust_emb
        top_indices = np.argsort(scores)[::-1][:3]
        top_pids = [prod_ids[i] for i in top_indices]
        top_cats = [products[i].get('category', 97) for i in top_indices]
        
        hit = cust.get('purchased_product_id', '') in top_pids
        cat_hit = cust.get('product_category', 97) in top_cats
        
        if skin_type not in persona_hits:
            persona_hits[skin_type] = {'hits': 0, 'total': 0}
            persona_cat_hits[skin_type] = {'hits': 0, 'total': 0}
        persona_hits[skin_type]['total'] += 1
        persona_cat_hits[skin_type]['total'] += 1
        if hit:
            persona_hits[skin_type]['hits'] += 1
        if cat_hit:
            persona_cat_hits[skin_type]['hits'] += 1
    
    personas = list(persona_hits.keys())
    hit_rates = [persona_hits[p]['hits'] / persona_hits[p]['total'] * 100 for p in personas]
    cat_hit_rates = [persona_cat_hits[p]['hits'] / persona_cat_hits[p]['total'] * 100 for p in personas]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(personas))
    width = 0.35
    bars1 = ax.bar(x - width/2, hit_rates, width, label='Hit@3', color='#5DA5DA')
    bars2 = ax.bar(x + width/2, cat_hit_rates, width, label='Category Hit@3', color='#FAA43A')
    
    ax.set_title('Performance by Skin Type', fontsize=14)
    ax.set_xlabel('Skin Type')
    ax.set_ylabel('Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(personas, rotation=45)
    ax.legend()
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{bar.get_height():.1f}%', ha='center', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{bar.get_height():.1f}%', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results_by_persona.png'), dpi=150)
    plt.close()
    print(f"Saved: results_by_persona.png")
    
    # 4. Generate PERSONA_DEFINITION.md
    print("Creating PERSONA_DEFINITION.md...")
    skin_type_counts = Counter([c.get('skin_type', 'normal') for c in customers])
    total = len(customers)
    
    persona_md = f"""# Persona Definition Log

**생성일**: {datetime.now().strftime('%Y-%m-%d')}  
**데이터 기반**: `products.json` ({len(products)}개 제품, {len(customers)}개 리뷰)  
**방법론**: 리뷰 프로필 정보 기반 자동 분류 (나이/성별/피부타입/피부고민)

---

## 1. 피부타입별 분포

| Skin Type | 샘플 수 | 비율 |
|-----------|--------|------|
"""
    for skin_type, count in skin_type_counts.most_common():
        ratio = count / total * 100
        persona_md += f"| {skin_type} | {count} | {ratio:.1f}% |\n"
    
    persona_md += """
---

## 2. 피부타입 정의

"""
    skin_type_info = {
        'dry': ('건성', '보습 필수, 수분감 중시'),
        'oily': ('지성', '피지 컨트롤, 가벼운 제형 선호'),
        'combination': ('복합성', 'T존/U존 다른 케어 필요'),
        'sensitive': ('민감성', '저자극, 진정 효과 중시'),
        'normal': ('중성', '균형 잡힌 피부, 다양한 제형 수용'),
    }
    
    for skin_type, count in skin_type_counts.most_common():
        kr_name, desc = skin_type_info.get(skin_type, (skin_type, ''))
        hit_rate = persona_hits.get(skin_type, {}).get('hits', 0) / max(persona_hits.get(skin_type, {}).get('total', 1), 1) * 100
        cat_hit_rate = persona_cat_hits.get(skin_type, {}).get('hits', 0) / max(persona_cat_hits.get(skin_type, {}).get('total', 1), 1) * 100
        
        persona_md += f"""### `{skin_type}` ({kr_name})
- **샘플 수**: {count}명 ({count/total*100:.1f}%)
- **특징**: {desc}
- **Hit@3**: {hit_rate:.1f}%
- **Category Hit@3**: {cat_hit_rate:.1f}%

"""
    
    persona_md += """---

## 3. 데이터 특성

- **나이**: 리뷰 프로필에서 자동 추출 (예: "40대" → 40~49세 랜덤)
- **성별**: 프로필 내 "남성/여성" 키워드로 분류
- **피부고민**: 보습/주름/선물/재구매/진정/각질 자동 매핑

---

## 4. 활용 방안

- **추천 시스템**: 신규 유저 프로필 입력 → 피부타입 기반 상품 추천
- **마케팅**: 피부타입별 타겟 메시지 (예: 건성에게는 "깊은 보습" 강조)
- **상품 기획**: 피부타입별 선호 카테고리 분석
"""
    
    with open(os.path.join(output_dir, 'PERSONA_DEFINITION.md'), 'w', encoding='utf-8') as f:
        f.write(persona_md)
    print(f"Saved: PERSONA_DEFINITION.md")


# ====================== MAIN ======================
def main():
    # Create versioned output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir}")
    
    # Load data
    print("\n" + "=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        products_raw = json.load(f)
    print(f"Loaded {len(products_raw)} products from {DATA_PATH}")
    
    # Transform data
    print("\nTransforming data...")
    products, customers, brand_map = transform_data(products_raw)
    print(f"Transformed: {len(products)} products, {len(customers)} customers")
    
    # Save transformed data
    with open(os.path.join(output_dir, 'customers.json'), 'w', encoding='utf-8') as f:
        json.dump(customers, f, ensure_ascii=False, indent=2)
    print(f"Saved: customers.json")
    
    # Train
    model, brand_map, losses = train_model(customers, products, brand_map, epochs=500)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'brand_map': brand_map
    }, os.path.join(output_dir, 'model.pth'))
    print(f"\nSaved: model.pth")
    
    # Evaluate
    metrics, results = evaluate_model(model, customers, products, brand_map, top_k=3)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: metrics.json")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    print(f"Saved: results.csv")
    
    # Visualize
    visualize_results(model, customers, products, brand_map, output_dir)
    
    # Save config
    config = {
        'data_path': DATA_PATH,
        'timestamp': timestamp,
        'num_products': len(products),
        'num_customers': len(customers),
        'num_brands': len(brand_map),
        'epochs': 500,
        'batch_size': 64,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    main()
