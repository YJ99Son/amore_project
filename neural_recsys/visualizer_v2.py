"""
Visualizer V2: Dynamic Persona Visualization & Definition Generation

1. Loads trained model and new data
2. Generates embedding space visualization (Customers + Products)
3. Generates Persona Distribution charts
4. Generates PERSONA_DEFINITION.md dynamically
"""

import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import Counter
import os
from datetime import datetime
import re

from model import CustomerProductModel, SKIN_TYPE_MAP, SKIN_CONCERN_MAP, GENDER_MAP, CATEGORY_MAP

# Extended Category Map
CATEGORY_MAP_KR = {
    "스킨케어": 97, "메이크업": 98, "향수": 99, "생활용품": 100,
    "소품&도구": 101, "뷰티푸드": 102, "남성": 103, "베이비": 104,
    "뷰티디바이스": 105, "반려동물용품": 106,
}
# Inverse for display
CATEGORY_NAMES = {v: k for k, v in CATEGORY_MAP_KR.items()}
CATEGORY_NAMES.update({97: "Skincare", 98: "Makeup", 99: "Perfume", 103: "Men"}) # English defaults

def parse_price(price_val):
    if isinstance(price_val, (int, float)):
        return float(price_val)
    if isinstance(price_val, str):
        clean = re.sub(r'[^\d]', '', price_val)
        if clean:
            return float(clean)
    return 50000.0

def load_all():
    """Load model, customers, products"""
    # Find latest results dir for updated files
    results_base = "../results"
    try:
        dirs = [d for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d))]
        latest_dir = os.path.join(results_base, sorted(dirs)[-1])
        print(f"Using latest results dir: {latest_dir}")
    except:
        latest_dir = "."
        
    data_path_customers = "../data/customers.json"
    data_path_products = "../data/products.json"
    
    # Fallback
    if not os.path.exists(data_path_customers): data_path_customers = "data/customers.json"
    if not os.path.exists(data_path_products): data_path_products = "data/products.json"
    
    print(f"Loading customers from {data_path_customers}")
    with open(data_path_customers, "r", encoding="utf-8") as f:
        customers = json.load(f)
        
    print(f"Loading products from {data_path_products}")
    with open(data_path_products, "r", encoding="utf-8") as f:
        products = json.load(f)
    
    # Load model
    model_path = os.path.join(latest_dir, "model.pth")
    if not os.path.exists(model_path):
        model_path = "model.pth"
        
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)
    brand_map = checkpoint['brand_map']
    
    model = CustomerProductModel(num_brands=len(brand_map))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return customers, products, model, brand_map, latest_dir


def visualize_results():
    customers, products, model, brand_map, output_dir = load_all()
    
    print("\n" + "=" * 50)
    print("VISUALIZATION")
    print("=" * 50)
    
    # Pre-compute product embeddings
    print("Computing product embeddings...")
    prod_embeddings = []
    prod_categories = []
    
    with torch.no_grad():
        for prod in products:
            cat_val = prod.get('category', 97)
            if isinstance(cat_val, str):
                cat_code = CATEGORY_MAP_KR.get(cat_val, 97)
            else:
                cat_code = int(cat_val)

            brand = brand_map.get(prod.get('brand_name', ''), 0)
            price = min(parse_price(prod.get('price', 0)) / 200000.0, 1.0)
            
            emb = model.encode_product(
                torch.tensor([CATEGORY_MAP.get(cat_code, 0)]), 
                torch.tensor([brand]), 
                torch.tensor([price])
            ).numpy()[0]
            prod_embeddings.append(emb)
            prod_categories.append(CATEGORY_NAMES.get(cat_code, str(cat_code)))
    
    prod_embeddings = np.array(prod_embeddings)
    
    # 1. Combined Embedding Space (Customers + Products)
    print("Creating embedding space visualization...")
    all_embeddings = []
    all_labels = []
    all_types = []  # 'customer' or 'product'
    
    # Add customer embeddings (sample for speed)
    sample_size = min(500, len(customers))
    for cust in customers[:sample_size]:
        st = SKIN_TYPE_MAP.get(cust.get('skin_type', 'normal'), 4)
        sc = SKIN_CONCERN_MAP.get(cust.get('skin_concern', 'moisture'), 0)
        g = GENDER_MAP.get(cust.get('gender', 'F'), 0)
        age = cust.get('age', 30) / 50.0
        
        with torch.no_grad():
            emb = model.encode_customer(
                torch.tensor([st]), torch.tensor([sc]), torch.tensor([g]), torch.tensor([age])
            ).numpy()[0]
        all_embeddings.append(emb)
        all_labels.append(cust.get('persona_name', 'Unknown'))
        all_types.append('customer')
    
    # Add product embeddings (sample)
    prod_sample_size = min(300, len(products))
    for i in range(prod_sample_size):
        all_embeddings.append(prod_embeddings[i])
        all_labels.append(prod_categories[i])
        all_types.append('product')
    
    all_embeddings = np.array(all_embeddings)
    # Tune t-SNE for better separation
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=2000, init='pca', learning_rate='auto')
    embedded = tsne.fit_transform(all_embeddings)
    
    # Plot
    plt.figure(figsize=(16, 12))
    sns.set_style("whitegrid")  # Back to whitegrid for readability
    
    # 1. Plot Customers (Cloud)
    cust_mask = [t == 'customer' for t in all_types]
    if any(cust_mask):
        cust_emb = embedded[cust_mask]
        cust_lbl_raw = [l for i, l in enumerate(all_labels) if all_types[i] == 'customer']
        
        name_map = {
            'Loyal_User': 'Light_Satisfied',
            'Loyal_User_3': 'Brand_Fan'
        }
        cust_lbl = [name_map.get(l, l) for l in cust_lbl_raw]
        
        unique_personas = sorted(list(set(cust_lbl)))
        palette = sns.color_palette("husl", len(unique_personas))
        
        for i, persona in enumerate(unique_personas):
            mask = np.array([l == persona for l in cust_lbl])
            points = cust_emb[mask]
            color = palette[i]
            
            # Plot dots with varying transparency to show density
            plt.scatter(points[:, 0], points[:, 1], c=[color], label=persona, 
                       alpha=0.4, s=30, marker='o', edgecolors='white', linewidth=0.1)
            
            # Add Centroid Label (Keep this, it's useful)
            centroid = points.mean(axis=0)
            plt.text(centroid[0], centroid[1], persona, fontsize=11, fontweight='bold', 
                     bbox=dict(facecolor='white', edgecolor=color, alpha=0.9, boxstyle='round,pad=0.3'))


    # 2. Plot Products (Stars)
    prod_mask = [t == 'product' for t in all_types]
    if any(prod_mask):
        prod_emb = embedded[prod_mask]
        prod_lbl = [l for i, l in enumerate(all_labels) if all_types[i] == 'product']
        unique_cats = sorted(list(set(prod_lbl)))
        
        for i, cat in enumerate(unique_cats):
            mask = np.array([l == cat for l in prod_lbl])
            plt.scatter(prod_emb[mask, 0], prod_emb[mask, 1], 
                       label=f'Product: {cat}', alpha=1.0, s=150, marker='*', edgecolors='black', linewidth=0.5)

    plt.title('Customer Personas & Product Embedding Space', fontsize=20, pad=20)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_space.png'), dpi=150)
    plt.close()
    print(f"Saved: embedding_space.png")
    
    # Rename duplicate personas for clarity based on analysis
    # Loyal_User -> Light_Satisfied (Simple positive reviews)
    # Loyal_User_3 -> Brand_Fan (Detailed, high-engagement reviews)
    final_personas = []
    for c in customers:
        p_name = c.get('persona_name', 'Unknown')
        if p_name == 'Loyal_User':
            final_personas.append('Light_Satisfied')
            c['persona_name'] = 'Light_Satisfied'
        elif p_name == 'Loyal_User_3':
            final_personas.append('Brand_Fan')
            c['persona_name'] = 'Brand_Fan'
        else:
            final_personas.append(p_name)
            
    counts = Counter(final_personas)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis")
    plt.title('Persona Distribution (Refined)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'persona_distribution.png'), dpi=150)
    plt.close()
    
    # 3. Generate PERSONA_DEFINITION.md
    print("Generating PERSONA_DEFINITION.md...")
    
    total = len(customers)
    md_content = f"""# Persona Definition Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Total Profiles**: {total}
**Method**: Hybrid Embedding (SBERT + TF-IDF) + KMeans

---

## 1. Persona Overview

| Persona Name | Count | Share | Characteristics |
|---|---|---|---|
"""
    
    descriptions = {
        'Light_Satisfied': '일반 만족 고객 - "좋아요", "만족해요" 등 짧고 긍정적인 평이 주를 이룸.',
        'Brand_Fan': '고관여 찐팬 - "아모레", "행사", "가격" 등을 언급하며 장문으로 상세한 피드백 제공.',
        'Scent_Lover': '향기 중시형 - 향, 냄새에 민감하며 이를 중요 구매 요인으로 삼음.',
        'Sensitive': '민감성 피부 - 트러블, 진정, 자극 여부를 최우선으로 고려.',
        'Dry_Skin': '악건성 보습 - 강력한 보습력과 수분감을 찾는 그룹.'
    }

    for persona, count in counts.most_common():
        share = count / total * 100
        desc = descriptions.get(persona, '자동 분류된 그룹')
        md_content += f"| {persona} | {count} | {share:.1f}% | {desc} |\n"
        
    md_content += "\n---\n\n## 2. Detailed Characteristics\n"
    
    for persona, count in counts.most_common():
        # Get representative profile info for this persona
        p_custs = [c for c in customers if c.get('persona_name') == persona]
        
        # Dominant Skin Type
        skin_counts = Counter([c.get('skin_type') for c in p_custs])
        top_skin = skin_counts.most_common(1)[0][0] if skin_counts else "N/A"
        
        # Dominant Concern
        concern_counts = Counter([c.get('skin_concern') for c in p_custs])
        top_concern = concern_counts.most_common(1)[0][0] if concern_counts else "N/A"
        
        desc = descriptions.get(persona, 'Description generated from clustering keywords.')
        
        md_content += f"""
### **{persona}**
- **Size**: {count} users ({count/total*100:.1f}%)
- **Dominant Skin Type**: `{top_skin}`
- **Primary Concern**: `{top_concern}`
- **Description**: {desc}
"""

    md_path = os.path.join(output_dir, 'PERSONA_DEFINITION.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Saved: {md_path}")
    print("\nVISUALIZATION COMPLETE")

if __name__ == "__main__":
    visualize_results()
