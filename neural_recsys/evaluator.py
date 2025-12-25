"""
Evaluation V2: Structured Feature Model Evaluation

Measures:
- Hit@K: Exact product match
- Category Hit@K: Category match
- Visualizes results by persona
"""

import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from model import CustomerProductModel, SKIN_TYPE_MAP, SKIN_CONCERN_MAP, GENDER_MAP, CATEGORY_MAP


def load_model_and_data():
    """Load trained model and all data"""
    
    # 1. Load customers
    with open("data/customers.json", "r", encoding="utf-8") as f:
        customers = json.load(f)
    
    # 2. Load products
    with open("data/amoremall_reviews.json", "r", encoding="utf-8") as f:
        products_raw = json.load(f)
    products_df = pd.DataFrame(products_raw)
    
    # 3. Load model
    checkpoint = torch.load("model.pth")
    brand_map = checkpoint['brand_map']
    
    model = CustomerProductModel(
        num_skin_types=5,
        num_skin_concerns=6,
        num_genders=2,
        num_categories=5,
        num_brands=len(brand_map),
        embed_dim=16,
        hidden_dim=64,
        output_dim=32
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded {len(customers)} customers, {len(products_df)} products")
    return customers, products_df, model, brand_map


def encode_all_products(products_df, model, brand_map):
    """Pre-compute embeddings for all products"""
    product_ids = []
    product_embs = []
    product_cats = []
    
    with torch.no_grad():
        for _, row in products_df.iterrows():
            pid = row['product_id']
            cat = CATEGORY_MAP.get(row['category'], 0)
            brand = brand_map.get(row['brand_name'], 0)
            price = min(row['price'] / 200000.0, 1.0)
            
            emb = model.encode_product(
                torch.tensor([cat]),
                torch.tensor([brand]),
                torch.tensor([price])
            )
            
            product_ids.append(pid)
            product_embs.append(emb.numpy()[0])
            product_cats.append(row['category'])
    
    return product_ids, np.array(product_embs), product_cats


def recommend_for_customer(cust, model, product_ids, product_embs, product_cats, top_k=5):
    """Get top-K recommendations for a customer"""
    
    with torch.no_grad():
        cust_emb = model.encode_customer(
            torch.tensor([SKIN_TYPE_MAP.get(cust['skin_type'], 4)]),
            torch.tensor([SKIN_CONCERN_MAP.get(cust['skin_concern'], 0)]),
            torch.tensor([GENDER_MAP.get(cust['gender'], 0)]),
            torch.tensor([cust['age'] / 50.0])
        ).numpy()[0]
    
    # Compute similarities
    scores = product_embs @ cust_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    recs = [(product_ids[i], product_cats[i], scores[i]) for i in top_indices]
    return recs


def evaluate(customers, model, product_ids, product_embs, product_cats, top_k=3):
    """Full evaluation"""
    
    results = []
    
    for cust in customers:
        gt_pid = cust['purchased_product_id']
        gt_cat = cust['product_category']
        persona = cust['persona_name']
        
        # Get recommendations
        recs = recommend_for_customer(cust, model, product_ids, product_embs, product_cats, top_k)
        rec_pids = [r[0] for r in recs]
        rec_cats = [r[1] for r in recs]
        
        # Evaluate
        hit = 1 if gt_pid in rec_pids else 0
        cat_hit = 1 if gt_cat in rec_cats else 0
        
        results.append({
            'user_id': cust['user_id'],
            'persona': persona,
            'gt_product': gt_pid,
            'gt_category': gt_cat,
            'hit': hit,
            'category_hit': cat_hit,
            'top_rec_pid': rec_pids[0],
            'top_rec_cat': rec_cats[0]
        })
    
    return pd.DataFrame(results)


def visualize_results(results_df, output_dir="."):
    """Generate evaluation charts"""
    
    # Overall metrics
    overall_hit = results_df['hit'].mean() * 100
    overall_cat_hit = results_df['category_hit'].mean() * 100
    
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS (Top-3)")
    print(f"{'='*50}")
    print(f"Hit@3 (Exact Product Match): {overall_hit:.1f}%")
    print(f"Category Hit@3: {overall_cat_hit:.1f}%")
    
    # By Persona
    persona_metrics = results_df.groupby('persona').agg({
        'hit': 'mean',
        'category_hit': 'mean',
        'user_id': 'count'
    }).rename(columns={'user_id': 'count'})
    persona_metrics['hit'] *= 100
    persona_metrics['category_hit'] *= 100
    persona_metrics = persona_metrics.round(1)
    
    print(f"\n{'='*50}")
    print("METRICS BY PERSONA")
    print(f"{'='*50}")
    print(persona_metrics.to_string())
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Hit@5 by Persona
    ax1 = axes[0]
    persona_metrics['hit'].sort_values().plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Hit@3 (%)')
    ax1.set_title('Exact Product Match Rate by Persona')
    ax1.axvline(x=overall_hit, color='red', linestyle='--', label=f'Avg: {overall_hit:.1f}%')
    ax1.legend()
    
    # Chart 2: Category Hit@3 by Persona
    ax2 = axes[1]
    persona_metrics['category_hit'].sort_values().plot(kind='barh', ax=ax2, color='forestgreen')
    ax2.set_xlabel('Category Hit@3 (%)')
    ax2.set_title('Category Match Rate by Persona')
    ax2.axvline(x=overall_cat_hit, color='red', linestyle='--', label=f'Avg: {overall_cat_hit:.1f}%')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_v2_by_persona.png", dpi=150)
    print(f"\nSaved: {output_dir}/evaluation_v2_by_persona.png")
    
    return persona_metrics


def run_evaluation_v2():
    print("="*50)
    print("EVALUATION V2: Structured Feature Model")
    print("="*50)
    
    # Load
    customers, products_df, model, brand_map = load_model_and_data()
    
    # Encode all products
    print("\nEncoding all products...")
    product_ids, product_embs, product_cats = encode_all_products(products_df, model, brand_map)
    
    # Evaluate
    print("Evaluating...")
    results_df = evaluate(customers, model, product_ids, product_embs, product_cats, top_k=5)
    
    # Save
    results_df.to_csv("evaluation_v2_results.csv", index=False)
    print("Saved: evaluation_v2_results.csv")
    
    # Visualize
    visualize_results(results_df)
    
    return results_df


if __name__ == "__main__":
    run_evaluation_v2()
