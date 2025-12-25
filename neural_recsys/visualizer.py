"""
Visualizer: Embedding Space and Persona Distribution

1. Persona distribution pie/bar chart
2. Customer-Product embedding space (t-SNE)
"""

import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from model import CustomerProductModel, SKIN_TYPE_MAP, SKIN_CONCERN_MAP, GENDER_MAP, CATEGORY_MAP

# Category name mapping
CATEGORY_NAMES = {97: 'Skincare', 98: 'Makeup', 99: 'Perfume', 101: 'Accessories', 103: 'Men'}


def load_all():
    """Load model, customers, products"""
    with open("data/customers.json", "r", encoding="utf-8") as f:
        customers = json.load(f)
    
    with open("data/amoremall_reviews.json", "r", encoding="utf-8") as f:
        products = json.load(f)
    products_df = pd.DataFrame(products)
    
    checkpoint = torch.load("model.pth")
    brand_map = checkpoint['brand_map']
    
    model = CustomerProductModel(num_brands=len(brand_map))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return customers, products_df, model, brand_map


def visualize_persona_distribution(customers, output_path="persona_distribution.png"):
    """Pie + Bar chart of persona distribution"""
    
    df = pd.DataFrame(customers)
    persona_counts = df['persona_name'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie Chart
    ax1 = axes[0]
    colors = sns.color_palette("husl", len(persona_counts))
    ax1.pie(persona_counts.values, labels=persona_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax1.set_title('Persona Distribution (Pie)')
    
    # Bar Chart
    ax2 = axes[1]
    persona_counts.plot(kind='bar', ax=ax2, color=colors)
    ax2.set_xlabel('Persona')
    ax2.set_ylabel('Count')
    ax2.set_title('Persona Distribution (Bar)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def visualize_embedding_space(customers, products_df, model, brand_map, output_path="embedding_space.png"):
    """
    Enhanced t-SNE visualization with cluster boundaries.
    """
    
    print("Encoding customers...")
    cust_embeddings = []
    cust_personas = []
    
    with torch.no_grad():
        for c in customers:
            emb = model.encode_customer(
                torch.tensor([SKIN_TYPE_MAP.get(c['skin_type'], 4)]),
                torch.tensor([SKIN_CONCERN_MAP.get(c['skin_concern'], 0)]),
                torch.tensor([GENDER_MAP.get(c['gender'], 0)]),
                torch.tensor([c['age'] / 50.0])
            ).numpy()[0]
            cust_embeddings.append(emb)
            cust_personas.append(c['persona_name'])
    
    cust_embeddings = np.array(cust_embeddings)
    
    print("Encoding products...")
    prod_embeddings = []
    prod_categories = []
    prod_names = []
    
    with torch.no_grad():
        for idx, (_, row) in enumerate(products_df.iterrows()):
            emb = model.encode_product(
                torch.tensor([CATEGORY_MAP.get(row['category'], 0)]),
                torch.tensor([brand_map.get(row['brand_name'], 0)]),
                torch.tensor([min(row['price'] / 200000.0, 1.0)])
            ).numpy()[0]
            prod_embeddings.append(emb)
            prod_categories.append(CATEGORY_NAMES.get(row['category'], 'Unknown'))
            prod_names.append(f"P_{idx+1}")
    
    prod_embeddings = np.array(prod_embeddings)
    
    # Combine for t-SNE
    all_embeddings = np.vstack([cust_embeddings, prod_embeddings])
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=15, n_iter=1500, random_state=42)
    tsne_result = tsne.fit_transform(all_embeddings)
    
    # Split back
    cust_tsne = tsne_result[:len(cust_embeddings)]
    prod_tsne = tsne_result[len(cust_embeddings):]
    
    # Create figure with dark background for better visibility
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#f5f5f5')
    
    # Persona colors
    persona_colors = {
        'Dry_Winter': '#3498db',
        'Anti_Aging': '#e74c3c',
        'Gift_Buyer': '#2ecc71',
        'Loyal_Repurchase': '#9b59b6',
        'Sensitive_Calming': '#f39c12',
        'Detail_Reviewer': '#1abc9c'
    }
    
    # Draw convex hull for each persona cluster
    from scipy.spatial import ConvexHull
    
    cust_df = pd.DataFrame({
        'x': cust_tsne[:, 0],
        'y': cust_tsne[:, 1],
        'persona': cust_personas
    })
    
    for persona, color in persona_colors.items():
        subset = cust_df[cust_df['persona'] == persona]
        if len(subset) >= 3:
            try:
                points = subset[['x', 'y']].values
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])  # Close the polygon
                ax.fill(hull_points[:, 0], hull_points[:, 1], color=color, alpha=0.15)
                ax.plot(hull_points[:, 0], hull_points[:, 1], color=color, alpha=0.5, linewidth=2)
            except:
                pass
    
    # Plot customers
    for persona, color in persona_colors.items():
        subset = cust_df[cust_df['persona'] == persona]
        ax.scatter(subset['x'], subset['y'], c=color, s=60, alpha=0.7, 
                   label=f'{persona} ({len(subset)})', edgecolors='white', linewidths=0.5)
    
    # Plot products with category colors
    category_colors = {
        'Skincare': '#e91e63',
        'Makeup': '#673ab7',
        'Perfume': '#ff9800',
        'Accessories': '#00bcd4',
        'Men': '#607d8b'
    }
    
    for i, (x, y, cat, name) in enumerate(zip(prod_tsne[:, 0], prod_tsne[:, 1], prod_categories, prod_names)):
        color = category_colors.get(cat, '#333333')
        ax.scatter(x, y, c=color, s=250, marker='s', edgecolors='black', linewidths=2, zorder=5)
    
    # Add category legend manually
    for cat, color in category_colors.items():
        ax.scatter([], [], c=color, s=150, marker='s', label=f'[Product] {cat}', edgecolors='black')
    
    ax.set_title('Customer-Product Embedding Space\n(Customers: circles, Products: squares)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")


def run_visualizations():
    print("="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    customers, products_df, model, brand_map = load_all()
    
    # 1. Persona Distribution
    visualize_persona_distribution(customers, "persona_distribution.png")
    
    # 2. Embedding Space
    visualize_embedding_space(customers, products_df, model, brand_map, "embedding_space.png")
    
    print("\nDone!")


if __name__ == "__main__":
    run_visualizations()
