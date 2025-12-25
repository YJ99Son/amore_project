"""
Recommender API

Input: Customer info (age, gender, skin_type, skin_concern)
Output: Top-N product recommendations (excluding already purchased)
"""

import json
import torch
import numpy as np
from typing import List, Dict, Optional

from model import CustomerProductModel, SKIN_TYPE_MAP, SKIN_CONCERN_MAP, GENDER_MAP, CATEGORY_MAP

CATEGORY_NAMES = {97: 'Skincare', 98: 'Makeup', 99: 'Perfume', 101: 'Accessories', 103: 'Men'}


class Recommender:
    def __init__(self, model_path="model.pth", products_path="data/amoremall_reviews.json"):
        # Load model
        checkpoint = torch.load(model_path)
        self.brand_map = checkpoint['brand_map']
        
        self.model = CustomerProductModel(num_brands=len(self.brand_map))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load products
        with open(products_path, "r", encoding="utf-8") as f:
            self.products = json.load(f)
        
        # Pre-compute product embeddings
        self._precompute_product_embeddings()
        
    def _precompute_product_embeddings(self):
        """Pre-compute embeddings for all products"""
        self.product_ids = []
        self.product_embeddings = []
        self.product_info = {}  # id -> info
        
        with torch.no_grad():
            for prod in self.products:
                pid = prod['product_id']
                emb = self.model.encode_product(
                    torch.tensor([CATEGORY_MAP.get(prod['category'], 0)]),
                    torch.tensor([self.brand_map.get(prod['brand_name'], 0)]),
                    torch.tensor([min(prod['price'] / 200000.0, 1.0)])
                ).numpy()[0]
                
                self.product_ids.append(pid)
                self.product_embeddings.append(emb)
                self.product_info[pid] = {
                    'name': prod['product_name'],
                    'brand': prod['brand_name'],
                    'category': CATEGORY_NAMES.get(prod['category'], 'Unknown'),
                    'price': prod['price']
                }
        
        self.product_embeddings = np.array(self.product_embeddings)
        print(f"Loaded {len(self.products)} products")
    
    def recommend(self, 
                  age: int, 
                  gender: str, 
                  skin_type: str, 
                  skin_concern: str,
                  purchased_products: Optional[List[str]] = None,
                  top_n: int = 3) -> List[Dict]:
        """
        Get top-N product recommendations for a customer.
        
        Args:
            age: Customer age (e.g., 25)
            gender: 'F' or 'M'
            skin_type: 'dry', 'oily', 'combination', 'sensitive', 'normal'
            skin_concern: 'moisture', 'wrinkle', 'gift', 'repurchase', 'calming', 'texture'
            purchased_products: List of product_ids already purchased (to exclude)
            top_n: Number of recommendations to return
            
        Returns:
            List of product dicts with id, name, brand, category, price, score
        """
        
        # Encode customer
        with torch.no_grad():
            cust_emb = self.model.encode_customer(
                torch.tensor([SKIN_TYPE_MAP.get(skin_type, 4)]),
                torch.tensor([SKIN_CONCERN_MAP.get(skin_concern, 0)]),
                torch.tensor([GENDER_MAP.get(gender, 0)]),
                torch.tensor([age / 50.0])
            ).numpy()[0]
        
        # Compute similarities
        scores = self.product_embeddings @ cust_emb
        
        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Filter out already purchased products
        purchased_set = set(purchased_products) if purchased_products else set()
        
        recommendations = []
        for idx in sorted_indices:
            pid = self.product_ids[idx]
            
            if pid in purchased_set:
                continue  # Skip already purchased
            
            info = self.product_info[pid]
            recommendations.append({
                'product_id': pid,
                'name': info['name'],
                'brand': info['brand'],
                'category': info['category'],
                'price': info['price'],
                'score': float(scores[idx])
            })
            
            if len(recommendations) >= top_n:
                break
        
        return recommendations


def demo():
    """Demo usage"""
    print("="*50)
    print("RECOMMENDER DEMO")
    print("="*50)
    
    rec = Recommender()
    
    # Test Case 1: 건성 피부 고객
    print("\n[Case 1] 32세 여성, 건성 피부, 보습 고민")
    results = rec.recommend(
        age=32,
        gender='F',
        skin_type='dry',
        skin_concern='moisture',
        top_n=3
    )
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name']} ({r['brand']}) - {r['category']} - {r['price']:,}원")
    
    # Test Case 2: 같은 고객인데 이미 구매한 상품 제외
    print("\n[Case 2] 같은 고객, 첫번째 상품 이미 구매했다고 가정")
    already_bought = [results[0]['product_id']]
    results2 = rec.recommend(
        age=32,
        gender='F',
        skin_type='dry',
        skin_concern='moisture',
        purchased_products=already_bought,
        top_n=3
    )
    for i, r in enumerate(results2, 1):
        print(f"  {i}. {r['name']} ({r['brand']}) - {r['category']} - {r['price']:,}원")
    
    # Test Case 3: 남성 고객
    print("\n[Case 3] 28세 남성, 지성 피부, 텍스처 중시")
    results3 = rec.recommend(
        age=28,
        gender='M',
        skin_type='oily',
        skin_concern='texture',
        top_n=3
    )
    for i, r in enumerate(results3, 1):
        print(f"  {i}. {r['name']} ({r['brand']}) - {r['category']} - {r['price']:,}원")


if __name__ == "__main__":
    demo()
