"""
Clustering: SBERT + KMeans with Elbow Method for Persona Generation

1. Load review texts from products.json
2. Embed with review-optimized model (Korean RoBERTa or multilingual)
3. Apply Elbow Method to find optimal k (max 7)
4. Run KMeans clustering
5. Output labeled_reviews.csv for downstream pipeline
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ====================== CONFIGURATION ======================
# ====================== CONFIGURATION ======================
DATA_PATH = "../data/products.json"
OUTPUT_BASE = "../results"
MIN_K = 5
MAX_K = 7

# Category mapping
CATEGORY_MAP_KR = {
    "스킨케어": 97, "메이크업": 98, "향수": 99, "생활용품": 100,
    "소품&도구": 101, "뷰티푸드": 102, "남성": 103, "베이비": 104,
    "뷰티디바이스": 105, "반려동물용품": 106,
}

def load_reviews():
    """Load all reviews from products.json"""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        products = json.load(f)
    
    reviews = []
    for prod in products:
        category_kr = prod.get('category', '스킨케어')
        category_code = CATEGORY_MAP_KR.get(category_kr, 97)
        
        for review in prod.get('reviews', []):
            text = review.get('text', '')
            if text and len(text) > 20:
                reviews.append({
                    'product_id': prod.get('product_id', ''),
                    'product_name': prod.get('name', ''),
                    'category': category_code,
                    'profile_info': review.get('profile_info', ''),
                    'content': text,
                })
    return reviews


def get_hybrid_embeddings(reviews, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Combine SBERT (Semantic) + TF-IDF (Keyword) embeddings.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    
    print(f"1. SBERT Embedding ({model_name})...")
    model = SentenceTransformer(model_name)
    
    texts = []
    for r in reviews:
        profile = r.get('profile_info', '')
        content = r.get('content', '')
        # Weigh profile info heavily for SBERT
        if profile:
            combined = f"{profile} {profile} {content}" 
        else:
            combined = content
        texts.append(combined)
        
    sbert_emb = model.encode(texts, show_progress_bar=True, batch_size=32)
    sbert_emb = normalize(sbert_emb)
    
    print("2. TF-IDF Embedding...")
    # Korean Stopwords to remove generic terms
    korean_stopwords = [
        '너무', '좋아요', '제품', '같아요', '많이', '있어서', '좋은', '좋아서', '구매했어요', 
        '좋고', '피부가', '피부', '없이', '피부에', '사용할', '특히', '사용', '피부를', 
        '느낌이', '꾸준히', '바르고', '있어요', '않고', '저는', '진짜', 'ㅎㅎ', 'ㅋㅋ', 
        '완전', '향도', '제품을', '하고', '있는', '정말', '것', '수', '그리고', '거', 
        '합니다', '입니다', '쓰고', '잘', '더', '좀', '한', '했습니다', '게', '때문에'
    ]
    
    # Use only review content for TF-IDF to capture specific words like 'soft', 'gift'
    raw_contents = [r['content'] for r in reviews]
    tfidf = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=korean_stopwords)
    tfidf_emb = tfidf.fit_transform(raw_contents).toarray()
    
    # Reduce TF-IDF dimension to match SBERT scale roughly
    pca = PCA(n_components=128)
    tfidf_reduced = pca.fit_transform(tfidf_emb)
    tfidf_reduced = normalize(tfidf_reduced)
    
    print("3. Combining & Finalizing...")
    # Concatenate: 70% SBERT + 30% TF-IDF influence
    final_emb = np.hstack([sbert_emb * 0.7, tfidf_reduced * 0.3])
    
    return final_emb, tfidf, tfidf_emb


def find_optimal_k_forced(embeddings, min_k=5, max_k=7, output_dir=None):
    """
    Force K to be within range [min_k, max_k].
    Selects best silhouette score within that range.
    """
    print(f"\nFinding best K in range [{min_k}, {max_k}]...")
    
    inertias = []
    silhouettes = []
    K_range = range(min_k, max_k + 1)
    
    for k in tqdm(K_range, desc="Testing K values"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(embeddings, kmeans.labels_))
    
    # Select K with best Silhouette Score in this valid range
    best_idx = np.argmax(silhouettes)
    optimal_k = list(K_range)[best_idx]
    
    print(f"Detailed Scores:")
    for k, s in zip(K_range, silhouettes):
        print(f"K={k}: Silhouette={s:.4f}")
        
    print(f"Selected Best K={optimal_k}")
    
    # Plot
    fig, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(list(K_range), silhouettes, 'go-', linewidth=2)
    ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Selected k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score (Forced Range)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'elbow_method.png'), dpi=150)
    plt.close()
    
    return optimal_k, inertias, silhouettes


def run_clustering(embeddings, k):
    print(f"\nRunning KMeans with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans

# ... (omitted code) ...

def generate_persona_names_tfidf(reviews_df, tfidf_matrix, vectorizer, k):
    """
    Name clusters based on top TF-IDF keywords in that cluster.
    """
    print("\nNaming personas using Top Keywords...")
    feature_names = np.array(vectorizer.get_feature_names_out())
    persona_map = {}
    
    # Keyword mapping to english persona names
    keyword_to_persona = {
        '선물': 'Gift_Buyer', '세트': 'Gift_Buyer', '포장': 'Gift_Buyer',
        '주름': 'Anti_Aging', '탄력': 'Anti_Aging', '어머니': 'Anti_Aging', '엄마': 'Anti_Aging', '영양': 'Anti_Aging',
        '건조': 'Dry_Skin', '보습': 'Dry_Skin', '촉촉': 'Dry_Skin', '수분': 'Dry_Skin', '속건조': 'Dry_Skin',
        '진정': 'Sensitive', '트러블': 'Sensitive', '민감': 'Sensitive', '자극': 'Sensitive', '순한': 'Sensitive',
        '재구매': 'Loyal_User', '항상': 'Loyal_User', '쓰던': 'Loyal_User', '계속': 'Loyal_User', '정착': 'Loyal_User',
        '향': 'Scent_Lover', '냄새': 'Scent_Lover', '향기': 'Scent_Lover',
        '지성': 'Oily_Skin', '끈적': 'Oily_Skin', '산뜻': 'Oily_Skin', '유분': 'Oily_Skin', '피지': 'Oily_Skin',
        '남성': 'Men_Grooming', '남편': 'Men_Grooming', '아빠': 'Men_Grooming', '남자': 'Men_Grooming',
        '흡수': 'Texture_Focused', '제형': 'Texture_Focused', '발림성': 'Texture_Focused', '마무리': 'Texture_Focused'
    }
    
    for cluster_id in range(k):
        # Get indices of reviews in this cluster
        indices = reviews_df[reviews_df['cluster'] == cluster_id].index
        
        # Calculate average TF-IDF vector for this cluster
        cluster_tfidf = tfidf_matrix[indices].mean(axis=0)
        # Convert matrix to 1D array properly
        if hasattr(cluster_tfidf, 'A1'):
            cluster_tfidf = cluster_tfidf.A1
        else:
            cluster_tfidf = np.array(cluster_tfidf).flatten()
            
        top_indices = cluster_tfidf.argsort()[::-1][:10]
        top_keywords = feature_names[top_indices]
        
        print(f"Cluster {cluster_id} Top Keywords: {', '.join(top_keywords)}")
        
        # Determine name
        detected_name = f"Cluster_{cluster_id}"
        score_max = 0
        
        # Simple scoring
        scores = {}
        for text_kw in top_keywords:
            for map_kw, p_name in keyword_to_persona.items():
                if map_kw in text_kw:
                    scores[p_name] = scores.get(p_name, 0) + 1
        
        if scores:
            detected_name = max(scores, key=scores.get)
            
        # Handle duplicates by appending number
        if detected_name in persona_map.values():
            detected_name = f"{detected_name}_{cluster_id}"
            
        persona_map[cluster_id] = detected_name
        
    return persona_map


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load
    reviews = load_reviews()
    print(f"Loaded {len(reviews)} reviews")
    
    # 2. Hybrid Embed
    embeddings, vectorizer, tfidf_matrix = get_hybrid_embeddings(reviews)
    
    # 3. Find K (Forced)
    optimal_k, inertias, silhouettes = find_optimal_k_forced(embeddings, MIN_K, MAX_K, output_dir)
    
    # 4. Cluster
    labels, kmeans = run_clustering(embeddings, optimal_k)
    
    # 5. Name Personas
    reviews_df = pd.DataFrame(reviews)
    reviews_df['cluster'] = labels
    persona_names = generate_persona_names_tfidf(reviews_df, tfidf_matrix, vectorizer, optimal_k)
    reviews_df['persona_name'] = reviews_df['cluster'].map(persona_names)
    
    print("\nFinal Persona Distribution:")
    print(reviews_df['persona_name'].value_counts())
    
    # 6. Save
    csv_path = os.path.join(output_dir, 'labeled_reviews.csv')
    reviews_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    reviews_df.to_csv('labeled_reviews.csv', index=False, encoding='utf-8-sig')
    
    # 7. Save Info
    info = {
        'total': len(reviews),
        'k': optimal_k,
        'personas': persona_names,
        'distribution': reviews_df['persona_name'].value_counts().to_dict()
    }
    with open(os.path.join(output_dir, 'clustering_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
        
    print("\nDONE. Ready for customer_generator.py")
    return output_dir



if __name__ == "__main__":
    main()
