# Neural Recommendation System:

# 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [접근 방법론](#2-접근-방법론)
3. [Two-Tower 아키텍처](#3-two-tower-아키텍처)
4. [데이터 전처리](#4-데이터-전처리)
5. [모델 학습](#5-모델-학습)
6. [평가 결과](#6-평가-결과)
7. [시스템 구조](#7-시스템-구조)
8. [한계점 및 개선 방향](#8-한계점-및-개선-방향)

---

## 1. 프로젝트 개요

### 1.1 목표

**아모레몰 고객에게 개인화된 상품을 추천하는 시스템 구축**

### 1.2 데이터

- **상품 데이터**: 93개 상품 (스킨케어, 메이크업, 향수, 소품, 남성)
- **리뷰 데이터**: 277개 리뷰 (실제 구매 내역 대용)
- **고객 데이터**: 277명 (리뷰 기반 가상 고객 프로필 생성)

### 1.3 핵심 챌린지

- **Cold Start 문제**: 신규 고객에 대한 추천
- **Sparse Data**: 적은 상호작용 데이터
- **해결책**: 고객 속성(페르소나) 기반 추천으로 전환

---

## 2. 접근 방법론

### 2.1 협업 필터링(Collaborative Filtering)을 사용하지 않은 이유

**데이터 분석 결과**:

```
총 리뷰 수: 277
고유 리뷰어 수: 1 (모든 reviewer 필드가 비어있음 '')
```

| 문제                   | 상태                            | 영향                       |
| ---------------------- | ------------------------------- | -------------------------- |
| **고객 ID 없음** | reviewer = '' (전부 빈값)       | User-Item Matrix 생성 불가 |
| 상호작용 추적 불가     | 누가 뭘 샀는지 모름             | CF 알고리즘 적용 불가      |
| 데이터 구조            | 1 리뷰 = 1 익명 고객으로 처리됨 | 재구매/패턴 분석 불가      |

**결론**: 리뷰어 ID가 없어서 전통적인 협업 필터링(User-based CF, Item-based CF, Matrix Factorization) 모두 적용 불가능

### 2.2 우리의 접근: 페르소나 기반 추천

```
개선된 방식:
고객 속성 (age, skin_type, ...) ──┐
                                  ├──→ 유사도 계산 ──→ 추천
상품 속성 (category, brand, ...) ──┘
```

**핵심 아이디어**:

- User ID 대신 **고객의 특성(페르소나)**을 사용
- 비슷한 특성의 고객이 구매한 상품을 추천
- 신규 고객도 특성만 알면 추천 가능

### 2.3 페르소나 정의

리뷰 텍스트를 SBERT로 임베딩 → K-Means 클러스터링 → 6개 페르소나 도출:

| 페르소나          | 특성                   | 샘플 수 |
| ----------------- | ---------------------- | ------- |
| Dry_Winter        | 건성, 보습 고민        | 54      |
| Anti_Aging        | 주름, 탄력, 영양       | 17      |
| Gift_Buyer        | 선물 구매 목적         | 4       |
| Loyal_Repurchase  | 재구매, 충성 고객      | 91      |
| Sensitive_Calming | 민감성, 진정 필요      | 50      |
| Detail_Reviewer   | 성분 분석, 텍스처 중시 | 61      |

---

## 3. Two-Tower 아키텍처

### 3.1 개념

Two-Tower(쌍둥이 타워)는 **두 개의 독립적인 인코더**로 구성된 추천 모델입니다.

```
┌─────────────────────┐     ┌─────────────────────┐
│   Customer Tower    │     │   Product Tower     │
│                     │     │                     │
│  고객 특성 입력     │     │  상품 특성 입력     │
│  (age, skin_type,   │     │  (category, brand,  │
│   gender, concern)  │     │   price)            │
│         ↓           │     │         ↓           │
│   Embedding Layer   │     │   Embedding Layer   │
│         ↓           │     │         ↓           │
│       MLP           │     │       MLP           │
│         ↓           │     │         ↓           │
│  L2 Normalize       │     │  L2 Normalize       │
│         ↓           │     │         ↓           │
│   32-dim Vector     │     │   32-dim Vector     │
└─────────┬───────────┘     └───────────┬─────────┘
          │                             │
          └──────────┬──────────────────┘
                     ↓
              Cosine Similarity
                     ↓
               추천 점수 (0~1)
```

### 3.2 왜 Two-Tower인가?

1. **효율성**: 상품 임베딩을 미리 계산해두고, 고객이 올 때마다 빠르게 유사도 계산
2. **확장성**: 새 상품 추가 시 Product Tower만 한 번 실행
3. **유연성**: 고객/상품 특성이 다른 형태여도 같은 공간에 매핑 가능

### 3.3 MLP 구조 상세

#### Customer Tower MLP

```python
# 입력: 49차원 (3개 임베딩 × 16 + 1개 나이)
# - skin_type embedding: 16차원
# - skin_concern embedding: 16차원  
# - gender embedding: 16차원
# - age (normalized): 1차원

nn.Sequential(
    nn.Linear(49, 64),        # 49 → 64
    nn.ReLU(),                # 활성화 함수
    nn.BatchNorm1d(64),       # 배치 정규화
    nn.Dropout(0.2),          # 20% 드롭아웃 (과적합 방지)
    nn.Linear(64, 32)         # 64 → 32 (최종 임베딩)
)
```

#### Product Tower MLP

```python
# 입력: 33차원 (2개 임베딩 × 16 + 1개 가격)
# - category embedding: 16차원
# - brand embedding: 16차원
# - price (normalized): 1차원

nn.Sequential(
    nn.Linear(33, 64),        # 33 → 64
    nn.ReLU(),                # 활성화 함수
    nn.BatchNorm1d(64),       # 배치 정규화
    nn.Dropout(0.2),          # 20% 드롭아웃
    nn.Linear(64, 32)         # 64 → 32 (최종 임베딩)
)
```

### 3.4 Categorical Embedding

범주형 변수(skin_type, category 등)를 숫자 벡터로 변환:

```python
# 예: skin_type (5개 종류) → 16차원 벡터
self.skin_type_emb = nn.Embedding(5, 16)

# 작동 방식:
# "dry" → 0 → [0.12, -0.34, 0.56, ..., 0.23]  (16차원)
# "oily" → 1 → [-0.45, 0.67, 0.11, ..., -0.78] (16차원)
```

**장점**: 원-핫 인코딩보다 정보 밀도가 높고, 학습 과정에서 의미 있는 표현을 배움

---

## 4. 데이터 전처리

### 4.1 고객 데이터 생성 (`customer_generator.py`)

리뷰 데이터에서 가상 고객 프로필 생성:

```python
customer = {
    "user_id": "user_0001",
    "persona_id": 0,           # 클러스터링 결과
    "persona_name": "Dry_Winter",
    "age": 35,                 # 페르소나 기반 시뮬레이션
    "gender": "F",
    "skin_type": "dry",        # 페르소나에서 유도
    "skin_concern": "moisture",
    "purchased_product_id": "abc-123",  # 실제 리뷰한 상품
    "product_category": 97     # 스킨케어
}
```

### 4.2 Feature Encoding

```python
# 고객 특성 인코딩
SKIN_TYPE_MAP = {'dry': 0, 'oily': 1, 'combination': 2, 'sensitive': 3, 'normal': 4}
SKIN_CONCERN_MAP = {'moisture': 0, 'wrinkle': 1, 'gift': 2, 'repurchase': 3, 'calming': 4, 'texture': 5}
GENDER_MAP = {'F': 0, 'M': 1}

# 상품 특성 인코딩
CATEGORY_MAP = {97: 0, 98: 1, 99: 2, 101: 3, 103: 4}  # 스킨케어, 메이크업, ...
# brand_map은 데이터에서 동적 생성 (19개 브랜드)

# 수치형 정규화
age_normalized = age / 50.0      # 0~1 범위
price_normalized = price / 200000.0  # 0~1 범위
```

---

## 5. 모델 학습

### 5.1 Triplet Loss

**개념**: "고객이 실제 산 상품은 가깝게, 안 산 상품은 멀게" 학습

```
Anchor (고객) ──── Positive (구매 상품)   → 거리 최소화
     ↓
     └────────── Negative (미구매 상품)  → 거리 유지(margin)
```

**수식**:

```
Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

- `d()`: 유클리드 거리
- `margin`: 1.0 (positive와 negative 사이 최소 거리)

### 5.2 학습 파이프라인 (`trainer.py`)

```python
# 1. 데이터 준비
dataset = CustomerProductDataset(customers, products, brand_map)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 모델 & 최적화
model = CustomerProductModel(num_brands=19)
criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 학습 루프 (100 Epochs)
for epoch in range(100):
    for batch in dataloader:
        # Forward
        cust_emb = model.encode_customer(...)
        pos_emb = model.encode_product(pos_product)
        neg_emb = model.encode_product(neg_product)
      
        # Loss & Backprop
        loss = criterion(cust_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()
```

### 5.3 학습 결과 (500 Epochs)

```
Epoch 100/500 | Loss: 0.6322
Epoch 200/500 | Loss: 0.6288
Epoch 300/500 | Loss: 0.6558
Epoch 400/500 | Loss: 0.5941
Epoch 500/500 | Loss: 0.5837
```

- Loss가 ~0.58까지 수렴 (초기 0.63 대비 개선)
- 500 에포크 이후 의미 있는 임베딩 학습 완료

---

## 6. 평가 결과

### 6.1 평가 메트릭

| 메트릭                   | 설명                              | 결과 (Top-3)    |
| ------------------------ | --------------------------------- | --------------- |
| **Hit@3**          | 실제 구매 상품이 Top-3에 포함     | **19.9%** |
| **Category Hit@3** | 같은 카테고리 상품이 Top-3에 포함 | **52.3%** |

### 6.2 페르소나별 성능

| 페르소나          | Hit@3 | Category Hit@3 | 해석                     |
| ----------------- | ----- | -------------- | ------------------------ |
| Gift_Buyer        | 50.0% | 75.0%          | ⭐ 최고 성능 (특성 명확) |
| Anti_Aging        | 23.5% | 41.2%          | 좋음                     |
| Sensitive_Calming | 22.0% | 66.0%          | 카테고리 잘 맞춤         |
| Detail_Reviewer   | 21.3% | 39.3%          | 다양한 상품 구매         |
| Loyal_Repurchase  | 17.6% | 41.8%          | 개선됨                   |
| Dry_Winter        | 16.7% | 74.1%          | 카테고리 잘 맞춤         |

### 6.3 Baseline 비교

| 방법                | Hit@3           | Category Hit@3  |
| ------------------- | --------------- | --------------- |
| Random              | ~1.1%           | ~20%            |
| **Our Model** | **18.4%** | **49.1%** |

---

## 7. 시스템 구조

### 7.1 파일 구조

```
amore_project/
├── neural_recsys/              # 핵심 모듈
│   ├── model.py               # Two-Tower 모델 정의
│   ├── trainer.py             # 학습 파이프라인
│   ├── evaluator.py           # 평가 메트릭
│   ├── recommender.py         # 추천 API
│   ├── visualizer.py          # 시각화
│   └── customer_generator.py  # 고객 데이터 생성
├── data/
│   ├── customers.json         # 277명 고객 프로필
│   └── amoremall_reviews.json # 상품 + 리뷰 원본
├── model.pth                   # 학습된 가중치
├── results.csv                 # 평가 상세 결과
├── results_by_persona.png      # 성능 차트
├── persona_distribution.png    # 페르소나 분포
├── embedding_space.png         # 임베딩 시각화
└── PERSONA_DEFINITION.md       # 페르소나 정의서
```

### 7.2 사용 방법

```python
from neural_recsys.recommender import Recommender

# 초기화 (모델 로드)
rec = Recommender()

# 추천 받기
recommendations = rec.recommend(
    age=32,
    gender='F',
    skin_type='dry',
    skin_concern='moisture',
    purchased_products=['이미구매상품ID'],  # 제외 목록
    top_n=3
)

# 결과
for r in recommendations:
    print(f"{r['name']} ({r['brand']}) - {r['category']} - {r['price']}원")
```

---

## 8. 한계점 및 개선 방향

### 8.1 현재 한계

| 한계점                  | 원인                                 | 영향                    |
| ----------------------- | ------------------------------------ | ----------------------- |
| 낮은 Hit Rate           | 데이터 부족 (277건)                  | 정확한 상품 매칭 어려움 |
| 카테고리 불일치         | skin_type↔category 명시적 규칙 없음 | 건성에 메이크업 추천됨  |
| Loyal_Repurchase 저성능 | 특성이 불분명                        | 패턴 학습 어려움        |

### 8.2 개선 방향

1. **데이터 증강**

   - 더 많은 리뷰 수집
   - Synthetic 데이터 생성
2. **Rule-based Boost**

   ```python
   # skin_type='dry' → Skincare 가중치 +0.3
   if skin_type == 'dry' and category == 'Skincare':
       score += 0.3
   ```
3. **Hard Negative Mining**

   - 가장 가까운 오답 상품을 Negative로 사용
   - 더 정교한 경계 학습
4. **Contrastive Learning 도입**

   - InfoNCE Loss로 전환
   - Batch 내 모든 Negative 활용

---

## 부록: 모델 하이퍼파라미터

```python
# Embedding
embed_dim = 16          # 각 범주형 특성의 임베딩 차원

# MLP
hidden_dim = 64         # 은닉층 뉴런 수
output_dim = 32         # 최종 임베딩 차원
dropout = 0.2           # 드롭아웃 비율

# Training
batch_size = 32
learning_rate = 0.001
epochs = 500            # 500 에포크 학습
margin = 1.0            # Triplet Loss margin
optimizer = Adam
```

---

**작성일**: 2025-12-26
**버전**: 1.0
