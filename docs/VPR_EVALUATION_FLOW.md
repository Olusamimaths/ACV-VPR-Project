# VPR Evaluation & Comparison Flow - Technical Documentation

This document explains the complete evaluation and visualization pipeline used in the VPR Tutorial codebase. Use this as a reference when implementing evaluation for the campus dataset.

---

## 1. PIPELINE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VPR EVALUATION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  1. LOAD     │───▶│ 2. EXTRACT   │───▶│ 3. COMPUTE   │                  │
│  │   DATASET    │    │   FEATURES   │    │  SIMILARITY  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│        │                                        │                           │
│        ▼                                        ▼                           │
│  ┌──────────────┐                       ┌──────────────┐                   │
│  │  imgs_db     │                       │  S matrix    │                   │
│  │  imgs_q      │                       │  [N_db × N_q]│                   │
│  │  GThard      │                       └──────────────┘                   │
│  │  GTsoft      │                              │                           │
│  └──────────────┘                              ▼                           │
│        │                               ┌──────────────┐                    │
│        │                               │ 4. MATCHING  │                    │
│        │                               │  STRATEGIES  │                    │
│        │                               └──────────────┘                    │
│        │                                      │                            │
│        │         ┌────────────────────────────┼────────────────────────┐   │
│        │         ▼                            ▼                        ▼   │
│        │  ┌─────────────┐             ┌─────────────┐          ┌──────────┐│
│        │  │ M1: Best    │             │ M2: Thresh- │          │ TP & FP  ││
│        │  │ Match/Query │             │ olding      │          │ Arrays   ││
│        │  └─────────────┘             └─────────────┘          └──────────┘│
│        │                                                             │     │
│        └─────────────────────────┬───────────────────────────────────┘     │
│                                  ▼                                         │
│                          ┌──────────────┐                                  │
│                          │ 5. EVALUATE  │                                  │
│                          │   & COMPARE  │                                  │
│                          └──────────────┘                                  │
│                                  │                                         │
│         ┌────────────────────────┼────────────────────────┐                │
│         ▼                        ▼                        ▼                │
│  ┌─────────────┐         ┌─────────────┐          ┌─────────────┐         │
│  │ PR Curve    │         │ Recall@K    │          │ Visual      │         │
│  │ AUC, R@100P │         │ R@1,R@5,R@10│          │ Matches     │         │
│  └─────────────┘         └─────────────┘          └─────────────┘         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. KEY DATA STRUCTURES

### 2.1 Images
```python
imgs_db: List[np.ndarray]  # Database/reference images (e.g., day images)
imgs_q:  List[np.ndarray]  # Query images (e.g., night images)
```

### 2.2 Similarity Matrix S
```python
S: np.ndarray  # Shape: [N_db × N_q]
               # S[i,j] = similarity between database image i and query image j
               # Higher values = more similar
```

**How S is computed (for most descriptors):**
```python
# 1. Extract feature descriptors
db_D = feature_extractor.compute_features(imgs_db)  # [N_db × D]
q_D = feature_extractor.compute_features(imgs_q)    # [N_q × D]

# 2. L2 normalize
db_D = db_D / np.linalg.norm(db_D, axis=1, keepdims=True)
q_D = q_D / np.linalg.norm(q_D, axis=1, keepdims=True)

# 3. Compute cosine similarity
S = np.matmul(db_D, q_D.transpose())  # [N_db × N_q]
```

### 2.3 Ground Truth Matrices

**GThard (Hard Ground Truth):**
```python
GThard: np.ndarray  # Shape: [N_db × N_q], dtype: bool
                    # GThard[i,j] = True if db image i MATCHES query image j
                    # Strict matching - exact location match only
```

**GTsoft (Soft Ground Truth):**
```python
GTsoft: np.ndarray  # Shape: [N_db × N_q], dtype: bool
                    # GTsoft[i,j] = True if db image i is NEAR query image j
                    # Relaxed matching - nearby locations are acceptable
                    # Created by dilating GThard (±K positions tolerance)
```

**Example: Creating GTsoft from GThard:**
```python
from scipy.signal import convolve2d

# Dilate GThard by ±8 positions (17 total including center)
GTsoft = convolve2d(GThard.astype('int'),
                    np.ones((17, 1), 'int'),
                    mode='same').astype('bool')
```

**Visual representation:**
```
GThard (strict):           GTsoft (relaxed, ±2):
   Q0 Q1 Q2 Q3 Q4            Q0 Q1 Q2 Q3 Q4
D0 [1  0  0  0  0]        D0 [1  1  1  0  0]
D1 [0  1  0  0  0]        D1 [1  1  1  1  0]
D2 [0  0  1  0  0]   →    D2 [0  1  1  1  1]
D3 [0  0  0  1  0]        D3 [0  0  1  1  1]
D4 [0  0  0  0  1]        D4 [0  0  0  1  1]
```

---

## 3. MATCHING STRATEGIES

### 3.1 Best Match Per Query (Single-Best-Match VPR)
**Location:** `matching/matching.py` → `best_match_per_query()`

**Purpose:** For each query, select the single most similar database image.

```python
def best_match_per_query(S):
    # For each column (query), find row (db) with maximum similarity
    i = np.argmax(S, axis=0)  # Index of best match for each query
    j = np.arange(len(i))     # Query indices

    M = np.zeros_like(S, dtype='bool')
    M[i, j] = True  # Mark best match as True

    return M  # Binary matrix: M[i,j]=True means "db i is best match for query j"
```

**Output M1:**
```
S (similarity):            M1 (best match):
   Q0   Q1   Q2              Q0 Q1 Q2
D0 [0.9  0.2  0.1]        D0 [1  0  0]   ← D0 is best match for Q0
D1 [0.3  0.8  0.2]   →    D1 [0  1  0]   ← D1 is best match for Q1
D2 [0.1  0.3  0.7]        D2 [0  0  1]   ← D2 is best match for Q2
```

### 3.2 Thresholding (Multi-Match VPR)
**Location:** `matching/matching.py` → `thresholding()`

**Purpose:** Accept all matches above a threshold (allows multiple matches per query).

```python
def thresholding(S, thresh):
    if thresh == 'auto':
        # Automatic threshold using robust statistics
        mu = np.median(S)                           # Median similarity
        sig = np.median(np.abs(S - mu)) / 0.675     # Robust std estimate
        thresh = norm.ppf(1 - 1e-6, loc=mu, scale=sig)  # High confidence threshold

    M = S >= thresh  # All similarities above threshold
    return M
```

**Output M2:**
```
S (similarity):            M2 (thresholding, thresh=0.5):
   Q0   Q1   Q2              Q0 Q1 Q2
D0 [0.9  0.2  0.6]        D0 [1  0  1]   ← D0 matches Q0 AND Q2
D1 [0.3  0.8  0.2]   →    D1 [0  1  0]
D2 [0.1  0.3  0.7]        D2 [0  0  1]   ← D2 also matches Q2
```

### 3.3 Computing True/False Positives
```python
# After thresholding
M2 = matching.thresholding(S, 'auto')

# True Positives: Matches that ARE in ground truth
TP = np.argwhere(M2 & GThard)   # [[db_idx, q_idx], ...]

# False Positives: Matches that are NOT in ground truth (even soft)
FP = np.argwhere(M2 & ~GTsoft)  # [[db_idx, q_idx], ...]
```

---

## 4. EVALUATION METRICS

### 4.1 Precision-Recall Curve
**Location:** `evaluation/metrics.py` → `createPR()`

**Concept:**
- Sweep threshold from highest to lowest similarity
- At each threshold, compute precision and recall
- Plot P vs R curve

```python
def createPR(S, GThard, GTsoft=None, matching='multi', n_thresh=100):
    # Ignore GTsoft region (not penalized, not rewarded)
    S_eval = S.copy()
    if GTsoft is not None:
        S_eval[GTsoft & ~GThard] = S.min()  # Ignore soft-only matches

    # For multi-match VPR
    GTP = np.count_nonzero(GThard)  # Total ground truth positives

    P, R = [1], [0]  # Start at (R=0, P=1)

    for thresh in np.linspace(S.max(), S.min(), n_thresh):
        B = S_eval >= thresh  # Binary predictions at this threshold

        TP = np.count_nonzero(GThard & B)      # True positives
        FP = np.count_nonzero((~GThard) & B)   # False positives

        precision = TP / (TP + FP)  # Of predicted matches, how many correct?
        recall = TP / GTP           # Of actual matches, how many found?

        P.append(precision)
        R.append(recall)

    return P, R
```

### 4.2 Area Under Curve (AUC)
```python
P, R = createPR(S, GThard, GTsoft, matching='multi')
AUC = np.trapz(P, R)  # Numerical integration
# AUC ∈ [0, 1], higher is better
# AUC = 1.0 means perfect separation
```

### 4.3 Recall at 100% Precision (R@100P)
**Location:** `evaluation/metrics.py` → `recallAt100precision()`

**Concept:** Maximum recall achievable without ANY false positives.

```python
def recallAt100precision(S, GThard, GTsoft=None, matching='multi'):
    P, R = createPR(S, GThard, GTsoft, matching=matching)
    P, R = np.array(P), np.array(R)

    # Find recall values where precision = 100%
    R_at_100P = R[P == 1]

    return R_at_100P.max()  # Maximum recall at 100% precision
```

**Interpretation:**
- R@100P = 0.3 means: "We can correctly identify 30% of matches with zero false alarms"
- Important for safety-critical applications

### 4.4 Recall at K (R@K)
**Location:** `evaluation/metrics.py` → `recallAtK()`

**Concept:** For what fraction of queries is the correct match in the top-K results?

```python
def recallAtK(S, GT, K=1):
    # Only consider queries that have a ground truth match
    valid_queries = GT.sum(0) > 0
    S = S[:, valid_queries]
    GT = GT[:, valid_queries]

    # Get top-K database indices for each query
    top_k_indices = S.argsort(axis=0)[-K:, :]  # [K × N_q]

    # Check if any of top-K are correct
    # For each query, is there at least one correct match in top-K?
    correct = GT[top_k_indices, np.arange(GT.shape[1])].any(axis=0)

    return correct.sum() / len(correct)
```

**Common values:**
- **R@1:** Is the #1 match correct? (most strict)
- **R@5:** Is correct match in top 5?
- **R@10:** Is correct match in top 10? (most lenient)

---

## 5. VISUALIZATION

### 5.1 Similarity Matrix
```python
plt.figure()
plt.imshow(S)
plt.colorbar()
plt.xlabel('Query images')
plt.ylabel('Database images')
plt.title('Similarity Matrix S')
```

**What to look for:**
- Bright diagonal = good performance (matching images have high similarity)
- Bright off-diagonal spots = potential confusion/false matches

### 5.2 Matching Matrices
```python
fig, (ax1, ax2) = plt.subplots(1, 2)

# M1: Best match per query
ax1.imshow(M1)
ax1.set_title('Best match per query')

# M2: Thresholding
ax2.imshow(M2)
ax2.set_title('Thresholding (S >= thresh)')
```

### 5.3 Correct/Wrong Match Examples
**Location:** `evaluation/show_correct_and_wrong_matches.py`

```python
# TP: True Positive indices [[db_idx, q_idx], ...]
# FP: False Positive indices [[db_idx, q_idx], ...]

show_correct_and_wrong_matches.show(imgs_db, imgs_q, TP, FP)
```

**Visual output:**
```
┌────────────────────────────────────┐
│ GREEN BORDER: Correct Match (TP)   │
│ [Database Image] | [Query Image]   │
├────────────────────────────────────┤
│ RED BORDER: Wrong Match (FP)       │
│ [Database Image] | [Query Image]   │
└────────────────────────────────────┘
```

### 5.4 Precision-Recall Curve
```python
P, R = createPR(S, GThard, GTsoft, matching='multi')

plt.figure()
plt.plot(R, P)
plt.xlim(0, 1)
plt.ylim(0, 1.01)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
```

---

## 6. COMPLETE EVALUATION CODE FLOW

```python
# ============ STEP 1: Load Data ============
dataset = GardensPointDataset()
imgs_db, imgs_q, GThard, GTsoft = dataset.load()

# ============ STEP 2: Extract Features ============
feature_extractor = CosPlaceFeatureExtractor()
db_D = feature_extractor.compute_features(imgs_db)
q_D = feature_extractor.compute_features(imgs_q)

# ============ STEP 3: Compute Similarity ============
db_D = db_D / np.linalg.norm(db_D, axis=1, keepdims=True)
q_D = q_D / np.linalg.norm(q_D, axis=1, keepdims=True)
S = np.matmul(db_D, q_D.transpose())

# ============ STEP 4: Matching ============
M1 = matching.best_match_per_query(S)
M2 = matching.thresholding(S, 'auto')

TP = np.argwhere(M2 & GThard)    # True positives
FP = np.argwhere(M2 & ~GTsoft)   # False positives

# ============ STEP 5: Evaluation ============
# 5a. PR Curve and AUC
P, R = createPR(S, GThard, GTsoft, matching='multi')
AUC = np.trapz(P, R)

# 5b. Recall at 100% Precision
R_at_100P = recallAt100precision(S, GThard, GTsoft)

# 5c. Recall at K
R_at_1 = recallAtK(S, GThard, K=1)
R_at_5 = recallAtK(S, GThard, K=5)
R_at_10 = recallAtK(S, GThard, K=10)

# ============ STEP 6: Visualization ============
# 6a. Similarity matrix
plt.figure()
plt.imshow(S)
plt.title('Similarity Matrix')

# 6b. Matching visualizations
plt.figure()
plt.subplot(121), plt.imshow(M1), plt.title('Best Match')
plt.subplot(122), plt.imshow(M2), plt.title('Thresholding')

# 6c. PR Curve
plt.figure()
plt.plot(R, P)
plt.xlabel('Recall'), plt.ylabel('Precision')

# 6d. Example matches
show_correct_and_wrong_matches.show(imgs_db, imgs_q, TP, FP)

# ============ STEP 7: Print Results ============
print(f'AUC: {AUC:.3f}')
print(f'R@100P: {R_at_100P:.3f}')
print(f'R@1: {R_at_1:.3f}, R@5: {R_at_5:.3f}, R@10: {R_at_10:.3f}')
```

---

## 7. KEY DIFFERENCES: GThard vs GTsoft

| Aspect | GThard | GTsoft |
|--------|--------|--------|
| Definition | Exact location match | Nearby location acceptable |
| Strictness | Strict | Relaxed |
| Use in TP | `M & GThard` | - |
| Use in FP | - | `M & ~GTsoft` |
| Use in PR | Defines true matches | Ignores near-matches |

**Why GTsoft matters:**
- In real-world VPR, adjacent frames often show nearly the same location
- Matching "image_042" to "image_041" is almost correct, not a failure
- GTsoft prevents penalizing these near-misses as false positives

---

## 8. APPLYING TO CAMPUS DATASET

For your campus dataset, follow the same flow:

```python
# 1. Load campus data
dataset = CampusDataset()
imgs_db, imgs_q, GThard, GTsoft = dataset.load()
# GThard: Created from filename matching (image042.jpg <-> image042.jpg)
# GTsoft: Dilated version (±2 tolerance)

# 2-6. Same as above...

# Key considerations:
# - Your dataset has -npm images (no perfect match) → column of zeros in GThard
# - These queries are excluded from R@K calculation automatically
# - They still appear in PR curve (as queries that should NOT match anything)
```

---

## 9. INTERPRETING RESULTS

### Good Performance Indicators:
- **AUC > 0.8:** Excellent
- **AUC 0.6-0.8:** Good
- **AUC < 0.6:** Poor

- **R@1 > 0.7:** Most queries find correct match immediately
- **R@100P > 0.5:** Can confidently match half the dataset

### Day-to-Night Challenges:
Expect lower performance due to:
- Lighting changes (shadows, artificial lights)
- Exposure differences
- Dynamic objects (cars, people)

Typical day-night performance:
- CosPlace/EigenPlaces: AUC ~0.7, R@1 ~0.6
- HDC-DELF: AUC ~0.6, R@1 ~0.4

---

## 10. FILE REFERENCE

| File | Purpose |
|------|---------|
| `demo.py` | Main pipeline orchestration |
| `datasets/load_dataset.py` | Load images and ground truth |
| `feature_extraction/*.py` | Extract image descriptors |
| `matching/matching.py` | Matching decision strategies |
| `evaluation/metrics.py` | PR curve, AUC, R@K metrics |
| `evaluation/show_correct_and_wrong_matches.py` | Visual match examples |

---

*Document created for CMU-Africa VPR Tutorial*
*Authors: Rhoda Ojetola, Peter Adeyemo, Samuel Olusola*
