# Methodology

## Validation Study Design

This document describes the detailed methodology for the empirical validation study conducted as part of the systematic literature review on AI-based cryptojacking detection.

## Research Questions Addressed

The validation study empirically tests three conclusions from the SLR:

1. **RQ2 (Performance):** Can reported high accuracy (98-99%+) be replicated using public datasets?
2. **RQ3 (Challenges):** What preprocessing and deployment barriers exist?
3. **Reproducibility:** What prevents cross-study comparison?

## Dataset Selection Rationale

### Why Proxy Datasets?

Our systematic review identified a critical gap: **no publicly available datasets capture genuine cloud VM, container, or Kubernetes telemetry with labeled cryptomining activity**. This forced us to use proxy environments that share behavioral characteristics:

| Characteristic | Cloud Cryptojacking | DS2OS (IoT) | NSL-KDD (Network) |
|---------------|---------------------|-------------|-------------------|
| High resource consumption | ✓ | ✓ | ✓ |
| Distributed attack patterns | ✓ | ✓ | ✓ |
| Anomalous process behavior | ✓ | ✓ | ✓ |
| Hypervisor/container metrics | ✓ | ✗ | ✗ |
| Mining pool communication | ✓ | ✗ | ✗ |

### Dataset Characteristics

#### DS2OS
- **Domain:** IoT smart building sensor network
- **Attack Types:** DoS, scanning, malicious control, data probing, spying, wrong setup, data type probing
- **Relevance:** IoT botnets frequently incorporate cryptomining (e.g., Mirai variants)
- **Challenge:** Severe class imbalance (97.2% benign)

#### NSL-KDD
- **Domain:** Network intrusion detection (improved KDD Cup 99)
- **Attack Types:** DoS, Probe, R2L, U2R
- **Relevance:** Network-level patterns of resource hijacking
- **Challenge:** Aggregated attack categories require binary conversion

## Model Selection

Models selected based on frequency of appearance in reviewed literature:

| Model | Studies Using | Justification |
|-------|---------------|---------------|
| Random Forest | 15 (37%) | Most common classical ML approach |
| XGBoost | 8 (20%) | High-performance gradient boosting |
| LightGBM | 4 (10%) | Efficient large-scale training |
| Decision Tree | 6 (15%) | Interpretable baseline |
| KNN | 5 (12%) | Distance-based anomaly detection |
| Gradient Boosting | 4 (10%) | Ensemble comparison |

## Preprocessing Pipeline

### Step 1: Label Encoding
```python
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
```

### Step 2: Train/Test Split
- **Ratio:** 70% train, 30% test
- **Stratification:** Preserved class proportions
- **Random State:** 42 (reproducibility)

### Step 3: Class Balancing (SMOTE)
Applied when imbalance ratio < 0.3:
```python
smote = SMOTE(random_state=42, k_neighbors=min(5, minority_count - 1))
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**Critical:** SMOTE applied only to training set; test set preserved original distribution.

### Step 4: Feature Scaling
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Evaluation Metrics

### Primary Metrics
- **Accuracy:** Overall correct predictions
- **F1-Score (weighted):** Harmonic mean of precision/recall
- **Precision (weighted):** True positives / predicted positives
- **Recall (weighted):** True positives / actual positives

### Secondary Metrics
- **Confusion Matrix:** FP/FN analysis
- **False Positive Rate:** fp / (fp + tn)
- **False Negative Rate:** fn / (fn + tp)

### Computational Metrics
- **Training Time:** Wall-clock seconds
- **Inference Time:** Prediction latency on test set

## Hyperparameter Configuration

All hyperparameters documented for reproducibility:

```python
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=5,
        random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
}
```

## Computational Environment

- **Platform:** Google Colab (Standard Runtime)
- **Python:** 3.10
- **Libraries:** scikit-learn 1.3.0, XGBoost 2.0.0, LightGBM 4.0.0
- **Hardware:** Variable (Colab allocation)

## Limitations

### Dataset Limitations
1. **Proxy Nature:** DS2OS/NSL-KDD approximate but don't replicate cloud cryptojacking
2. **Temporal Validity:** NSL-KDD is dated (1999 base); modern attack patterns differ
3. **Feature Space:** IoT (12) vs Network (41) features are incompatible

### Methodological Limitations
1. **No Hyperparameter Tuning:** Used reasonable defaults; optimal may differ
2. **Single Split:** No cross-validation (computational constraints)
3. **No Adversarial Testing:** Models not tested against evasion techniques

### Reproducibility Notes
- Random seeds fixed for deterministic results
- SMOTE k_neighbors adapted to minority class size
- All preprocessing artifacts saved for inspection

## Threats to Validity

### Internal Validity
- **Selection Bias:** Proxy datasets may not represent cloud cryptojacking behavior
- **Measurement Bias:** Colab hardware variability affects timing measurements

### External Validity
- **Generalizability:** Results may not transfer to production cloud environments
- **Dataset Representativeness:** Public datasets lack modern cryptojacking signatures

### Construct Validity
- **Metric Selection:** Accuracy alone insufficient; F1 addresses class imbalance
- **Binary Classification:** Aggregating attack types may lose discriminative information

## Reproducibility Checklist

- [x] Code publicly available
- [x] Random seeds documented
- [x] Hyperparameters specified
- [x] Preprocessing steps detailed
- [x] Dataset sources linked
- [x] Environment specified
- [ ] Docker container (future work)
- [ ] Cross-validation results (computational constraints)
