# Cryptojacking Detection Validation

**Empirical Validation for AI-Based Cloud Cryptojacking Detection**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## Overview

This repo contains the validation code for our systematic literature review:

> **Detecting Cryptojacking in Cloud Environments: A Systematic Review of AI-Based Defenses, Deployment Challenges, and Research Gaps**  
> *Amitabh Chakravorty, Nelly Elsayed*  
> School of Information Technology, University of Cincinnati  

We took representative ML models from the reviewed literature and tested them on public datasets to see how well reported results actually hold up. The short answer: they often don't. Reported accuracies are frequently inflated by data leakage and testing only on attack types the model already knows.

## Key Findings

### Our Results (After Fixing Data Leakage + Rigorous Evaluation)

| Dataset | Best Model | Accuracy | F1-Score | Training Time |
|---------|-----------|----------|----------|---------------|
| DS2OS | XGBoost | 96.26% | 0.9695 | 3.16s |
| NSL-KDD | XGBoost | 80.82% | 0.8073 | 3.67s |

### How That Compares to the Literature

| Study | Model | Dataset | Reported | Ours | Gap |
|-------|-------|---------|----------|------|-----|
| Tekin et al. | RF | DS2OS | ~99.00% | 96.26% | -2.74% |
| Tiwari et al. | LightGBM | DS2OS | 98.52% | 96.26% | -2.26% |
| Safaei Pour et al. | RF | NSL-KDD | 99.60% | 77.17% | -22.43% |
| Safaei Pour et al. | Gradient Boosting | NSL-KDD | 99.60% | 78.25% | -21.35% |

**Why the gap?**

- **DS2OS (~3% drop):** Original studies kept identifier columns (timestamp, sourceID, sourceAddress) that leak the target variable. Once you remove those, the model has to actually learn behavioral patterns instead of memorizing IDs.
- **NSL-KDD (~22% drop):** Original studies tested on random splits of training data, so models only ever saw attack types they'd been trained on. We used the official KDDTest+ holdout, which includes novel attacks (mscan, saint, apache2, processtable) absent from training. That's a much harder test, and it's closer to what real deployment looks like.

> **Note:** Both datasets are *proxies* for cloud cryptojacking. No public dataset captures actual cloud VM, container, or Kubernetes telemetry with labeled cryptomining activity. That's one of the biggest findings from our review.

## Repository Structure

```
cryptojacking-validation/
├── README.md
├── requirements.txt
├── LICENSE
│
├── notebooks/                   # Run these in order
│   ├── 1_Master.ipynb          # Setup + data download
│   ├── 2_Exploration.ipynb     # Dataset exploration
│   ├── 3_Preprocessing.ipynb   # Cleaning, SMOTE, scaling
│   └── 4_Models.ipynb          # Training + evaluation
│
├── data/
│   ├── raw/                    # Downloaded datasets
│   └── processed/              # Preprocessed arrays
│
├── models/                      # Saved .pkl files
│
├── results/
│   ├── figures/                # Plots and visualizations
│   └── metrics/                # CSV metrics
│
├── scripts/
│   └── utils.py                # Helper functions
│
└── docs/
    └── METHODOLOGY.md          # Detailed methodology
```

## Quick Start

### Google Colab (Recommended)

1. Click the Colab badge above
2. Run notebooks in order: `1_Master` > `2_Exploration` > `3_Preprocessing` > `4_Models`
3. You'll need a Kaggle account and API key for DS2OS download

### Local Setup

```bash
git clone https://github.com/AmitabhCh822/cryptojacking-validation.git
cd cryptojacking-validation

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

## Datasets

### DS2OS (Distributed Smart Space Orchestration System)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/libamariyam/ds2os-dataset)
- **Samples:** 357,952
- **Original features:** 12 (IoT device telemetry)
- **After preprocessing:** 5 features (sourceType, sourceLocation, destinationServiceAddress, destinationServiceType, destinationLocation)
- **Removed for leakage:** sourceID, sourceAddress, timestamp, value, accessedNodeAddress, accessedNodeType, operation
- **Class split:** 97.2% normal, 2.8% attack

### NSL-KDD
- **Source:** [UNB CIC](https://www.unb.ca/cic/datasets/nsl.html)
- **Train:** 125,973 samples (KDDTrain+)
- **Test:** 22,544 samples (KDDTest+, includes novel attack types not seen in training)
- **Features:** 41 (network traffic patterns)
- **Class split:** ~53% normal, ~47% attack

### Why proxy datasets?

Our review found zero public datasets with real cloud cryptojacking telemetry. The closest options (CREMEv2, VIKRANT honeypot, AWS simulation repo) only capture host-level sequences or network flows. None of them include hypervisor metrics, Kubernetes pod stats, or container runtime telemetry. See Section 4.4.1 of the paper for the full breakdown.

## Models

We picked six model families based on what shows up most in the reviewed literature. Classical ML covers 57% of the studies we reviewed.

| Model | Config |
|-------|--------|
| Random Forest | 100 estimators, max_depth=20 |
| XGBoost | 100 estimators, max_depth=10, lr=0.1 |
| LightGBM | 100 estimators, max_depth=10 |
| Decision Tree | max_depth=15 |
| KNN | 5 neighbors |
| Gradient Boosting | 100 estimators, max_depth=5, lr=0.1 |

We used configs commonly reported in the literature. No automated tuning. The point here is reproducibility, not chasing the highest number.

## Results

### DS2OS (After Data Leakage Removal)

| Model | Accuracy | F1-Score | Precision | Time |
|-------|----------|----------|-----------|------|
| Random Forest | 96.26% | 0.9695 | 0.9830 | 21.92s |
| XGBoost | 96.26% | 0.9695 | 0.9830 | 3.16s |
| LightGBM | 96.26% | 0.9695 | 0.9830 | 3.78s |
| Gradient Boosting | 96.23% | 0.9693 | 0.9829 | 45.31s |
| Decision Tree | 96.26% | 0.9695 | 0.9830 | 0.55s |
| KNN | 99.21% | 0.9915 | 0.9921 | 1.19s |

All tree-based models land at basically the same accuracy (~96.26%). That happens because only 5 low-cardinality features survive after you strip out the leaky columns. KNN hits 99.21% but trades off attack recall to get there.

### NSL-KDD (Official KDDTest+ with Novel Attacks)

| Model | Accuracy | F1-Score | Precision | Time |
|-------|----------|----------|-----------|------|
| Random Forest | 77.17% | 0.7686 | 0.8345 | 8.61s |
| XGBoost | 80.82% | 0.8073 | 0.8527 | 3.67s |
| LightGBM | 80.35% | 0.8023 | 0.8503 | 3.44s |
| Gradient Boosting | 78.25% | 0.7802 | 0.8391 | 47.81s |
| Decision Tree | 77.66% | 0.7740 | 0.8354 | 1.01s |
| KNN | 76.76% | 0.7639 | 0.8347 | 0.12s |

77 to 81%. Way below the ~99% you see in published papers. The difference comes down to one thing: we tested on attacks the models never trained on. That's the reality of deploying a cryptojacking detector where attackers keep changing their techniques.

### What We Learned

1. **Data leakage matters.** Removing identifier columns from DS2OS drops accuracy by ~3%. Studies that kept those columns were essentially memorizing record IDs, not learning attack behavior.
2. **Generalization is the real test.** The 22% accuracy drop on NSL-KDD shows that models tested only on familiar attacks massively overstate how well they'll work in production.
3. **XGBoost and LightGBM hit the sweet spot.** Both achieve top accuracy in 3-4 seconds. Gradient Boosting takes 45+ seconds for no real improvement.
4. **Cross-dataset transfer doesn't work.** 5 IoT features and 41 network features are fundamentally different modalities. You can't just move a model from one to the other.
5. **Class imbalance needs handling.** Without SMOTE on DS2OS, models hit 97% accuracy by predicting everything as normal. Zero attack recall.

## Preprocessing Pipeline

```
Raw Data
    │
    ├── Check for data leakage
    │   └── Remove identifiers correlated with the target
    │
    ├── Label encode categorical features
    │
    ├── Stratified 70/30 train/test split
    │
    ├── SMOTE on training set (DS2OS only, 1:1 ratio)
    │   └── NSL-KDD is already ~53/47, no resampling needed
    │
    └── StandardScaler (zero mean, unit variance)
```

KDDTest+ is used as-is for the NSL-KDD test set. Resampling it would defeat the purpose of testing on novel attacks.

## Reproducing Results

We report single stratified train-test splits to match how the primary studies we're comparing against ran their experiments. That means there's some partition-dependent variance. Treat the numbers as point estimates, not guarantees.

**Environment:** Google Colab standard runtime, Python 3.10, scikit-learn 1.3.0, XGBoost 2.0.0, LightGBM 4.0.0.

## Citation

### Software / Replication Package

```bibtex
@software{chakravorty2026cryptojacking_code,
  title   = {Cryptojacking Validation: AI Against Cloud Cryptojacking (Replication Package)},
  author  = {Chakravorty, Amitabh},
  year    = {2026},
  version = {v1.0.0},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.18565269},
  url     = {https://github.com/AmitabhCh822/cryptojacking-validation}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- University of Cincinnati CECH Impact Accelerator Grant
- Canadian Institute for Cybersecurity (NSL-KDD dataset)
- DS2OS dataset contributors

## Contact

- **Amitabh Chakravorty** - [chakraa4@mail.uc.edu](mailto:chakraa4@mail.uc.edu)
- **Nelly Elsayed** - [elsayeny@ucmail.uc.edu](mailto:elsayeny@ucmail.uc.edu)

---

This repo is part of a systematic literature review. The main takeaway: high accuracy on proxy datasets doesn't mean much if it's driven by data leakage or testing only on known attacks. The field needs public cloud-specific cryptojacking datasets before any of these detection approaches can be taken seriously in production.
