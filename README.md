# Cryptojacking Detection Validation

**Empirical Validation for AI-Based Cloud Cryptojacking Detection: A Systematic Literature Review**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## Overview

This repository contains the empirical validation code and materials for the systematic literature review paper:

> **AI-Based Detection of Cloud Cryptojacking: A Systematic Review of Models, Deployment Challenges, and Future Directions**  
> *Amitabh Chakravorty, Nelly Elsayed*  
> School of Information Technology, University of Cincinnati

The validation study evaluates representative machine learning models from the reviewed literature using publicly available datasets to assess detection performance, computational cost, and reproducibility challenges in AI-based cryptojacking detection.

## Key Findings

| Dataset | Best Model | Accuracy | F1-Score | Training Time |
|---------|-----------|----------|----------|---------------|
| DS2OS | Random Forest | 99.59% | 0.9959 | 47.95s |
| NSL-KDD | XGBoost | 99.62% | 0.9962 | 54.49s |

**Important Note:** These datasets serve as *proxy environments* for cloud cryptojacking detection. No publicly available datasets capture genuine cloud VM, container, or Kubernetes telemetry with labeled cryptomining activity—a key finding of our systematic review.

## Repository Structure

```
cryptojacking-validation/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
│
├── notebooks/                   # Jupyter notebooks (run in order)
│   ├── 1_Master.ipynb          # Environment setup & data download
│   ├── 2_Exploration.ipynb     # Dataset exploration & visualization
│   ├── 3_Preprocessing.ipynb   # Data preprocessing & SMOTE
│   └── 4_Models.ipynb          # Model training & evaluation
│
├── data/                        # Data directory (created by notebooks)
│   ├── raw/                    # Original downloaded datasets
│   └── processed/              # Preprocessed numpy arrays
│
├── models/                      # Trained model files (.pkl)
│
├── results/                     # Output files
│   ├── figures/                # Generated visualizations
│   └── metrics/                # Performance metrics (CSV)
│
├── scripts/                     # Utility scripts
│   └── utils.py                # Helper functions
│
└── docs/                        # Additional documentation
    └── METHODOLOGY.md          # Detailed methodology description
```

## Quick Start

### Option 1: Google Colab (Recommended)

1. Open notebooks directly in Google Colab by clicking the badge above
2. Run notebooks in order: `1_Master.ipynb` → `2_Exploration.ipynb` → `3_Preprocessing.ipynb` → `4_Models.ipynb`
3. You'll need a Kaggle account and API key for data download

### Option 2: Local Environment

```bash
# Clone the repository
git clone https://github.com/AmitabhCh822/cryptojacking-validation.git
cd cryptojacking-validation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks with Jupyter
jupyter notebook
```

## Datasets

### DS2OS (Distributed Smart Space Orchestration System)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/libamariyam/ds2os-dataset)
- **Samples:** 357,952 records
- **Features:** 12 (IoT device telemetry)
- **Classes:** Normal vs. Anomalous (8 attack types)
- **Imbalance Ratio:** 97.2% normal, 2.8% attack

### NSL-KDD
- **Source:** [UNB CIC](https://www.unb.ca/cic/datasets/nsl.html)
- **Train Samples:** 125,973
- **Test Samples:** 22,544
- **Features:** 41 (network traffic patterns)
- **Classes:** Normal vs. Attack (22 attack categories → binary)

## Models Evaluated

Based on the most frequently reported approaches in the systematic review:

| Model | Hyperparameters |
|-------|-----------------|
| Random Forest | n_estimators=100, max_depth=20 |
| XGBoost | n_estimators=100, max_depth=10, lr=0.1 |
| LightGBM | n_estimators=100, max_depth=10 |
| Decision Tree | max_depth=15, min_samples_split=5 |
| K-Nearest Neighbors | n_neighbors=5 |
| Gradient Boosting | n_estimators=100, max_depth=5, lr=0.1 |

## Results Summary

### Performance Metrics

| Dataset | Model | Accuracy | F1-Score | Precision | Recall | Train Time (s) |
|---------|-------|----------|----------|-----------|--------|----------------|
| DS2OS | Random Forest | 99.59% | 0.9959 | 0.9959 | 0.9959 | 47.95 |
| DS2OS | XGBoost | 99.53% | 0.9953 | 0.9953 | 0.9953 | 88.71 |
| DS2OS | LightGBM | 97.42% | 0.9742 | 0.9746 | 0.9742 | 18.67 |
| DS2OS | Gradient Boosting | 99.52% | 0.9952 | 0.9952 | 0.9952 | 184.90 |
| DS2OS | Decision Tree | 99.45% | 0.9945 | 0.9945 | 0.9945 | 3.74 |
| DS2OS | KNN | 97.47% | 0.9745 | 0.9748 | 0.9745 | 48.01 |
| NSL-KDD | Random Forest | 99.33% | 0.9933 | 0.9934 | 0.9933 | 48.09 |
| NSL-KDD | XGBoost | 99.62% | 0.9962 | 0.9962 | 0.9962 | 54.49 |
| NSL-KDD | LightGBM | 99.47% | 0.9947 | 0.9948 | 0.9947 | 24.87 |
| NSL-KDD | Gradient Boosting | 99.47% | 0.9947 | 0.9948 | 0.9947 | 121.88 |
| NSL-KDD | Decision Tree | 99.09% | 0.9909 | 0.9911 | 0.9909 | 1.37 |
| NSL-KDD | KNN | 96.99% | 0.9694 | 0.9722 | 0.9694 | 72.94 |

### Key Observations

1. **Class Imbalance Impact:** DS2OS required SMOTE (97% normal → 50/50 split) to prevent majority-class bias
2. **Computational Trade-offs:** Decision Tree fastest (1.37-3.74s) but slightly lower accuracy; Gradient Boosting slowest (122-185s) with marginal gains
3. **Cross-Dataset Generalization:** Feature space incompatibility (12 vs 41 features) prevented direct model transfer
4. **Reproducibility Challenges:** Minor preprocessing differences produced 0.5-2.5% accuracy variations

## Preprocessing Pipeline

```
Raw Data
    │
    ├── Label Encoding (categorical → numeric)
    │
    ├── Stratified Train/Test Split (70/30)
    │
    ├── SMOTE (if imbalance ratio < 0.3)
    │
    └── StandardScaler (zero mean, unit variance)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{chakravorty2025cryptojacking,
  title={AI-Based Detection of Cloud Cryptojacking: A Systematic Review of Models, Deployment Challenges, and Future Directions},
  author={Chakravorty, Amitabh and Elsayed, Nelly},
  journal={Journal of Information Security and Applications},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- University of Cincinnati CECH Impact Accelerator Grant
- Canadian Institute for Cybersecurity (NSL-KDD dataset)
- DS2OS dataset contributors

## Contact

- **Amitabh Chakravorty** - [chakraa4@mail.uc.edu](mailto:chakraa4@mail.uc.edu)
- **Nelly Elsayed** - [elsayeny@ucmail.uc.edu](mailto:elsayeny@ucmail.uc.edu)

---

**Note:** This repository is part of a systematic literature review. The validation demonstrates that while high accuracy is achievable on proxy datasets, the absence of public cloud-specific cryptojacking datasets remains the field's most critical reproducibility barrier.
