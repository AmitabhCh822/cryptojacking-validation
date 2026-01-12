# GitHub Repository Setup Guide

## Complete Guide to Uploading Your Cryptojacking Validation Code

This guide walks you through setting up your GitHub repository properly for the empirical validation study.

---

## Prerequisites

1. **GitHub Account**: Create one at https://github.com if you don't have one
2. **Git Installed**: Download from https://git-scm.com/downloads
3. **Your Files**: The 4 Jupyter notebooks (1_Master.ipynb through 4_Models.ipynb)

---

## Step 1: Create a New GitHub Repository

1. Go to https://github.com/new
2. Fill in the details:
   - **Repository name**: `cryptojacking-validation`
   - **Description**: `Empirical validation for AI-based cloud cryptojacking detection SLR`
   - **Visibility**: Public (for academic reproducibility)
   - **Initialize**: Do NOT check "Add a README file" (we'll add our own)
   - **Add .gitignore**: None (we have our own)
   - **Choose a license**: MIT License

3. Click **Create repository**

---

## Step 2: Clone and Set Up Locally

Open your terminal/command prompt and run:

```bash
# Navigate to where you want the project
cd ~/Documents  # or any preferred location

# Clone the empty repository
git clone https://github.com/AmitabhCh822/cryptojacking-validation.git
cd cryptojacking-validation
```

---

## Step 3: Create the Directory Structure

```bash
# Create all necessary directories
mkdir -p notebooks data/raw data/processed models results/figures results/metrics scripts docs

# Create .gitkeep files to preserve empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep
touch results/figures/.gitkeep
touch results/metrics/.gitkeep
```

---

## Step 4: Copy Your Files

### Option A: Manual Copy (Recommended)

1. Copy your 4 notebooks to the `notebooks/` folder:
   - `1_Master.ipynb` → `notebooks/1_Master.ipynb`
   - `2_Exploration.ipynb` → `notebooks/2_Exploration.ipynb`
   - `3_Preprocessing.ipynb` → `notebooks/3_Preprocessing.ipynb`
   - `4_Models.ipynb` → `notebooks/4_Models.ipynb`

2. Download the files I created from this conversation:
   - `README.md` → root directory
   - `requirements.txt` → root directory
   - `LICENSE` → root directory
   - `.gitignore` → root directory
   - `CONTRIBUTING.md` → root directory
   - `docs/METHODOLOGY.md` → docs directory
   - `scripts/utils.py` → scripts directory

### Option B: Command Line (if files are in Downloads)

```bash
# Assuming notebooks are in ~/Downloads
cp ~/Downloads/1_Master.ipynb notebooks/
cp ~/Downloads/2_Exploration.ipynb notebooks/
cp ~/Downloads/3_Preprocessing.ipynb notebooks/
cp ~/Downloads/3_Preprocessing.ipynb notebooks/
cp ~/Downloads/4_Models.ipynb notebooks/
```

---

## Step 5: Create the Support Files

Create each of these files (content provided in the conversation):

### README.md
The main documentation file - copy the full content from my earlier response.

### requirements.txt
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
tensorflow>=2.13.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
notebook>=7.0.0
openpyxl>=3.1.0
kaggle>=1.5.0
tqdm>=4.65.0
```

### .gitignore
Copy the content from my earlier response (includes data files, model files, credentials, etc.)

---

## Step 6: Verify Your Structure

Your repository should look like this:

```
cryptojacking-validation/
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── docs/
│   └── METHODOLOGY.md
├── models/
│   └── .gitkeep
├── notebooks/
│   ├── 1_Master.ipynb
│   ├── 2_Exploration.ipynb
│   ├── 3_Preprocessing.ipynb
│   └── 4_Models.ipynb
├── results/
│   ├── figures/
│   │   └── .gitkeep
│   └── metrics/
│       └── .gitkeep
└── scripts/
    └── utils.py
```

Verify with:
```bash
ls -la
ls -la notebooks/
ls -la docs/
```

---

## Step 7: Stage and Commit Files

```bash
# Add all files to git
git add .

# Check what will be committed
git status

# Commit with a meaningful message
git commit -m "Initial commit: Add empirical validation code for cryptojacking SLR"
```

---

## Step 8: Push to GitHub

```bash
# Push to the main branch
git push -u origin main
```

If you get an error about "main" vs "master":
```bash
git branch -M main
git push -u origin main
```

---

## Step 9: Verify on GitHub

1. Go to https://github.com/AmitabhCh822/cryptojacking-validation
2. Check that all files appear correctly
3. Click on README.md to verify it renders properly
4. Check the notebooks folder

---

## Step 10: Add Repository Topics (Optional but Recommended)

On your GitHub repository page:
1. Click the gear icon next to "About"
2. Add topics: `cryptojacking`, `machine-learning`, `cybersecurity`, `cloud-security`, `intrusion-detection`, `systematic-review`
3. Add description and website if applicable

---

## Troubleshooting

### "Permission denied" error
```bash
# Use HTTPS instead of SSH, or set up SSH keys
git remote set-url origin https://github.com/AmitabhCh822/cryptojacking-validation.git
```

### Large file warnings
The notebooks with outputs can be large. If you get warnings:
```bash
# Clear notebook outputs before committing (optional)
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

### Authentication issues
GitHub now requires personal access tokens instead of passwords:
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token with "repo" permissions
3. Use this token as your password when pushing

---

## Updating Your Repository

After making changes:
```bash
git add .
git commit -m "Description of changes"
git push
```

---

## Making a Release (For Paper Submission)

When your paper is submitted:
1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Tag version: `v1.0.0`
4. Release title: `Paper Submission Version`
5. Add description of the release
6. Publish release

This creates a permanent, citable version of your code.

---

## Need Help?

- GitHub Docs: https://docs.github.com
- Git Tutorial: https://git-scm.com/doc
- Contact: chakraa4@mail.uc.edu
