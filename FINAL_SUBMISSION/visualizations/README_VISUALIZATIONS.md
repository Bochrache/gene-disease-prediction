# Visualizations Index

This folder contains all data visualizations for the Gene-Disease Prediction competition submission.

---

## 📊 Generated Visualizations (Custom Analysis)

### 1. `alpha_sweep_results.png`
**Type:** Line plot with dual axes  
**Purpose:** Shows the impact of alpha hyperparameter on validation and test performance  
**Key Insights:**
- Validation AP peaks at α=0.80 (0.0878)
- Test scores correlate well with validation
- Optimal α is higher than typical (0.80 vs 0.45 baseline)
- Demonstrates that trusting graph structure (80%) outperforms trusting features (20%)

**Axes:**
- X-axis: Alpha values [0.65, 0.70, 0.75, 0.80, 0.85]
- Y-axis: Average Precision (AP)
- Two series: Validation AP (blue circles) and Test AP (pink squares)

---

### 2. `kfold_validation_results.png`
**Type:** Bar chart with mean line and std bands  
**Purpose:** Demonstrates consistency of 10-fold cross-validation  
**Key Insights:**
- Very low variance across folds (std = 0.0004)
- Mean validation AP = 0.0878
- All folds within ±0.5% of mean
- Proves model is robust, not dependent on specific train/val split

**Axes:**
- X-axis: Fold numbers 1-10
- Y-axis: Validation Average Precision
- Red dashed line: Mean AP across folds
- Red shaded region: ±1 standard deviation

---

### 3. `score_progression_timeline.png`
**Type:** Dual-panel stacked visualization  
**Purpose:** Shows complete evolution from baseline to final model  
**Key Insights:**
- Total improvement: +34.6% over baseline
- Major breakthrough: K-fold CV (+17.3%)
- Second breakthrough: Higher alpha (+13.8%)
- Incremental gains: Alpha sweep (+0.4%)
- Demonstrates systematic optimization process

**Top Panel:**
- Bar chart of absolute test scores
- Green dashed line: Target score (0.070)
- Annotations: Percentage improvements between stages

**Bottom Panel:**
- Line plot of cumulative improvement percentage
- Shows acceleration curve toward final result

---

### 4. `network_statistics.png`
**Type:** Multi-panel dashboard (6 subplots)  
**Purpose:** Comprehensive dataset overview and statistics  
**Key Insights:**
- Network: 19,765 nodes, 2.33M edges (50% augmented)
- Labels: 98.9% sparse (highly imbalanced)
- Disease distribution: Long-tail (few diseases have many genes)
- Train/test split: 60/40 approximately

**Subplots:**
1. **Top:** Dataset statistics text summary
2. **Middle Left:** Disease label distribution histogram
3. **Middle Right:** Gene-disease association distribution
4. **Bottom Left:** Graph augmentation comparison (before/after k-NN)
5. **Bottom Right:** Train/test split pie chart

---

### 5. `methodology_flowchart.png`
**Type:** Text-based flowchart diagram  
**Purpose:** Visual representation of complete pipeline  
**Key Insights:**
- 5-step process from preprocessing to predictions
- Clear data flow and transformations
- Key hyperparameters annotated (α=0.80, C=0.1, k=20)
- Mathematical formulas for label propagation

**Sections:**
1. Data Preprocessing
2. Base Model Training
3. Label Propagation
4. K-Fold Cross-Validation
5. Final Predictions
6. Key Insights box

---

## 📈 Copied Visualizations (From Project Analysis)

### 6. `degree_distribution.png`
**Type:** Log-log histogram  
**Purpose:** Shows PPI network degree distribution  
**Key Insights:**
- Power-law distribution (scale-free network)
- Few hub genes with very high connectivity
- Most genes have low degree (1-10 connections)
- Confirms biological PPI networks follow small-world properties

**Interpretation:** Hub genes are critical for disease associations (spread information widely)

---

### 7. `label_distribution.png`
**Type:** Histogram of disease prevalence  
**Purpose:** Shows imbalanced disease label distribution  
**Key Insights:**
- Most diseases have < 50 associated genes
- Few diseases have > 200 associated genes
- Extreme imbalance challenges traditional ML
- Justifies use of Average Precision metric (handles imbalance)

**Statistics:**
- Min: 3 genes per disease
- Max: 412 genes per disease
- Mean: 52.3 genes per disease
- Median: 34 genes per disease

---

### 8. `feature_importance_logistic_regression.png`
**Type:** Horizontal bar chart  
**Purpose:** Shows feature importance from base logistic regression  
**Key Insights:**
- Graph centrality features dominate (63% total importance)
- Degree centrality is most important (0.245)
- Genomic features are secondary (gene length, GC content)
- Justifies using graph structure in label propagation

**Top Features:**
1. Degree Centrality (24.5%)
2. PageRank (19.8%)
3. Betweenness Centrality (18.7%)
4. Gene Length (14.2%)
5. GC Content (8.9%)

---

### 9. `final_model_comparison.png`
**Type:** Grouped bar chart comparing methods  
**Purpose:** Compares all tested approaches (deep learning, classical ML, label propagation)  
**Key Insights:**
- Label Propagation >> GNN models (GCN, GAT, APPNP)
- Deep learning severely overfits (Train 0.95+ → Test 0.04)
- Simple methods outperform complex on sparse labels
- Validates winning strategy choice

**Models Compared:**
- GCN: Train 0.9532, Test 0.0382
- GAT: Train 0.9687, Test 0.0368
- MLP: Train 0.8234, Test 0.0415
- **Label Propagation: Train 0.9234, Test 0.0532** ✅

---

## 🎨 Visualization Guidelines

### Color Scheme
- **Blue tones (#2E86AB, #457B9D):** Validation data, primary results
- **Pink/Purple (#A23B72, #E63946):** Test data, Kaggle scores
- **Green (#06A77D):** Positive results, improvements
- **Red/Orange (#E63946, #F77F00):** Warnings, baselines
- **Yellow/Gold:** Highlights, best configurations

### Typography
- **Font:** Default matplotlib (DejaVu Sans)
- **Title Size:** 16pt, bold
- **Axis Labels:** 14pt, bold
- **Annotations:** 11-12pt, regular or bold
- **Grid:** Light gray, alpha=0.3

### Quality
- **Resolution:** 300 DPI (publication quality)
- **Format:** PNG with transparent backgrounds where applicable
- **Size:** 10-14 inches width, 6-10 inches height
- **Style:** Seaborn "whitegrid" for clean, academic look

---

## 📐 How to Regenerate

All visualizations can be regenerated by running:

```bash
cd ../code
python generate_visualizations.py
```

**Requirements:**
- numpy
- matplotlib
- seaborn
- torch (for loading dataset)
- scipy (for label propagation visualization)

**Runtime:** ~30 seconds for all 5 generated plots

**Note:** Plots 6-9 are copied from the main project `figures/` folder and represent earlier analysis during model development.

---

## 🔍 Reading the Visualizations

### For Technical Audience
- Focus on quantitative metrics (AP scores, fold variance)
- Examine validation-test correlation in alpha sweep
- Note low std in k-fold results (robustness indicator)
- Compare deep learning overfitting in model comparison

### For Non-Technical Audience
- **Alpha sweep:** Higher is better, peak at 0.80
- **K-fold bars:** All bars similar height = consistent model
- **Score progression:** Upward trend = improving performance
- **Network stats:** Many small numbers, few big ones (imbalanced data)

### For Academic Reviewers
- **Statistical significance:** Low variance, reproducible results
- **Methodological rigor:** 10-fold CV, systematic hyperparameter search
- **Biological validity:** Power-law degree distribution matches literature
- **Ablation studies:** Each stage contribution measured

---

## 📊 Data Sources

All visualizations are based on:
- **Primary Data:** Kaggle Gene-Disease Prediction dataset
- **Validation Results:** 10-fold cross-validation on training set
- **Test Results:** Kaggle public leaderboard submissions
- **Computational Resources:** MacBook M4, Python 3.13.6

**Data Integrity:** All results are reproducible with fixed random seeds (seed=42)

---

**Last Updated:** December 12, 2025  
**Total Visualizations:** 9 (5 generated + 4 copied)  
**Combined File Size:** ~3.2 MB
