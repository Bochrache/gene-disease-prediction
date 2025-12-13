"""
🏆 WINNING STRATEGY: ALPHA SWEEP + 10-FOLD CV
==============================================

INSIGHT FROM PROGRESS:
- α=0.45 → 0.046358
- α=0.60 → 0.052736 (+13.8% improvement!)
- Trend: HIGHER α = BETTER (trust features more than graph)

HYPOTHESIS: α=0.70-0.85 will perform even better!

Strategy:
1. Test α = [0.65, 0.70, 0.75, 0.80, 0.85]
2. Use 10-fold CV (proven: more folds = better)
3. Keep everything else the same (simple = better)

Expected: 0.055-0.062 range
Target: 0.068

Time: ~40 minutes total
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, normalize as sk_normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🏆 WINNING STRATEGY: ALPHA SWEEP + 10-FOLD CV")
print("="*80)
print(f"Testing α = [0.65, 0.70, 0.75, 0.80, 0.85]")
print(f"Using 10-fold CV for maximum ensemble diversity\n")

# Config
DATA_DIR = Path('dataset')
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

# Alpha values to test (higher = trust features more)
ALPHA_VALUES = [0.65, 0.70, 0.75, 0.80, 0.85]
K_FOLDS = 10  # More folds = better ensemble

# =============================================================================
# LOAD DATA
# =============================================================================
print("📂 Loading data...")
node_features = torch.load(DATA_DIR / 'node_features.pt').numpy()
edge_index = torch.load(DATA_DIR / 'edge_index.pt').numpy()
y = torch.load(DATA_DIR / 'y.pt').numpy()
train_idx = torch.load(DATA_DIR / 'train_idx.pt').numpy()
test_idx = torch.load(DATA_DIR / 'test_idx.pt').numpy()

n_nodes = node_features.shape[0]
n_diseases = y.shape[1]

print(f"  Nodes: {n_nodes:,}")
print(f"  Diseases: {n_diseases}")
print(f"  Train: {len(train_idx):,} | Test: {len(test_idx):,}")

# =============================================================================
# PREPARE FEATURES
# =============================================================================
print("\n📊 Preparing features...")

# Clean NaNs
col_means = np.nanmean(node_features, axis=0)
nan_mask = np.isnan(node_features)
node_features[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
node_features = np.nan_to_num(node_features, nan=0.0)

# Standardize
scaler = StandardScaler()
node_features_scaled = scaler.fit_transform(node_features)

# Graph features
print("  Computing graph features...")
G = nx.Graph()
G.add_nodes_from(range(n_nodes))
G.add_edges_from(edge_index.T.tolist())

try:
    degree_centrality = torch.load(DATA_DIR / 'degree_centrality.pt').numpy()
    pagerank = torch.load(DATA_DIR / 'pagerank.pt').numpy()
except:
    degree_dict = dict(nx.degree_centrality(G))
    degree_centrality = np.array([degree_dict[i] for i in range(n_nodes)])
    pagerank_dict = nx.pagerank(G, max_iter=100)
    pagerank = np.array([pagerank_dict[i] for i in range(n_nodes)])

clustering_dict = nx.clustering(G)
clustering = np.array([clustering_dict[i] for i in range(n_nodes)])

graph_features = np.column_stack([degree_centrality, pagerank, clustering])
features = np.concatenate([node_features_scaled, graph_features], axis=1)

print(f"✅ Total features: {features.shape[1]}")

# =============================================================================
# LOAD AUGMENTED GRAPH
# =============================================================================
print("\n📈 Loading augmented graph...")
edge_index_aug = torch.load('dataset/edge_index_augmented_k20.pt').numpy()
print(f"  Edges: {edge_index_aug.shape[1]:,}")

# Build normalized adjacency
adj = csr_matrix(
    (np.ones(edge_index_aug.shape[1]), (edge_index_aug[0], edge_index_aug[1])), 
    shape=(n_nodes, n_nodes)
)
adj_norm = sk_normalize(adj, norm='l1', axis=1)
print("✅ Graph normalized")

# =============================================================================
# ALPHA SWEEP WITH 10-FOLD CV
# =============================================================================

results = {}

for alpha in ALPHA_VALUES:
    print("\n" + "="*80)
    print(f"🔍 TESTING α = {alpha:.2f}")
    print("="*80)
    
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    
    fold_aps = []
    fold_predictions_test = []
    
    for fold_num, (fold_train_idx, fold_val_idx) in enumerate(kf.split(train_idx), 1):
        print(f"\n  Fold {fold_num}/{K_FOLDS}...", end=" ")
        
        # Get actual indices
        actual_train = train_idx[fold_train_idx]
        actual_val = train_idx[fold_val_idx]
        
        # Train logistic regression (one per disease)
        preds_lr = np.zeros((n_diseases, n_nodes))
        
        for disease_idx in range(n_diseases):
            lr = LogisticRegression(
                C=0.1, 
                max_iter=500, 
                solver='saga',
                random_state=SEED + fold_num,
                n_jobs=1
            )
            lr.fit(features[actual_train], y[actual_train, disease_idx])
            preds_lr[disease_idx] = lr.predict_proba(features)[:, 1]
        
        # Label propagation
        Y_prop = preds_lr.copy()  # Shape: (n_diseases, n_nodes)
        Y_prop[:, actual_train] = y[actual_train].T  # Fix training labels
        
        for iteration in range(100):
            Y_new = alpha * (adj_norm @ Y_prop.T).T + (1 - alpha) * preds_lr
            Y_new[:, actual_train] = y[actual_train].T
            Y_prop = Y_new
        
        # Clip predictions
        preds_fold = np.clip(Y_prop.T, 0, 1)  # Shape: (n_nodes, n_diseases)
        
        # Validation AP
        val_ap = average_precision_score(
            y[actual_val], 
            preds_fold[actual_val],
            average='micro'
        )
        fold_aps.append(val_ap)
        
        # Save test predictions
        fold_predictions_test.append(preds_fold[test_idx])
        
        print(f"Val AP: {val_ap:.4f}")
    
    # Average results
    avg_val_ap = np.mean(fold_aps)
    std_val_ap = np.std(fold_aps)
    
    print(f"\n{'─'*80}")
    print(f"📊 α={alpha:.2f} RESULTS:")
    print(f"   Validation AP: {avg_val_ap:.4f} ± {std_val_ap:.4f}")
    print(f"   Range: {min(fold_aps):.4f} - {max(fold_aps):.4f}")
    print(f"{'─'*80}")
    
    # Ensemble test predictions (simple average of folds)
    test_predictions = np.mean(fold_predictions_test, axis=0)
    
    # Store results
    results[alpha] = {
        'val_ap': avg_val_ap,
        'val_std': std_val_ap,
        'fold_aps': fold_aps,
        'test_preds': test_predictions
    }

# =============================================================================
# FIND BEST ALPHA
# =============================================================================
print("\n" + "="*80)
print("📊 ALPHA COMPARISON")
print("="*80)

best_alpha = max(results.keys(), key=lambda a: results[a]['val_ap'])
best_val_ap = results[best_alpha]['val_ap']

print("\nValidation AP by α:")
for alpha in ALPHA_VALUES:
    val_ap = results[alpha]['val_ap']
    val_std = results[alpha]['val_std']
    marker = "⭐" if alpha == best_alpha else "  "
    print(f"{marker} α={alpha:.2f}: {val_ap:.4f} ± {val_std:.4f}")

print(f"\n🏆 BEST: α={best_alpha:.2f} with Val AP={best_val_ap:.4f}")

# Historical validation→test ratio
historical_ratio = 0.052736 / 0.0837  # Previous best
predicted_test = best_val_ap * historical_ratio

print(f"\n📈 Predicted test score: {predicted_test:.6f}")
print(f"   Current best: 0.052736")
print(f"   Improvement: {predicted_test - 0.052736:+.6f} ({((predicted_test/0.052736 - 1)*100):+.1f}%)")
print(f"   Target: 0.068000")
print(f"   Remaining gap: {0.068 - predicted_test:.6f}")

# =============================================================================
# SAVE BEST SUBMISSION
# =============================================================================
print("\n" + "="*80)
print("💾 Saving Submission")
print("="*80)

best_test_preds = results[best_alpha]['test_preds']

submission_file = OUTPUT_DIR / f'submission_WINNING_alpha{best_alpha:.2f}_10fold.csv'
with open(submission_file, 'w') as f:
    f.write('ID,Predicted\n')
    for i, node_idx in enumerate(test_idx):
        preds_str = ' '.join([f'{p:.6f}' for p in best_test_preds[i]])
        f.write(f'{node_idx},{preds_str}\n')

print(f"✅ Saved: {submission_file}")

# Also save all alphas for comparison
print("\n💾 Saving all alpha results...")
for alpha in ALPHA_VALUES:
    test_preds = results[alpha]['test_preds']
    submission_file = OUTPUT_DIR / f'submission_alpha{alpha:.2f}_10fold.csv'
    
    with open(submission_file, 'w') as f:
        f.write('ID,Predicted\n')
        for i, node_idx in enumerate(test_idx):
            preds_str = ' '.join([f'{p:.6f}' for p in test_preds[i]])
            f.write(f'{node_idx},{preds_str}\n')
    
    print(f"   α={alpha:.2f}: {submission_file.name}")

print("\n" + "="*80)
print("✅ COMPLETE!")
print("="*80)
print(f"""
NEXT STEPS:
1. Submit best file: submission_WINNING_alpha{best_alpha:.2f}_10fold.csv
2. Expected score: {predicted_test:.6f}
3. If score ≥ 0.060: Try ensemble of alphas
4. If score < 0.060: Try calibration or feature selection
""")
