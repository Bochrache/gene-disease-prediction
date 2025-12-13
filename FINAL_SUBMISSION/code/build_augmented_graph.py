"""
Build and save augmented graph (original + k-NN) to avoid recomputing each time.
Saves edge_index_augmented.pt to dataset/ folder.
"""

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import os

print("=" * 80)
print("BUILDING AUGMENTED GRAPH (ONE-TIME COMPUTATION)")
print("=" * 80)

# Load data
print("\n1. Loading Data...")
edge_index = torch.load('dataset/edge_index.pt')
node_features = torch.load('dataset/node_features_cleaned.pt')

num_nodes = node_features.shape[0]
print(f"   Nodes: {num_nodes}")
print(f"   Original edges: {edge_index.shape[1]}")
print(f"   Features: {node_features.shape[1]}")

# Build original adjacency matrix
print("\n2. Building Original Adjacency Matrix...")
adj = csr_matrix((np.ones(edge_index.shape[1]), 
                  (edge_index[0].numpy(), edge_index[1].numpy())),
                 shape=(num_nodes, num_nodes))
adj = adj + adj.T  # Make symmetric
adj = (adj > 0).astype(float)
print(f"   Original adjacency shape: {adj.shape}")
print(f"   Original edges (undirected): {adj.nnz}")

# Build k-NN graph
k = 20
print(f"\n3. Building k-NN Graph (k={k})...")
print("   Fitting NearestNeighbors (this may take a few minutes)...")
nn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=1)
nn_model.fit(node_features.numpy())

print("   Finding k-nearest neighbors...")
distances, indices = nn_model.kneighbors(node_features.numpy())

# Build k-NN adjacency
print("   Building k-NN adjacency matrix...")
knn_edges = []
for node_idx in range(num_nodes):
    for neighbor in indices[node_idx, 1:]:  # Skip self (index 0)
        knn_edges.append([node_idx, neighbor])
        knn_edges.append([neighbor, node_idx])  # Symmetric

knn_edges = np.array(knn_edges).T
print(f"   k-NN edges: {knn_edges.shape[1]}")

# Combine original + k-NN
print("\n4. Combining Original + k-NN Graphs...")
edge_index_augmented = np.concatenate([edge_index.numpy(), knn_edges], axis=1)

# Remove duplicates
print("   Removing duplicate edges...")
edge_set = set()
unique_edges = []
for i in range(edge_index_augmented.shape[1]):
    src, dst = edge_index_augmented[0, i], edge_index_augmented[1, i]
    edge = (min(src, dst), max(src, dst))
    if edge not in edge_set:
        edge_set.add(edge)
        unique_edges.append([src, dst])
        if src != dst:
            unique_edges.append([dst, src])

edge_index_augmented = torch.tensor(np.array(unique_edges).T, dtype=torch.long)

print(f"\n5. Final Statistics:")
print(f"   Original edges: {edge_index.shape[1]:,}")
print(f"   k-NN edges: {knn_edges.shape[1]:,}")
print(f"   Combined edges (after deduplication): {edge_index_augmented.shape[1]:,}")
print(f"   Added edges: {edge_index_augmented.shape[1] - edge_index.shape[1]:,}")

# Save
output_path = 'dataset/edge_index_augmented_k20.pt'
torch.save(edge_index_augmented, output_path)
print(f"\n✅ Saved augmented graph to: {output_path}")

# Also save k-NN only for reference
knn_only_path = 'dataset/edge_index_knn_k20.pt'
torch.save(torch.tensor(knn_edges, dtype=torch.long), knn_only_path)
print(f"✅ Saved k-NN graph to: {knn_only_path}")

print("\n" + "=" * 80)
print("DONE! You can now load the augmented graph with:")
print("  edge_index_aug = torch.load('dataset/edge_index_augmented_k20.pt')")
print("=" * 80)
