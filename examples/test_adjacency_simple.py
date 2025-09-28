import sys
import pandas as pd
import numpy as np

def adjacency_to_links(adjacency_matrix, node_labels=None, include_zeros=False):
    if isinstance(adjacency_matrix, pd.DataFrame):
        if node_labels is None:
            node_labels = adjacency_matrix.columns.tolist()
        matrix = adjacency_matrix.values
    else:
        matrix = np.array(adjacency_matrix)
        if node_labels is None:
            node_labels = list(range(matrix.shape[0]))

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency matrix must be square")

    if len(node_labels) != matrix.shape[0]:
        raise ValueError("node_labels length must match matrix dimensions")

    links = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            weight = matrix[i, j]
            if include_zeros or weight != 0:
                links.append({
                    'from': node_labels[i],
                    'to': node_labels[j],
                    'weight': weight
                })

    return pd.DataFrame(links)

def links_to_adjacency(df, weight_column=None):
    if 'from' not in df.columns or 'to' not in df.columns:
        raise ValueError("DataFrame must contain 'from' and 'to' columns")

    all_nodes = sorted(set(df['from'].unique()) | set(df['to'].unique()))
    n = len(all_nodes)

    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    matrix = np.zeros((n, n))

    for _, row in df.iterrows():
        i = node_to_idx[row['from']]
        j = node_to_idx[row['to']]
        weight = row[weight_column] if weight_column and weight_column in df.columns else 1
        matrix[i, j] = weight

    adjacency_df = pd.DataFrame(matrix, index=all_nodes, columns=all_nodes)
    return adjacency_df

print("=" * 60)
print("Test 1: Convert adjacency matrix (numpy) to links")
print("=" * 60)

adj_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
])

print("\nOriginal adjacency matrix:")
print(adj_matrix)

links = adjacency_to_links(adj_matrix)
print("\nConverted to links (excluding zeros):")
print(links)

print("\n" + "=" * 60)
print("Test 2: Convert adjacency matrix with custom labels")
print("=" * 60)

node_labels = ['A', 'B', 'C', 'D']
links_labeled = adjacency_to_links(adj_matrix, node_labels=node_labels)
print("\nLinks with custom labels:")
print(links_labeled)

print("\n" + "=" * 60)
print("Test 3: Convert links back to adjacency matrix")
print("=" * 60)

reconstructed = links_to_adjacency(links_labeled, weight_column='weight')
print("\nReconstructed adjacency matrix:")
print(reconstructed)

print("\n" + "=" * 60)
print("Test 4: Verify round-trip conversion")
print("=" * 60)

original_values = adj_matrix
reconstructed_values = reconstructed.values

print("\nMatrices match:", np.allclose(original_values, reconstructed_values))

print("\n" + "=" * 60)
print("Test 5: Convert DataFrame adjacency matrix")
print("=" * 60)

adj_df = pd.DataFrame(adj_matrix,
                      index=['Node1', 'Node2', 'Node3', 'Node4'],
                      columns=['Node1', 'Node2', 'Node3', 'Node4'])

print("\nDataFrame adjacency matrix:")
print(adj_df)

links_from_df = adjacency_to_links(adj_df)
print("\nConverted to links:")
print(links_from_df)

print("\n" + "=" * 60)
print("Test 6: Weighted graph conversion")
print("=" * 60)

weighted_matrix = np.array([
    [0, 0.5, 0.8, 0],
    [0.5, 0, 0.3, 1.0],
    [0, 0, 0, 0.7],
    [0.9, 0, 0, 0]
])

weighted_links = adjacency_to_links(weighted_matrix, node_labels=['X', 'Y', 'Z', 'W'])
print("\nWeighted links:")
print(weighted_links)

reconstructed_weighted = links_to_adjacency(weighted_links, weight_column='weight')
print("\nReconstructed weighted adjacency matrix:")
print(reconstructed_weighted)

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)