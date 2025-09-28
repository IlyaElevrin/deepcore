"""
deepcore
classes:
- SyntheticGenerator: https://github.com/IlyaElevrin/synthetic_data_generation
- sort_duoblet
- cluster_links
"""

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import warnings
import networkx as nx
from community import community_louvain
warnings.filterwarnings('ignore')

__version__ = "0.1.0"
__all__ = ['SyntheticGenerator', 'sort_duoblet', 'cluster_links', 'adjacency_to_links', 'links_to_adjacency']  # Exported classes

class SyntheticDataGenerator:
    def __init__(self, model_path="deeplinks/ctgan_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.original_columns = []

    def _validate_input(self, data):
        required_columns = ['subject', 'verb', 'object']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"The mandatory column '{col}' is missing from the data")
        self.original_columns = data.columns.tolist()  # preserving the original speakers

    def _generate_dummy_columns(self, data):
        # One-hot for 'verb' and 'object'
        for col in ['verb', 'object']:
            for value in data[col].unique():
                dummy_col = f"{col}_{value}"
                data[dummy_col] = (data[col] == value).astype(int)

        # fictitious numeric columns
        data['amount'] = np.random.uniform(10, 1000, size=len(data))
        data['transaction_id'] = np.arange(len(data))

        # timestamps
        data['timestamp'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(
            np.random.randint(0, 365, len(data)), 'days'
        )

        # masks and flags
        data['is_valid'] = 1
        data['fraud_flag'] = 0

        return data

    def _train_model(self, data, epochs=100):
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)
        self.model = CTGANSynthesizer(self.metadata, epochs=epochs)
        self.model.fit(data)

        if self.model_path:
            self.model.save(self.model_path)

    def generate(self, input_data, num_rows=100, retrain_on_error=True):
        try:
            self._validate_input(input_data)
            enriched_data = self._generate_dummy_columns(input_data.copy())

            if not self.model:
                if self.model_path:
                    self.model = CTGANSynthesizer.load(self.model_path)
                else:
                    self._train_model(enriched_data)

            synthetic_data = self.model.sample(num_rows=num_rows)

            # restore original values from one-hot
            for col_type in ['verb', 'object']:
                for col in synthetic_data.columns:
                    if col.startswith(f"{col_type}_"):
                        mask = synthetic_data[col] == 1
                        synthetic_data.loc[mask, col_type] = col.replace(f"{col_type}_", "")

            # filtering of source columns only
            result = synthetic_data[self.original_columns].copy()

            return result

        except Exception as e:
            if retrain_on_error:
                print(f"Error: {str(e)}\nBegin retraining the model...")
                self._train_model(enriched_data)
                return self.generate(input_data, num_rows, retrain_on_error=False)
            else:
                raise RuntimeError("Failed to generate data after retraining")


#-------------------------------------------------------------------------

def sort_duoblet(df):
    # 1. closed links (from == to)
    closed = df[df['from'] == df['to']].copy()
    closed_sorted = closed.sort_values(by='from').reset_index(drop=True)

    # set of closed nodes
    closed_nodes = set(closed['from'])

    # 2. links between closed nodes (from and to in closed_nodes, but from != to)
    between_closed = df[
        (df['from'].isin(closed_nodes)) & 
        (df['to'].isin(closed_nodes)) & 
        (df['from'] != df['to'])
    ].copy()
    between_closed_sorted = between_closed.sort_values(by=['from', 'to']).reset_index(drop=True)

    # 3. other links (at least one node not in closed_nodes)
    other = df[
        ~(df['from'].isin(closed_nodes) & df['to'].isin(closed_nodes))
    ].copy()
    other_sorted = other.sort_values(by=['from', 'to']).reset_index(drop=True)

    # combine all the pieces
    result = pd.concat([closed_sorted, between_closed_sorted, other_sorted], ignore_index=True)
    return result

#-------------------------------------------------------------------------

def cluster_links(df):
    """
    link clustering by Louvain's algorithm.
    
    parameters:
        df: DataFrame with 'from' and 'to' columns.
        
    returns:
        A dictionary of the form {'from→to': cluster_id}.
    """
    # create unique link identifiers
    df['link'] = df['from'].astype(str) + '→' + df['to'].astype(str)
    
    # Строим граф связей
    G = nx.Graph()
    G.add_nodes_from(df['link'])
    
    # add edges between links if they have common nodes
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            # decompose the links into components
            a_from, a_to = df.iloc[i]['from'], df.iloc[i]['to']
            b_from, b_to = df.iloc[j]['from'], df.iloc[j]['to']
            
            # links are connected if they have at least one common node
            if ({a_from, a_to} & {b_from, b_to}):
                link_a = df.iloc[i]['link']
                link_b = df.iloc[j]['link']
                G.add_edge(link_a, link_b)
    
    # applying Louvain's algorithm
    partition = community_louvain.best_partition(G)

    return partition

#-------------------------------------------------------------------------

def adjacency_to_links(adjacency_matrix, node_labels=None, include_zeros=False):
    """
    converts an adjacency matrix to associative links format.

    parameters:
        adjacency_matrix: numpy array or DataFrame representing the adjacency matrix.
        node_labels: list of node labels. If None, uses indices.
        include_zeros: if True, includes links with zero weight. Default is False.

    returns:
        DataFrame with columns 'from', 'to', and optionally 'weight'.
    """
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
                })
                
    df = pd.DataFrame(links)
                
    df = df.iloc[df.apply(lambda x: (x['from'] != x['to'], x['from'], x['to']), axis=1).argsort()].reset_index(drop=True)

    return df

#-------------------------------------------------------------------------

def links_to_adjacency(df, weight_column=None):
    """
    converts associative links to an adjacency matrix.

    parameters:
        df: DataFrame with 'from' and 'to' columns.
        weight_column: name of the column to use as edge weights. If None, uses 1 for all edges.

    returns:
        DataFrame representing the adjacency matrix with node labels as index and columns.
    """
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
