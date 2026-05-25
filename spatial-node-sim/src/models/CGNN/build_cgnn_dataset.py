"""
Conditional GNN (CGNN) Dataset Builder

Prepares mandi network data for temporal graph neural network training:
- Creates temporal snapshots with lookback windows
- Builds graph structure from neighbor relationships
- Handles multi-step prediction targets
- Supports node-specific and global prediction tasks
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import argparse


class CGNNDatasetBuilder:
    """
    Build temporal graph datasets for Conditional GNN training.
    
    Features:
    - Temporal lookback windows (use past k timesteps)
    - Graph structure from neighbor relationships
    - Node-specific or network-wide prediction
    - Feature normalization
    - Train/val/test splitting
    """
    
    def __init__(
        self,
        csv_path: str,
        lookback: int = 10,
        horizon: int = 1,
        target_node: Optional[int] = None,
        feature_cols: Optional[List[str]] = None
    ):
        """
        Initialize dataset builder.
        
        Args:
            csv_path: Path to mandi dataset CSV
            lookback: Number of past timesteps to use (k)
            horizon: Number of future timesteps to predict
            target_node: Specific mandi ID to predict (None = all nodes)
            feature_cols: List of feature column names
        """
        self.csv_path = csv_path
        self.lookback = lookback
        self.horizon = horizon
        self.target_node = target_node
        
        # Default features
        if feature_cols is None:
            self.feature_cols = [
                'price_state',
                'weather_effect',
                'supply',
                'demand',
                'storage',
                'neighbor_mean_price',
                'degree'
            ]
        else:
            self.feature_cols = feature_cols
        
        self.df = None
        self.graphs = []
        self.scalers = {}
        self.node_mapping = {}
        
    def load_data(self):
        """Load and preprocess CSV data"""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        
        # Sort by time and node
        self.df = self.df.sort_values(['time', 'node_id']).reset_index(drop=True)
        
        print(f"✓ Loaded {len(self.df):,} rows")
        print(f"  Time range: {self.df['time'].min()} → {self.df['time'].max()}")
        print(f"  Unique nodes: {self.df['node_id'].nunique()}")
        print(f"  Timesteps: {self.df['time'].nunique()}")
        
        # Create node ID mapping (for consistent indexing)
        unique_nodes = sorted(self.df['node_id'].unique())
        self.node_mapping = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
        self.reverse_mapping = {idx: node_id for node_id, idx in self.node_mapping.items()}
        
        print(f"  Node mapping: {len(self.node_mapping)} nodes")
    
    def build_edge_index(self, time_snapshot: pd.DataFrame) -> torch.Tensor:
        """
        Build edge index from neighbor relationships.
        
        Args:
            time_snapshot: DataFrame for single timestep
        
        Returns:
            edge_index: [2, num_edges] tensor
        """
        edges = []
        
        # We need to infer edges from neighbor_mean_price existence
        # or reconstruct from spatial proximity
        
        # Simple approach: k-NN based on coordinates
        coords = time_snapshot[['x', 'y', 'z']].values
        node_ids = time_snapshot['node_id'].values
        
        from sklearn.neighbors import NearestNeighbors
        
        # Get average degree from dataset
        k = int(time_snapshot['degree'].mean()) if 'degree' in time_snapshot.columns else 3
        k = max(1, min(k, len(coords) - 1))
        
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self
                src = self.node_mapping[node_ids[i]]
                dst = self.node_mapping[node_ids[j]]
                edges.append([src, dst])
        
        if len(edges) == 0:
            # Fallback: fully connected small graph
            n = len(node_ids)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edges.append([i, j])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Normalize features using StandardScaler.
        
        Args:
            df: DataFrame with features
            fit: Whether to fit scaler (True for train, False for val/test)
        
        Returns:
            Normalized feature array
        """
        features = df[self.feature_cols].values
        
        if fit:
            self.scalers['features'] = StandardScaler()
            normalized = self.scalers['features'].fit_transform(features)
        else:
            if 'features' not in self.scalers:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            normalized = self.scalers['features'].transform(features)
        
        return normalized
    
    def create_temporal_snapshot(
        self, 
        current_time: int,
        fit_scaler: bool = False
    ) -> Optional[Data]:
        """
        Create graph snapshot for a given time with lookback.
        
        Args:
            current_time: Current timestep
            fit_scaler: Whether to fit normalizer
        
        Returns:
            PyG Data object or None if insufficient history
        """
        # Need lookback + horizon timesteps
        start_time = current_time - self.lookback + 1
        end_time = current_time + self.horizon
        
        if start_time < self.df['time'].min() or end_time > self.df['time'].max():
            return None
        
        # Get current snapshot
        current_data = self.df[self.df['time'] == current_time].copy()
        
        if len(current_data) == 0:
            return None
        
        # Sort by node_id for consistency
        current_data = current_data.sort_values('node_id')
        
        # Build edge index
        edge_index = self.build_edge_index(current_data)
        
        # Collect features from lookback window
        node_features_list = []
        
        for t in range(start_time, current_time + 1):
            t_data = self.df[self.df['time'] == t].sort_values('node_id')
            
            if len(t_data) != len(current_data):
                return None  # Inconsistent node count
            
            # Normalize features
            features = self.normalize_features(t_data, fit=fit_scaler)
            node_features_list.append(features)
        
        # Stack temporal features: [num_nodes, lookback, num_features]
        node_features = np.stack(node_features_list, axis=1)
        
        # Flatten to [num_nodes, lookback * num_features]
        num_nodes, lookback, num_features = node_features.shape
        node_features_flat = node_features.reshape(num_nodes, lookback * num_features)
        
        # Get target (price at t + horizon)
        target_time = current_time + self.horizon
        target_data = self.df[self.df['time'] == target_time].sort_values('node_id')
        
        if len(target_data) != len(current_data):
            return None
        
        # Target: price_state (normalized deviation)
        target = target_data['price_state'].values
        
        # Create PyG Data object
        data = Data(
            x=torch.FloatTensor(node_features_flat),
            edge_index=edge_index,
            y=torch.FloatTensor(target),
            time=current_time,
            num_nodes=num_nodes
        )
        
        # Add metadata
        data.node_ids = current_data['node_id'].values
        data.actual_prices = current_data['actual_price'].values
        
        return data
    
    def build_dataset(
        self,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Build complete dataset with train/val/test splits.
        
        Args:
            train_split: Fraction for training
            val_split: Fraction for validation
            test_split: Fraction for testing
        
        Returns:
            (train_graphs, val_graphs, test_graphs)
        """
        print("\nBuilding CGNN dataset...")
        print(f"  Lookback: {self.lookback} timesteps")
        print(f"  Horizon: {self.horizon} timestep(s)")
        print(f"  Features: {self.feature_cols}")
        
        # Get valid time range
        min_time = self.df['time'].min() + self.lookback - 1
        max_time = self.df['time'].max() - self.horizon
        
        valid_times = sorted([
            t for t in self.df['time'].unique()
            if min_time <= t <= max_time
        ])
        
        print(f"  Valid timesteps: {len(valid_times)}")
        
        # Split times
        n_times = len(valid_times)
        n_train = int(n_times * train_split)
        n_val = int(n_times * val_split)
        
        train_times = valid_times[:n_train]
        val_times = valid_times[n_train:n_train + n_val]
        test_times = valid_times[n_train + n_val:]
        
        print(f"  Train: {len(train_times)} timesteps")
        print(f"  Val: {len(val_times)} timesteps")
        print(f"  Test: {len(test_times)} timesteps")
        
        # Build training graphs (fit scaler)
        print("\nBuilding training graphs...")
        train_graphs = []
        for i, t in enumerate(train_times):
            data = self.create_temporal_snapshot(t, fit_scaler=(i == 0))
            if data is not None:
                train_graphs.append(data)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(train_times)}")
        
        print(f"✓ Created {len(train_graphs)} training graphs")
        
        # Build validation graphs
        print("\nBuilding validation graphs...")
        val_graphs = []
        for i, t in enumerate(val_times):
            data = self.create_temporal_snapshot(t, fit_scaler=False)
            if data is not None:
                val_graphs.append(data)
        
        print(f"✓ Created {len(val_graphs)} validation graphs")
        
        # Build test graphs
        print("\nBuilding test graphs...")
        test_graphs = []
        for i, t in enumerate(test_times):
            data = self.create_temporal_snapshot(t, fit_scaler=False)
            if data is not None:
                test_graphs.append(data)
        
        print(f"✓ Created {len(test_graphs)} test graphs")
        
        return train_graphs, val_graphs, test_graphs
    
    def save_dataset(
        self,
        train_graphs: List[Data],
        val_graphs: List[Data],
        test_graphs: List[Data],
        output_dir: str = "cgnn_data"
    ):
        """Save dataset to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save graphs
        torch.save(train_graphs, output_path / "train.pt")
        torch.save(val_graphs, output_path / "val.pt")
        torch.save(test_graphs, output_path / "test.pt")
        
        # Save scalers and metadata
        metadata = {
            'lookback': self.lookback,
            'horizon': self.horizon,
            'feature_cols': self.feature_cols,
            'target_node': self.target_node,
            'node_mapping': self.node_mapping,
            'reverse_mapping': self.reverse_mapping,
            'scalers': self.scalers,
            'num_features': len(self.feature_cols)
        }
        
        with open(output_path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n✓ Dataset saved to {output_dir}/")
        print(f"  train.pt: {len(train_graphs)} graphs")
        print(f"  val.pt: {len(val_graphs)} graphs")
        print(f"  test.pt: {len(test_graphs)} graphs")
        print(f"  metadata.pkl: Configuration and scalers")
    
    def build_and_save(
        self,
        output_dir: str = "cgnn_data",
        train_split: float = 0.7,
        val_split: float = 0.15
    ):
        """Complete pipeline: load, build, save"""
        self.load_data()
        train, val, test = self.build_dataset(train_split, val_split)
        self.save_dataset(train, val, test, output_dir)
        
        print("\n" + "="*60)
        print("DATASET BUILD COMPLETE")
        print("="*60)
        
        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Build CGNN dataset from mandi CSV'
    )
    parser.add_argument('csv_file', type=str, help='Input CSV file')
    parser.add_argument('--lookback', type=int, default=10,
                       help='Number of past timesteps (k)')
    parser.add_argument('--horizon', type=int, default=1,
                       help='Prediction horizon')
    parser.add_argument('--output', type=str, default='cgnn_data',
                       help='Output directory')
    parser.add_argument('--target-node', type=int, default=None,
                       help='Specific mandi to predict (None = all)')
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Training data fraction')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation data fraction')
    
    args = parser.parse_args()
    
    builder = CGNNDatasetBuilder(
        csv_path=args.csv_file,
        lookback=args.lookback,
        horizon=args.horizon,
        target_node=args.target_node
    )
    
    builder.build_and_save(
        output_dir=args.output,
        train_split=args.train_split,
        val_split=args.val_split
    )


if __name__ == "__main__":
    main()