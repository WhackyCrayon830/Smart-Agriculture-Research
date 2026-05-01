"""
Configurable Conditional GNN Trainer for Mandi Price Prediction

Interactive training script that:
- Asks user for configuration (k, target mandi, etc.)
- Trains temporal GNN model
- Saves best model
- Generates training plots
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.loader import DataLoader
import pickle
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict


class TemporalGNN(torch.nn.Module):
    """
    Temporal Graph Neural Network for price prediction.
    
    Architecture:
    - GNN layers process spatial structure
    - Temporal features encoded in node features
    - MLP head for final prediction
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        gnn_type: str = 'GCN'
    ):
        """
        Initialize model.
        
        Args:
            in_channels: Input feature dimension (lookback * num_features)
            hidden_channels: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            gnn_type: Type of GNN ('GCN', 'GAT', 'GraphSAGE')
        """
        super().__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Select GNN layer type
        if gnn_type == 'GCN':
            GNNLayer = GCNConv
        elif gnn_type == 'GAT':
            GNNLayer = lambda i, o: GATConv(i, o, heads=4, concat=False)
        elif gnn_type == 'GraphSAGE':
            GNNLayer = SAGEConv
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Input projection
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
        
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GNNLayer(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Output MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, data):
        """Forward pass"""
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection (if same dim)
            if i > 0:
                x = x + x_new
            else:
                x = x_new
        
        # Output prediction
        out = self.mlp(x)
        return out.squeeze(-1)


class CGNNTrainer:
    """Trainer for Conditional GNN models"""
    
    def __init__(
        self,
        data_dir: str = "cgnn_data",
        model_dir: str = "models",
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            data_dir: Directory with CGNN dataset
            model_dir: Directory to save models
            device: Device to use (cuda/cpu)
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load metadata
        with open(self.data_dir / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"\nDataset metadata:")
        print(f"  Lookback: {self.metadata['lookback']} timesteps")
        print(f"  Horizon: {self.metadata['horizon']} timestep(s)")
        print(f"  Features: {self.metadata['num_features']}")
        print(f"  Nodes: {len(self.metadata['node_mapping'])}")
        
        # Load data
        print(f"\nLoading data from {data_dir}...")
        self.train_data = torch.load(self.data_dir / "train.pt")
        self.val_data = torch.load(self.data_dir / "val.pt")
        self.test_data = torch.load(self.data_dir / "test.pt")
        
        print(f"  Train: {len(self.train_data)} graphs")
        print(f"  Val: {len(self.val_data)} graphs")
        print(f"  Test: {len(self.test_data)} graphs")
        
        self.model = None
        self.config = {}
        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    
    def get_user_config(self):
        """Interactive configuration"""
        print("\n" + "="*60)
        print("CGNN TRAINING CONFIGURATION")
        print("="*60)
        
        # Show available mandis
        node_ids = sorted(self.metadata['reverse_mapping'].values())
        print(f"\nAvailable mandis: {node_ids}")
        
        # Get target mandi
        while True:
            target_input = input(f"\nEnter target mandi ID to predict (or 'all' for all mandis): ").strip()
            if target_input.lower() == 'all':
                self.config['target_mandi'] = None
                print("→ Training for ALL mandis")
                break
            else:
                try:
                    target_id = int(target_input)
                    if target_id in node_ids:
                        self.config['target_mandi'] = target_id
                        print(f"→ Training for mandi {target_id}")
                        break
                    else:
                        print(f"Error: Mandi {target_id} not in dataset. Choose from {node_ids}")
                except ValueError:
                    print("Error: Please enter a number or 'all'")
        
        # Model architecture
        print("\n" + "-"*60)
        print("MODEL ARCHITECTURE")
        print("-"*60)
        
        gnn_types = ['GCN', 'GAT', 'GraphSAGE']
        print(f"GNN types: {gnn_types}")
        gnn_choice = input("Select GNN type (default: GCN): ").strip() or 'GCN'
        self.config['gnn_type'] = gnn_choice if gnn_choice in gnn_types else 'GCN'
        
        hidden = input("Hidden dimension (default: 64): ").strip()
        self.config['hidden_dim'] = int(hidden) if hidden else 64
        
        layers = input("Number of GNN layers (default: 3): ").strip()
        self.config['num_layers'] = int(layers) if layers else 3
        
        dropout = input("Dropout rate (default: 0.2): ").strip()
        self.config['dropout'] = float(dropout) if dropout else 0.2
        
        # Training parameters
        print("\n" + "-"*60)
        print("TRAINING PARAMETERS")
        print("-"*60)
        
        epochs = input("Number of epochs (default: 100): ").strip()
        self.config['epochs'] = int(epochs) if epochs else 100
        
        batch_size = input("Batch size (default: 32): ").strip()
        self.config['batch_size'] = int(batch_size) if batch_size else 32
        
        lr = input("Learning rate (default: 0.001): ").strip()
        self.config['learning_rate'] = float(lr) if lr else 0.001
        
        # Print summary
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print("="*60)
        
        confirm = input("\nProceed with training? (y/n): ").strip().lower()
        return confirm == 'y'
    
    def build_model(self):
        """Build model from config"""
        in_channels = self.metadata['lookback'] * self.metadata['num_features']
        
        self.model = TemporalGNN(
            in_channels=in_channels,
            hidden_channels=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            gnn_type=self.config['gnn_type']
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nModel created: {total_params:,} parameters")
    
    def train_epoch(self, loader, optimizer):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for data in loader:
            data = data.to(self.device)
            
            optimizer.zero_grad()
            pred = self.model(data)
            
            # Loss (MSE on price_state)
            if self.config['target_mandi'] is not None:
                # Single mandi prediction
                target_idx = self.metadata['node_mapping'][self.config['target_mandi']]
                loss = F.mse_loss(pred[target_idx], data.y[target_idx])
            else:
                # All mandis
                loss = F.mse_loss(pred, data.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def evaluate(self, loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                pred = self.model(data)
                
                if self.config['target_mandi'] is not None:
                    target_idx = self.metadata['node_mapping'][self.config['target_mandi']]
                    loss = F.mse_loss(pred[target_idx], data.y[target_idx])
                    mae = F.l1_loss(pred[target_idx], data.y[target_idx])
                else:
                    loss = F.mse_loss(pred, data.y)
                    mae = F.l1_loss(pred, data.y)
                
                total_loss += loss.item()
                total_mae += mae.item()
        
        return total_loss / len(loader), total_mae / len(loader)
    
    def train(self):
        """Complete training loop"""
        # Create data loaders
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            self.val_data,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60 + "\n")
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer)
            
            # Validate
            val_loss, val_mae = self.evaluate(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{self.config['epochs']} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Val MAE: {val_mae:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pt')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        # Load best model
        self.load_model('best_model.pt')
        
        # Final test evaluation
        test_loader = DataLoader(self.test_data, batch_size=self.config['batch_size'])
        test_loss, test_mae = self.evaluate(test_loader)
        
        print(f"\nTest set performance:")
        print(f"  Loss (MSE): {test_loss:.6f}")
        print(f"  MAE: {test_mae:.6f}")
        
        self.plot_training()
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'config': self.config,
            'metadata': self.metadata,
            'history': self.history
        }
        torch.save(checkpoint, self.model_dir / filename)
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(self.model_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.config = checkpoint['config']
        self.history = checkpoint['history']
    
    def plot_training(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # MAE curve
        axes[1].plot(epochs, self.history['val_mae'], label='Val MAE', 
                    linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('Validation MAE')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Training curves saved: {self.model_dir / 'training_curves.png'}")
        plt.close()
    
    def run(self):
        """Complete training pipeline"""
        if not self.get_user_config():
            print("Training cancelled.")
            return
        
        self.build_model()
        self.train()
        
        print(f"\n✓ Model saved to: {self.model_dir / 'best_model.pt'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CGNN for mandi price prediction')
    parser.add_argument('--data-dir', type=str, default='cgnn_data',
                       help='CGNN dataset directory')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    trainer = CGNNTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        device=args.device
    )
    
    trainer.run()


if __name__ == "__main__":
    main()