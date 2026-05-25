"""
CGNN Model Evaluator

Comprehensive evaluation of trained CGNN models:
- Multiple evaluation metrics
- Prediction vs actual plots over time
- Node-specific analysis
- Error distribution
- Interactive configuration
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class CGNNEvaluator:
    """Comprehensive evaluator for trained CGNN models"""
    
    def __init__(
        self,
        model_path: str = "models/best_model.pt",
        data_dir: str = "cgnn_data",
        output_dir: str = "evaluation"
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            data_dir: CGNN dataset directory
            output_dir: Output directory for plots
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.metadata = checkpoint['metadata']
        
        print(f"✓ Model loaded")
        print(f"  Target mandi: {self.config.get('target_mandi', 'ALL')}")
        print(f"  Lookback: {self.metadata['lookback']} timesteps")
        print(f"  GNN type: {self.config['gnn_type']}")
        
        # Rebuild model
        from train_cgnn import TemporalGNN
        
        in_channels = self.metadata['lookback'] * self.metadata['num_features']
        self.model = TemporalGNN(
            in_channels=in_channels,
            hidden_channels=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            gnn_type=self.config['gnn_type']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        # Load test data
        print(f"\nLoading test data...")
        self.test_data = torch.load(self.data_dir / "test.pt")
        print(f"✓ Loaded {len(self.test_data)} test graphs")
        
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        self.metrics = {}
    
    def get_user_config(self) -> Dict:
        """Interactive evaluation configuration"""
        print("\n" + "="*60)
        print("EVALUATION CONFIGURATION")
        print("="*60)
        
        eval_config = {}
        
        # Number of timesteps to plot
        n_steps = input(f"\nNumber of past timesteps to plot (default: {len(self.test_data)}): ").strip()
        eval_config['n_steps'] = int(n_steps) if n_steps else len(self.test_data)
        eval_config['n_steps'] = min(eval_config['n_steps'], len(self.test_data))
        
        # Which mandi to analyze (if trained on all)
        if self.config.get('target_mandi') is None:
            node_ids = sorted(self.metadata['reverse_mapping'].values())
            print(f"\nAvailable mandis: {node_ids}")
            
            mandi_input = input("Which mandi to analyze in detail? (default: first mandi): ").strip()
            if mandi_input:
                eval_config['focus_mandi'] = int(mandi_input)
            else:
                eval_config['focus_mandi'] = node_ids[0]
        else:
            eval_config['focus_mandi'] = self.config['target_mandi']
        
        print(f"\n→ Will plot last {eval_config['n_steps']} timesteps")
        print(f"→ Focus mandi: {eval_config['focus_mandi']}")
        
        return eval_config
    
    def run_predictions(self):
        """Run model on test set and collect predictions"""
        print("\n" + "="*60)
        print("RUNNING PREDICTIONS")
        print("="*60)
        
        loader = DataLoader(self.test_data, batch_size=1, shuffle=False)
        
        all_preds = []
        all_actuals = []
        all_times = []
        
        with torch.no_grad():
            for i, data in enumerate(loader):
                data = data.to(self.device)
                pred = self.model(data)
                
                all_preds.append(pred.cpu().numpy())
                all_actuals.append(data.y.cpu().numpy())
                all_times.append(data.time)
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i+1}/{len(loader)} graphs")
        
        self.predictions = all_preds
        self.actuals = all_actuals
        self.timestamps = all_times
        
        print(f"✓ Generated {len(self.predictions)} predictions")
    
    def calculate_metrics(self, eval_config: Dict) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        print("\n" + "="*60)
        print("CALCULATING METRICS")
        print("="*60)
        
        metrics = {}
        
        # Get focus mandi index
        if self.config.get('target_mandi') is not None:
            # Single mandi model
            focus_idx = self.metadata['node_mapping'][self.config['target_mandi']]
            
            preds = np.array([p[focus_idx] for p in self.predictions])
            actuals = np.array([a[focus_idx] for a in self.actuals])
        else:
            # Multi-mandi model, extract focus mandi
            focus_idx = self.metadata['node_mapping'][eval_config['focus_mandi']]
            
            preds = np.array([p[focus_idx] for p in self.predictions])
            actuals = np.array([a[focus_idx] for a in self.actuals])
        
        # Calculate metrics
        mse = np.mean((preds - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - actuals))
        
        # R² score
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - actuals.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # MAPE (if no zeros in actuals)
        if (actuals != 0).all():
            mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
        else:
            mape = None
        
        # Direction accuracy (did we predict the right direction?)
        actual_changes = np.diff(actuals)
        pred_changes = np.diff(preds)
        direction_correct = (np.sign(actual_changes) == np.sign(pred_changes)).sum()
        direction_accuracy = direction_correct / len(actual_changes) if len(actual_changes) > 0 else 0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'num_predictions': len(preds)
        }
        
        # Print metrics
        print(f"\nMETRICS (Mandi {eval_config['focus_mandi']}):")
        print(f"  MSE:  {metrics['mse']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"  R²:   {metrics['r2']:.6f}")
        if mape is not None:
            print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
        
        self.metrics = metrics
        return metrics
    
    def plot_predictions_vs_actual(self, eval_config: Dict):
        """Plot predicted vs actual prices over time"""
        print("\n" + "="*60)
        print("GENERATING PLOTS")
        print("="*60)
        
        # Get focus mandi data
        if self.config.get('target_mandi') is not None:
            focus_idx = self.metadata['node_mapping'][self.config['target_mandi']]
        else:
            focus_idx = self.metadata['node_mapping'][eval_config['focus_mandi']]
        
        preds = np.array([p[focus_idx] for p in self.predictions]).squeeze()
        actuals = np.array([a[focus_idx] for a in self.actuals]).squeeze()
        times = np.array(self.timestamps).squeeze()
        
        # Limit to n_steps
        n_steps = eval_config['n_steps']
        if len(preds) > n_steps:
            preds = preds[-n_steps:]
            actuals = actuals[-n_steps:]
            times = times[-n_steps:]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Time series plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(times, actuals, label='Actual', linewidth=2, alpha=0.8)
        ax1.plot(times, preds, label='Predicted', linewidth=2, alpha=0.8)
        ax1.fill_between(times, actuals, preds, alpha=0.2)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Price State (Normalized)', fontsize=12)
        ax1.set_title(f'Predicted vs Actual Price (Mandi {eval_config["focus_mandi"]})', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        
        # 2. Scatter plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(actuals, preds, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        min_val = min(actuals.min(), preds.min())
        max_val = max(actuals.max(), preds.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect prediction')
        ax2.set_xlabel('Actual', fontsize=12)
        ax2.set_ylabel('Predicted', fontsize=12)
        ax2.set_title(f'Prediction Accuracy (R² = {self.metrics["r2"]:.4f})', 
                     fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Residuals over time
        ax3 = fig.add_subplot(gs[1, 1])
        residuals = actuals - preds
        ax3.plot(times, residuals, linewidth=1.5, alpha=0.7)
        ax3.axhline(0, color='red', linestyle='--', linewidth=2)
        ax3.fill_between(times, 0, residuals, alpha=0.3)
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
        ax3.set_title(f'Residuals (MAE = {self.metrics["mae"]:.4f})', 
                     fontsize=13, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # 4. Error distribution
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax4.axvline(residuals.mean(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean: {residuals.mean():.4f}')
        ax4.set_xlabel('Residual', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Error Distribution', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Percentage error over time
        ax5 = fig.add_subplot(gs[2, 1])
        pct_error = np.abs((actuals - preds) / (actuals + 1e-8)) * 100
        ax5.plot(times, pct_error, linewidth=1.5, alpha=0.7, color='orange')
        ax5.axhline(pct_error.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {pct_error.mean():.2f}%')
        ax5.set_xlabel('Time', fontsize=12)
        ax5.set_ylabel('Absolute Percentage Error (%)', fontsize=12)
        ax5.set_title('Percentage Error Over Time', fontsize=13, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / f'predictions_mandi_{eval_config["focus_mandi"]}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_multi_mandi_comparison(self, eval_config: Dict):
        """Compare predictions across multiple mandis"""
        if self.config.get('target_mandi') is not None:
            print("  Skipping multi-mandi plot (single mandi model)")
            return
        
        # Select 4 representative mandis
        node_ids = sorted(self.metadata['reverse_mapping'].values())
        sample_mandis = node_ids[:4] if len(node_ids) >= 4 else node_ids
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, mandi_id in enumerate(sample_mandis):
            node_idx = self.metadata['node_mapping'][mandi_id]
            
            preds = np.array([p[node_idx] for p in self.predictions])
            actuals = np.array([a[node_idx] for a in self.actuals])
            times = np.array(self.timestamps)
            
            # Limit to n_steps
            n_steps = min(eval_config['n_steps'], len(preds))
            preds = preds[-n_steps:]
            actuals = actuals[-n_steps:]
            times = times[-n_steps:]
            
            # Calculate R²
            ss_res = np.sum((actuals - preds) ** 2)
            ss_tot = np.sum((actuals - actuals.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            axes[idx].plot(times, actuals, label='Actual', linewidth=2, alpha=0.8)
            axes[idx].plot(times, preds, label='Predicted', linewidth=2, alpha=0.8)
            axes[idx].set_xlabel('Time')
            axes[idx].set_ylabel('Price State')
            axes[idx].set_title(f'Mandi {mandi_id} (R² = {r2:.4f})', fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'multi_mandi_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def generate_report(self, eval_config: Dict):
        """Generate evaluation report"""
        report_file = self.output_dir / 'evaluation_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CGNN MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Target Mandi: {self.config.get('target_mandi', 'ALL')}\n")
            f.write(f"Focus Mandi: {eval_config['focus_mandi']}\n\n")
            
            f.write("MODEL CONFIGURATION\n")
            f.write("-"*80 + "\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("EVALUATION METRICS\n")
            f.write("-"*80 + "\n")
            for key, value in self.metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.6f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("DATASET INFO\n")
            f.write("-"*80 + "\n")
            f.write(f"Lookback: {self.metadata['lookback']} timesteps\n")
            f.write(f"Horizon: {self.metadata['horizon']} timestep(s)\n")
            f.write(f"Features: {self.metadata['num_features']}\n")
            f.write(f"Test samples: {len(self.test_data)}\n")
        
        print(f"✓ Saved: {report_file}")
    
    def evaluate(self):
        """Complete evaluation pipeline"""
        eval_config = self.get_user_config()
        
        self.run_predictions()
        self.calculate_metrics(eval_config)
        self.plot_predictions_vs_actual(eval_config)
        self.plot_multi_mandi_comparison(eval_config)
        self.generate_report(eval_config)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"\nResults saved to: {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained CGNN model')
    parser.add_argument('--model', type=str, default='models/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--data-dir', type=str, default='cgnn_data',
                       help='CGNN dataset directory')
    parser.add_argument('--output', type=str, default='evaluation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    evaluator = CGNNEvaluator(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output
    )
    
    evaluator.evaluate()


if __name__ == "__main__":
    main()