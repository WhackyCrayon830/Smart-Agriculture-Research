"""
CSV Dataset Evaluator for Mandi Network Data

Analyzes the quality and characteristics of generated mandi datasets:
- Data completeness and integrity
- Statistical distributions
- Temporal patterns
- Spatial correlations
- Blockchain consensus quality
- Graph structure properties
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MandiDatasetEvaluator:
    """Comprehensive evaluator for mandi network datasets"""
    
    def __init__(self, csv_path: str):
        """
        Initialize evaluator with dataset.
        
        Args:
            csv_path: Path to CSV file
        """
        self.csv_path = csv_path
        self.df = None
        self.stats = {}
        
    def load_data(self) -> bool:
        """Load and validate CSV file"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✓ Loaded dataset: {len(self.df):,} rows")
            return True
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return False
    
    def check_schema(self) -> Dict:
        """Validate dataset schema and columns"""
        required_cols = [
            'world_id', 'time', 'node_id', 'x', 'y', 'z',
            'price_state', 'actual_price', 'weather_effect',
            'supply', 'demand', 'storage'
        ]
        
        optional_cols = [
            'neighbor_mean_price', 'degree', 'volatility', 'price_trend',
            'blockchain_price', 'price_consensus', 'on_chain'
        ]
        
        present_required = [col for col in required_cols if col in self.df.columns]
        missing_required = [col for col in required_cols if col not in self.df.columns]
        present_optional = [col for col in optional_cols if col in self.df.columns]
        
        schema_info = {
            'total_columns': len(self.df.columns),
            'required_present': len(present_required),
            'required_missing': len(missing_required),
            'optional_present': len(present_optional),
            'has_blockchain': 'blockchain_price' in self.df.columns,
            'missing_cols': missing_required
        }
        
        print("\n" + "="*60)
        print("SCHEMA VALIDATION")
        print("="*60)
        print(f"Total columns: {schema_info['total_columns']}")
        print(f"Required columns: {schema_info['required_present']}/{len(required_cols)}")
        print(f"Optional columns: {schema_info['optional_present']}/{len(optional_cols)}")
        print(f"Blockchain enabled: {schema_info['has_blockchain']}")
        
        if missing_required:
            print(f"⚠ Missing required: {missing_required}")
        else:
            print("✓ All required columns present")
        
        self.stats['schema'] = schema_info
        return schema_info
    
    def analyze_completeness(self) -> Dict:
        """Check for missing values and data quality"""
        completeness = {}
        
        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            null_pct = 100 * null_count / len(self.df)
            completeness[col] = {
                'null_count': null_count,
                'null_pct': null_pct,
                'complete': null_count == 0
            }
        
        print("\n" + "="*60)
        print("DATA COMPLETENESS")
        print("="*60)
        
        incomplete_cols = [col for col, info in completeness.items() 
                          if not info['complete']]
        
        if incomplete_cols:
            print("⚠ Columns with missing values:")
            for col in incomplete_cols:
                print(f"  {col}: {completeness[col]['null_count']:,} "
                      f"({completeness[col]['null_pct']:.2f}%)")
        else:
            print("✓ No missing values detected")
        
        self.stats['completeness'] = completeness
        return completeness
    
    def analyze_temporal(self) -> Dict:
        """Analyze temporal structure"""
        temporal = {}
        
        # Time range
        temporal['min_time'] = self.df['time'].min()
        temporal['max_time'] = self.df['time'].max()
        temporal['total_timesteps'] = self.df['time'].nunique()
        temporal['time_range'] = temporal['max_time'] - temporal['min_time'] + 1
        
        # Check continuity
        expected_steps = temporal['time_range']
        actual_steps = temporal['total_timesteps']
        temporal['continuous'] = expected_steps == actual_steps
        
        # Nodes per timestep
        nodes_per_time = self.df.groupby('time')['node_id'].count()
        temporal['avg_nodes_per_time'] = nodes_per_time.mean()
        temporal['min_nodes_per_time'] = nodes_per_time.min()
        temporal['max_nodes_per_time'] = nodes_per_time.max()
        temporal['consistent_nodes'] = nodes_per_time.std() < 0.1
        
        print("\n" + "="*60)
        print("TEMPORAL STRUCTURE")
        print("="*60)
        print(f"Time range: {temporal['min_time']} → {temporal['max_time']}")
        print(f"Total timesteps: {temporal['total_timesteps']:,}")
        print(f"Continuous: {temporal['continuous']}")
        print(f"Nodes per timestep: {temporal['avg_nodes_per_time']:.1f} "
              f"(±{nodes_per_time.std():.2f})")
        
        if not temporal['continuous']:
            print("⚠ Warning: Temporal gaps detected")
        
        self.stats['temporal'] = temporal
        return temporal
    
    def analyze_spatial(self) -> Dict:
        """Analyze spatial distribution"""
        spatial = {}
        
        # Unique nodes
        spatial['total_nodes'] = self.df['node_id'].nunique()
        
        # Coordinate ranges
        spatial['x_range'] = (self.df['x'].min(), self.df['x'].max())
        spatial['y_range'] = (self.df['y'].min(), self.df['y'].max())
        spatial['z_range'] = (self.df['z'].min(), self.df['z'].max())
        
        # Spatial spread
        spatial['x_std'] = self.df['x'].std()
        spatial['y_std'] = self.df['y'].std()
        spatial['z_std'] = self.df['z'].std()
        
        # Graph connectivity
        if 'degree' in self.df.columns:
            spatial['avg_degree'] = self.df['degree'].mean()
            spatial['min_degree'] = self.df['degree'].min()
            spatial['max_degree'] = self.df['degree'].max()
        
        print("\n" + "="*60)
        print("SPATIAL STRUCTURE")
        print("="*60)
        print(f"Total unique nodes: {spatial['total_nodes']}")
        print(f"X range: {spatial['x_range'][0]:.1f} → {spatial['x_range'][1]:.1f}")
        print(f"Y range: {spatial['y_range'][0]:.1f} → {spatial['y_range'][1]:.1f}")
        print(f"Z range: {spatial['z_range'][0]:.1f} → {spatial['z_range'][1]:.1f}")
        
        if 'degree' in self.df.columns:
            print(f"Avg connectivity: {spatial['avg_degree']:.2f} neighbors")
        
        self.stats['spatial'] = spatial
        return spatial
    
    def analyze_prices(self) -> Dict:
        """Analyze price distributions and dynamics"""
        price_stats = {}
        
        # Price statistics
        price_stats['actual_price_mean'] = self.df['actual_price'].mean()
        price_stats['actual_price_std'] = self.df['actual_price'].std()
        price_stats['actual_price_min'] = self.df['actual_price'].min()
        price_stats['actual_price_max'] = self.df['actual_price'].max()
        
        # Price state (normalized)
        price_stats['state_mean'] = self.df['price_state'].mean()
        price_stats['state_std'] = self.df['price_state'].std()
        
        # Price volatility
        if 'volatility' in self.df.columns:
            price_stats['avg_volatility'] = self.df['volatility'].mean()
        
        # Price trends
        if 'price_trend' in self.df.columns:
            price_stats['avg_trend'] = self.df['price_trend'].mean()
            price_stats['trend_std'] = self.df['price_trend'].std()
        
        # Blockchain consensus
        if 'blockchain_price' in self.df.columns:
            price_stats['blockchain_price_mean'] = self.df['blockchain_price'].mean()
            price_stats['consensus_rate'] = self.df['price_consensus'].mean()
            
            # Price deviation from consensus
            on_chain_rows = self.df[self.df['on_chain'] == 1]
            if len(on_chain_rows) > 0:
                deviation = abs(on_chain_rows['actual_price'] - 
                              on_chain_rows['blockchain_price'])
                price_stats['avg_deviation_from_consensus'] = deviation.mean()
        
        print("\n" + "="*60)
        print("PRICE ANALYSIS")
        print("="*60)
        print(f"Price range: ₹{price_stats['actual_price_min']:.2f} → "
              f"₹{price_stats['actual_price_max']:.2f}")
        print(f"Mean price: ₹{price_stats['actual_price_mean']:.2f} "
              f"(±₹{price_stats['actual_price_std']:.2f})")
        print(f"State range: {self.df['price_state'].min():.3f} → "
              f"{self.df['price_state'].max():.3f}")
        
        if 'blockchain_price' in self.df.columns:
            print(f"Consensus rate: {price_stats['consensus_rate']:.1%}")
            if 'avg_deviation_from_consensus' in price_stats:
                print(f"Avg deviation: ₹{price_stats['avg_deviation_from_consensus']:.2f}")
        
        self.stats['prices'] = price_stats
        return price_stats
    
    def analyze_market_dynamics(self) -> Dict:
        """Analyze supply, demand, and market conditions"""
        market = {}
        
        # Supply/Demand
        market['avg_supply'] = self.df['supply'].mean()
        market['avg_demand'] = self.df['demand'].mean()
        market['supply_std'] = self.df['supply'].std()
        market['demand_std'] = self.df['demand'].std()
        
        # Market balance
        self.df['market_balance'] = self.df['demand'] - self.df['supply']
        market['avg_balance'] = self.df['market_balance'].mean()
        
        # Storage
        market['avg_storage'] = self.df['storage'].mean()
        market['storage_std'] = self.df['storage'].std()
        
        # Weather
        market['avg_weather'] = self.df['weather_effect'].mean()
        market['weather_std'] = self.df['weather_effect'].std()
        market['extreme_weather_pct'] = 100 * (abs(self.df['weather_effect']) > 1.0).sum() / len(self.df)
        
        print("\n" + "="*60)
        print("MARKET DYNAMICS")
        print("="*60)
        print(f"Supply: {market['avg_supply']:.3f} (±{market['supply_std']:.3f})")
        print(f"Demand: {market['avg_demand']:.3f} (±{market['demand_std']:.3f})")
        print(f"Market balance: {market['avg_balance']:+.3f}")
        print(f"Storage: {market['avg_storage']:.1%} (±{market['storage_std']:.1%})")
        print(f"Weather: {market['avg_weather']:+.3f} (±{market['weather_std']:.3f})")
        print(f"Extreme weather: {market['extreme_weather_pct']:.1f}% of timesteps")
        
        self.stats['market'] = market
        return market
    
    def check_stationarity(self) -> Dict:
        """Check if time series are stationary"""
        stationarity = {}
        
        # Group by node and check price stationarity
        for node_id in self.df['node_id'].unique()[:5]:  # Sample 5 nodes
            node_data = self.df[self.df['node_id'] == node_id].sort_values('time')
            prices = node_data['actual_price'].values
            
            # Simple trend check
            time_vals = np.arange(len(prices))
            if len(prices) > 10:
                trend = np.polyfit(time_vals, prices, 1)[0]
                stationarity[f'node_{node_id}_trend'] = trend
        
        avg_trend = np.mean(list(stationarity.values())) if stationarity else 0
        stationarity['avg_trend'] = avg_trend
        stationarity['appears_stationary'] = abs(avg_trend) < 0.01
        
        print("\n" + "="*60)
        print("STATIONARITY CHECK")
        print("="*60)
        print(f"Average price trend: {avg_trend:+.4f}")
        print(f"Appears stationary: {stationarity['appears_stationary']}")
        
        self.stats['stationarity'] = stationarity
        return stationarity
    
    def generate_visualizations(self, output_dir: str = "evaluation_plots"):
        """Generate diagnostic plots"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Price distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].hist(self.df['actual_price'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Price (₹/quintal)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].axvline(self.df['actual_price'].mean(), color='r', 
                          linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # 2. Supply vs Demand
        axes[0, 1].scatter(self.df['supply'], self.df['demand'], 
                          alpha=0.1, s=1)
        axes[0, 1].plot([0, 2], [0, 2], 'r--', label='Equilibrium')
        axes[0, 1].set_xlabel('Supply')
        axes[0, 1].set_ylabel('Demand')
        axes[0, 1].set_title('Supply vs Demand')
        axes[0, 1].legend()
        
        # 3. Weather effect distribution
        axes[1, 0].hist(self.df['weather_effect'], bins=50, 
                       edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Weather Effect')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Weather Distribution')
        
        # 4. Price over time (sample node)
        sample_node = self.df['node_id'].iloc[0]
        node_data = self.df[self.df['node_id'] == sample_node].sort_values('time')
        axes[1, 1].plot(node_data['time'], node_data['actual_price'])
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Price (₹/quintal)')
        axes[1, 1].set_title(f'Price Evolution (Mandi {sample_node})')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overview.png", dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/overview.png")
        plt.close()
        
        # 5. Correlation matrix
        numeric_cols = ['actual_price', 'weather_effect', 'supply', 
                       'demand', 'storage']
        if all(col in self.df.columns for col in numeric_cols):
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = self.df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', 
                       center=0, ax=ax)
            ax.set_title('Feature Correlations')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlations.png", dpi=150, 
                       bbox_inches='tight')
            print(f"✓ Saved: {output_dir}/correlations.png")
            plt.close()
        
        # 6. Blockchain consensus (if available)
        if 'blockchain_price' in self.df.columns:
            on_chain = self.df[self.df['on_chain'] == 1]
            if len(on_chain) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(on_chain['actual_price'], 
                          on_chain['blockchain_price'], 
                          alpha=0.5, s=10)
                min_p = min(on_chain['actual_price'].min(), 
                           on_chain['blockchain_price'].min())
                max_p = max(on_chain['actual_price'].max(), 
                           on_chain['blockchain_price'].max())
                ax.plot([min_p, max_p], [min_p, max_p], 'r--', 
                       label='Perfect consensus')
                ax.set_xlabel('Actual Price (₹/quintal)')
                ax.set_ylabel('Blockchain Price (₹/quintal)')
                ax.set_title('Price vs Blockchain Consensus')
                ax.legend()
                plt.tight_layout()
                plt.savefig(f"{output_dir}/blockchain_consensus.png", 
                           dpi=150, bbox_inches='tight')
                print(f"✓ Saved: {output_dir}/blockchain_consensus.png")
                plt.close()
    
    def generate_report(self, output_file: str = "dataset_report.txt"):
        """Generate comprehensive text report"""
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MANDI DATASET EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset: {self.csv_path}\n")
            f.write(f"Total rows: {len(self.df):,}\n\n")
            
            for section, data in self.stats.items():
                f.write(f"\n{section.upper()}\n")
                f.write("-" * 80 + "\n")
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
        
        print(f"\n✓ Report saved: {output_file}")
    
    def evaluate(self, generate_plots: bool = True):
        """Run complete evaluation pipeline"""
        if not self.load_data():
            return False
        
        print("\n" + "="*80)
        print("STARTING DATASET EVALUATION")
        print("="*80)
        
        self.check_schema()
        self.analyze_completeness()
        self.analyze_temporal()
        self.analyze_spatial()
        self.analyze_prices()
        self.analyze_market_dynamics()
        self.check_stationarity()
        
        if generate_plots:
            self.generate_visualizations()
        
        self.generate_report()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"\nDataset Quality Score: {self._quality_score():.1f}/100")
        
        return True
    
    def _quality_score(self) -> float:
        """Calculate overall dataset quality score"""
        score = 0.0
        
        # Schema completeness (20 points)
        if self.stats['schema']['required_missing'] == 0:
            score += 20
        
        # Data completeness (20 points)
        complete_cols = sum(1 for info in self.stats['completeness'].values() 
                          if info['complete'])
        total_cols = len(self.stats['completeness'])
        score += 20 * (complete_cols / total_cols)
        
        # Temporal continuity (20 points)
        if self.stats['temporal']['continuous']:
            score += 15
        if self.stats['temporal']['consistent_nodes']:
            score += 5
        
        # Price variability (20 points)
        price_cv = (self.stats['prices']['actual_price_std'] / 
                   self.stats['prices']['actual_price_mean'])
        if 0.05 < price_cv < 0.3:  # Good variability
            score += 20
        elif price_cv > 0:
            score += 10
        
        # Market dynamics (20 points)
        if abs(self.stats['market']['avg_balance']) < 0.2:
            score += 10
        if 0.3 < self.stats['market']['avg_storage'] < 0.7:
            score += 10
        
        return score


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Mandi Network Dataset'
    )
    parser.add_argument('csv_file', type=str, help='Path to CSV dataset')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip plot generation')
    parser.add_argument('--output-dir', type=str, default='evaluation_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    evaluator = MandiDatasetEvaluator(args.csv_file)
    evaluator.evaluate(generate_plots=not args.no_plots)


if __name__ == "__main__":
    main()