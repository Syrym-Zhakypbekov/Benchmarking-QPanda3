"""
Generate comprehensive visualizations for IoT sensor data analysis.

Creates professional plots for Scopus Q1 paper on Quantum ML for IoT anomaly detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.iot_sensor_data import load_iot_sensor_data, IoTSensorDataGenerator

# Professional style
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

RESULTS_DIR = Path("results/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_comprehensive_iot_visualizations(df: pd.DataFrame, save_dir: Path):
    """Create all visualizations for IoT sensor data paper."""
    
    print("Generating comprehensive IoT sensor visualizations...")
    
    # 1. Time Series Analysis
    create_time_series_plot(df, save_dir)
    
    # 2. Anomaly Detection Visualization
    create_anomaly_detection_plot(df, save_dir)
    
    # 3. Feature Distribution Analysis
    create_feature_distribution_plot(df, save_dir)
    
    # 4. Correlation Heatmap
    create_correlation_heatmap(df, save_dir)
    
    # 5. 3D Vibration Space
    create_3d_vibration_plot(df, save_dir)
    
    # 6. Aftershock Analysis
    create_aftershock_analysis(df, save_dir)
    
    print(f"\n[SUCCESS] All visualizations saved to {save_dir}")


def create_time_series_plot(df: pd.DataFrame, save_dir: Path):
    """Create time series plot showing sensor readings over time."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Sample data for performance (every 100th point)
    sample_df = df.iloc[::100].copy()
    sample_df['Index'] = range(len(sample_df))
    
    # Vibration X
    ax = axes[0]
    ax.plot(sample_df['Index'], sample_df['X'], 
            linewidth=1.5, alpha=0.7, color='#2E86AB', label='X Vibration')
    anomaly_mask = sample_df['Anomaly'] == 1
    ax.scatter(sample_df[anomaly_mask]['Index'], 
               sample_df[anomaly_mask]['X'],
               color='red', s=50, alpha=0.8, label='Anomaly', zorder=5)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('X Vibration (units)', fontsize=12)
    ax.set_title('X-Axis Vibration Time Series with Anomaly Detection', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Vibration Y
    ax = axes[1]
    ax.plot(sample_df['Index'], sample_df['Y'], 
            linewidth=1.5, alpha=0.7, color='#A23B72', label='Y Vibration')
    ax.scatter(sample_df[anomaly_mask]['Index'], 
               sample_df[anomaly_mask]['Y'],
               color='red', s=50, alpha=0.8, label='Anomaly', zorder=5)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Y Vibration (units)', fontsize=12)
    ax.set_title('Y-Axis Vibration Time Series with Anomaly Detection', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temperature
    if 'Temperature' in df.columns:
        ax = axes[2]
        ax.plot(sample_df['Index'], sample_df['Temperature'], 
                linewidth=1.5, alpha=0.7, color='#F18F01', label='Temperature')
        ax.scatter(sample_df[anomaly_mask]['Index'], 
                   sample_df[anomaly_mask]['Temperature'],
                   color='red', s=50, alpha=0.8, label='Anomaly', zorder=5)
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title('Temperature Time Series with Anomaly Detection', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "iot_time_series_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Time series plot created")


def create_anomaly_detection_plot(df: pd.DataFrame, save_dir: Path):
    """Create visualization showing anomaly detection results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Anomaly distribution
    ax = axes[0, 0]
    anomaly_counts = df['Anomaly'].value_counts().sort_index()
    colors = ['#06A77D', '#D00000']
    bars = ax.bar(['Normal', 'Anomaly'], anomaly_counts.values, color=colors, alpha=0.8)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Anomaly Distribution in Dataset', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(anomaly_counts.values):
        ax.text(i, v + max(anomaly_counts.values)*0.01, 
                f'{v:,}\n({v/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Vibration magnitude distribution
    ax = axes[0, 1]
    normal_data = df[df['Anomaly']==0]['Vibration_Magnitude']
    anomaly_data = df[df['Anomaly']==1]['Vibration_Magnitude']
    ax.hist(normal_data, bins=50, alpha=0.7, label='Normal', 
            color='#06A77D', edgecolor='black', linewidth=0.5)
    ax.hist(anomaly_data, bins=50, alpha=0.7, label='Anomaly', 
            color='#D00000', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Vibration Magnitude', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Vibration Magnitude Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. 2D scatter: X vs Y with anomalies
    ax = axes[1, 0]
    sample_normal = df[df['Anomaly']==0].sample(min(5000, len(df[df['Anomaly']==0])))
    sample_anomaly = df[df['Anomaly']==1]
    ax.scatter(sample_normal['X'], sample_normal['Y'], 
               alpha=0.4, s=10, color='#06A77D', label='Normal')
    ax.scatter(sample_anomaly['X'], sample_anomaly['Y'], 
               alpha=0.8, s=30, color='#D00000', label='Anomaly', zorder=5)
    ax.set_xlabel('X Vibration', fontsize=12)
    ax.set_ylabel('Y Vibration', fontsize=12)
    ax.set_title('Vibration Pattern: X vs Y (Anomaly Detection)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Feature importance (variance)
    ax = axes[1, 1]
    if 'Vibration_Magnitude' in df.columns:
        features = ['X', 'Y', 'Z', 'Vibration_Magnitude']
        variances = [df[f].var() for f in features]
        bars = ax.barh(features, variances, color='#2E86AB', alpha=0.8)
        ax.set_xlabel('Variance', fontsize=12)
        ax.set_title('Feature Variance Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(variances):
            ax.text(v, i, f'{v:.0f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / "iot_anomaly_detection_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Anomaly detection plot created")


def create_feature_distribution_plot(df: pd.DataFrame, save_dir: Path):
    """Create feature distribution analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    features = ['X', 'Y', 'Z']
    if 'Temperature' in df.columns:
        features.extend(['Temperature', 'Humidity', 'Pressure'])
    
    for idx, feature in enumerate(features[:6]):
        ax = axes[idx // 3, idx % 3]
        
        normal_data = df[df['Anomaly']==0][feature]
        anomaly_data = df[df['Anomaly']==1][feature]
        
        ax.hist(normal_data, bins=40, alpha=0.6, label='Normal', 
                color='#06A77D', edgecolor='black', linewidth=0.3)
        ax.hist(anomaly_data, bins=40, alpha=0.6, label='Anomaly', 
                color='#D00000', edgecolor='black', linewidth=0.3)
        
        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / "iot_feature_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Feature distribution plot created")


def create_correlation_heatmap(df: pd.DataFrame, save_dir: Path):
    """Create correlation heatmap of features."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select numeric features
    numeric_features = ['X', 'Y', 'Z', 'Aftershocks']
    if 'Temperature' in df.columns:
        numeric_features.extend(['Temperature', 'Humidity', 'Pressure', 'Vibration_Magnitude'])
    
    corr_matrix = df[numeric_features].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_dir / "iot_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Correlation heatmap created")


def create_3d_vibration_plot(df: pd.DataFrame, save_dir: Path):
    """Create 3D visualization of vibration space."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample data
    sample_normal = df[df['Anomaly']==0].sample(min(2000, len(df[df['Anomaly']==0])))
    sample_anomaly = df[df['Anomaly']==1].sample(min(500, len(df[df['Anomaly']==1])))
    
    ax.scatter(sample_normal['X'], sample_normal['Y'], sample_normal['Z'],
               c='#06A77D', alpha=0.4, s=10, label='Normal')
    ax.scatter(sample_anomaly['X'], sample_anomaly['Y'], sample_anomaly['Z'],
               c='#D00000', alpha=0.8, s=30, label='Anomaly')
    
    ax.set_xlabel('X Vibration', fontsize=12)
    ax.set_ylabel('Y Vibration', fontsize=12)
    ax.set_zlabel('Z Vibration', fontsize=12)
    ax.set_title('3D Vibration Space: Normal vs Anomaly', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / "iot_3d_vibration_space.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] 3D vibration plot created")


def create_aftershock_analysis(df: pd.DataFrame, save_dir: Path):
    """Create aftershock event analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Aftershock timeline
    ax = axes[0, 0]
    aftershock_col = 'Aftershocks_Binary' if 'Aftershocks_Binary' in df.columns else 'Aftershocks'
    if aftershock_col in df.columns:
        if df[aftershock_col].dtype == 'int64':
            aftershock_mask = df[aftershock_col] == 1
        else:
            aftershock_mask = df[aftershock_col] > df[aftershock_col].quantile(0.95)
        aftershock_indices = df[aftershock_mask].index
    else:
        aftershock_indices = df.index[:100]  # Fallback
    ax.scatter(aftershock_indices, df.loc[aftershock_indices, 'Vibration_Magnitude'],
               color='orange', s=50, alpha=0.7, label='Aftershock Events')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Vibration Magnitude', fontsize=12)
    ax.set_title('Aftershock Events Timeline', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Aftershock vs Normal comparison
    ax = axes[0, 1]
    aftershock_col = 'Aftershocks_Binary' if 'Aftershocks_Binary' in df.columns else 'Aftershocks'
    if aftershock_col in df.columns:
        aftershock_mask = df[aftershock_col] == 1 if df[aftershock_col].dtype == 'int64' else df[aftershock_col] > df[aftershock_col].quantile(0.95)
        aftershock_data = df[aftershock_mask]['Vibration_Magnitude']
        normal_mask = ~aftershock_mask
        normal_sample = df[normal_mask].sample(min(5000, len(df[normal_mask]))) if normal_mask.sum() > 0 else df.sample(min(5000, len(df)))
        normal_data = normal_sample['Vibration_Magnitude']
    ax.boxplot([normal_data, aftershock_data], labels=['Normal', 'Aftershock'],
               patch_artist=True,
               boxprops=dict(facecolor='#06A77D', alpha=0.7),
               medianprops=dict(color='black', linewidth=2))
    ax.set_ylabel('Vibration Magnitude', fontsize=12)
    ax.set_title('Aftershock vs Normal Vibration Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Aftershock frequency
    ax = axes[1, 0]
    if 'Timestamp' in df.columns:
        df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
        aftershock_by_hour = df[df['Aftershocks']==1].groupby('Hour').size()
        ax.bar(aftershock_by_hour.index, aftershock_by_hour.values, 
               color='orange', alpha=0.7)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Aftershock Count', fontsize=12)
        ax.set_title('Aftershock Frequency by Hour', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Aftershock impact on anomalies
    ax = axes[1, 1]
    aftershock_col = 'Aftershocks_Binary' if 'Aftershocks_Binary' in df.columns else 'Aftershocks'
    if aftershock_col in df.columns:
        if df[aftershock_col].dtype != 'int64':
            df_temp = df.copy()
            df_temp['Aftershocks_Binary'] = (df_temp[aftershock_col] > df_temp[aftershock_col].quantile(0.95)).astype(int)
            cross_tab = pd.crosstab(df_temp['Aftershocks_Binary'], df_temp['Anomaly'], normalize='index') * 100
        else:
            cross_tab = pd.crosstab(df[aftershock_col], df['Anomaly'], normalize='index') * 100
    cross_tab.plot(kind='bar', ax=ax, color=['#06A77D', '#D00000'], alpha=0.8)
    ax.set_xlabel('Aftershock Event', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Anomaly Rate: Aftershock vs Normal', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['No Aftershock', 'Aftershock'], rotation=0)
    ax.legend(['Normal', 'Anomaly'])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / "iot_aftershock_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Aftershock analysis plot created")


if __name__ == "__main__":
    print("="*70)
    print("GENERATING IoT SENSOR DATA VISUALIZATIONS")
    print("For Scopus Q1 Paper: Quantum ML for IoT Anomaly Detection")
    print("="*70)
    
    # Load or generate data
    if Path("1.exl.csv").exists():
        print("\nLoading existing sensor data...")
        df = load_iot_sensor_data("1.exl.csv")
    else:
        print("\nGenerating synthetic sensor data...")
        generator = IoTSensorDataGenerator(n_samples=50000, anomaly_rate=0.05)
        df = generator.generate_complete_dataset()
    
    print(f"\nDataset loaded: {len(df)} samples")
    print(f"  Anomalies: {df['Anomaly'].sum()} ({df['Anomaly'].mean()*100:.1f}%)")
    print(f"  Aftershocks: {df['Aftershocks'].sum()}")
    
    # Generate all visualizations
    create_comprehensive_iot_visualizations(df, RESULTS_DIR)
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS GENERATED!")
    print("="*70)
