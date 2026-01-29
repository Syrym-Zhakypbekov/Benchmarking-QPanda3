"""
Main script to run IoT sensor data experiments and generate visualizations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.iot_sensor_data import load_iot_sensor_data, IoTSensorDataGenerator
from src.experiments.generate_iot_visualizations import create_comprehensive_iot_visualizations

def main():
    """Run IoT experiments."""
    print("="*70)
    print("IoT SENSOR DATA EXPERIMENTS")
    print("Quantum ML for Building Monitoring - IITU, Almaty, Kazakhstan")
    print("="*70)
    
    # Load or generate data
    data_file = Path("1.exl.csv")
    if data_file.exists():
        print("\nLoading existing sensor data from 1.exl.csv...")
        df = load_iot_sensor_data("1.exl.csv")
        
        # Add anomaly labels based on data
        df['Vibration_Magnitude'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
        
        # Fix Aftershocks - convert to binary
        aftershock_threshold = df['Aftershocks'].quantile(0.95)
        df['Aftershocks_Binary'] = (df['Aftershocks'] > aftershock_threshold).astype(int)
        
        df['Anomaly'] = (
            (df['Aftershocks_Binary'] == 1) | 
            (df['Vibration_Magnitude'] > df['Vibration_Magnitude'].quantile(0.95))
        ).astype(int)
    else:
        print("\nGenerating synthetic sensor data...")
        generator = IoTSensorDataGenerator(n_samples=50000, anomaly_rate=0.05)
        df = generator.generate_complete_dataset()
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Anomalies: {df['Anomaly'].sum():,} ({df['Anomaly'].mean()*100:.1f}%)")
    if 'Aftershocks' in df.columns:
        print(f"  Aftershocks: {df['Aftershocks'].sum():,}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    create_comprehensive_iot_visualizations(df, Path("results/figures"))
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  - iot_time_series_analysis.png")
    print("  - iot_anomaly_detection_analysis.png")
    print("  - iot_feature_distributions.png")
    print("  - iot_correlation_heatmap.png")
    print("  - iot_3d_vibration_space.png")
    print("  - iot_aftershock_analysis.png")

if __name__ == "__main__":
    main()
