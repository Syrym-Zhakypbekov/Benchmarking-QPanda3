"""
IoT Sensor Data Generator for Building Monitoring and Anomaly Detection.

This module generates synthetic sensor data simulating IoT devices monitoring
buildings/houses for structural health, including:
- Vibration sensors (X, Y, Z accelerometers)
- Aftershock detection
- Temperature, humidity, pressure
- Timestamp data

Designed for quantum anomaly detection using QPanda3 at IITU, Almaty, Kazakhstan.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class IoTSensorDataGenerator:
    """
    Generate synthetic IoT sensor data for building monitoring.
    
    Simulates sensors monitoring structural health, vibrations, and environmental
    conditions in buildings/houses.
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        start_date: str = "2025-01-01",
        anomaly_rate: float = 0.05,
        random_state: int = 42
    ):
        """
        Initialize sensor data generator.
        
        Args:
            n_samples: Number of sensor readings to generate
            start_date: Starting date for timestamps
            anomaly_rate: Proportion of anomalous readings (0.0-1.0)
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.anomaly_rate = anomaly_rate
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_vibration_data(
        self,
        include_aftershocks: bool = True,
        aftershock_probability: float = 0.02
    ) -> pd.DataFrame:
        """
        Generate vibration sensor data (X, Y, Z accelerometers).
        
        Args:
            include_aftershocks: Whether to include aftershock events
            aftershock_probability: Probability of aftershock per reading
            
        Returns:
            DataFrame with Date, Time, Aftershocks, X, Y, Z columns
        """
        data = []
        current_time = self.start_date
        
        # Normal vibration baseline (micro-meters or g-force units)
        baseline_x = 2000 + np.random.normal(0, 50, self.n_samples)
        baseline_y = 65400 + np.random.normal(0, 100, self.n_samples)
        baseline_z = 180 + np.random.normal(0, 5, self.n_samples)
        
        # Generate aftershock events
        aftershock_indices = np.random.choice(
            self.n_samples,
            size=int(self.n_samples * aftershock_probability),
            replace=False
        )
        
        for i in range(self.n_samples):
            # Determine if this is an aftershock event
            is_aftershock = i in aftershock_indices if include_aftershocks else False
            
            if is_aftershock:
                # Aftershock: increased vibration
                x = baseline_x[i] + np.random.normal(100, 20)
                y = baseline_y[i] + np.random.normal(500, 100)
                z = baseline_z[i] + np.random.normal(10, 3)
                aftershock_flag = 1
            else:
                # Normal reading
                x = baseline_x[i]
                y = baseline_y[i]
                z = baseline_z[i]
                aftershock_flag = 0
            
            # Add anomalies (structural damage, equipment failure, etc.)
            if np.random.random() < self.anomaly_rate:
                # Anomaly: sudden spike or drop
                anomaly_type = np.random.choice(['spike', 'drop', 'oscillation'])
                if anomaly_type == 'spike':
                    x *= np.random.uniform(1.5, 3.0)
                    y *= np.random.uniform(1.3, 2.5)
                    z *= np.random.uniform(1.2, 2.0)
                elif anomaly_type == 'drop':
                    x *= np.random.uniform(0.3, 0.7)
                    y *= np.random.uniform(0.4, 0.8)
                    z *= np.random.uniform(0.5, 0.9)
                else:  # oscillation
                    freq = np.random.uniform(0.1, 0.5)
                    x += 200 * np.sin(2 * np.pi * freq * i)
                    y += 500 * np.sin(2 * np.pi * freq * i)
                    z += 20 * np.sin(2 * np.pi * freq * i)
            
            # Format timestamp
            date_str = current_time.strftime("%d.%m.%Y")
            time_str = current_time.strftime("%H:%M:%S")
            
            data.append({
                'Date': date_str,
                'Time': time_str,
                'Aftershocks': aftershock_flag,
                'X': int(x),
                'Y': int(y),
                'Z': int(z),
                'Timestamp': current_time
            })
            
            # Increment time (every 10 seconds)
            current_time += timedelta(seconds=10)
        
        df = pd.DataFrame(data)
        return df
    
    def generate_environmental_data(self) -> pd.DataFrame:
        """
        Generate environmental sensor data (temperature, humidity, pressure).
        
        Returns:
            DataFrame with environmental readings
        """
        data = []
        current_time = self.start_date
        
        # Baseline environmental conditions (Almaty, Kazakhstan climate)
        base_temp = 15.0  # Average temperature in Almaty
        base_humidity = 60.0
        base_pressure = 1013.25  # hPa
        
        for i in range(self.n_samples):
            # Daily temperature cycle
            hour = current_time.hour
            temp_variation = 10 * np.sin(2 * np.pi * hour / 24)
            temperature = base_temp + temp_variation + np.random.normal(0, 2)
            
            # Humidity inversely correlated with temperature
            humidity = base_humidity - (temp_variation / 2) + np.random.normal(0, 5)
            humidity = np.clip(humidity, 20, 90)
            
            # Pressure varies slowly
            pressure = base_pressure + np.random.normal(0, 5)
            
            # Add anomalies
            if np.random.random() < self.anomaly_rate:
                temperature += np.random.choice([-15, 15])  # Sudden temp change
                humidity += np.random.choice([-30, 30])  # Sudden humidity change
                pressure += np.random.choice([-20, 20])  # Pressure anomaly
            
            data.append({
                'Timestamp': current_time,
                'Temperature': round(temperature, 2),
                'Humidity': round(humidity, 2),
                'Pressure': round(pressure, 2)
            })
            
            current_time += timedelta(seconds=10)
        
        return pd.DataFrame(data)
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """
        Generate complete IoT sensor dataset with all features.
        
        Returns:
            Combined DataFrame with vibration and environmental data
        """
        vibration_df = self.generate_vibration_data()
        env_df = self.generate_environmental_data()
        
        # Merge on timestamp
        df = vibration_df.merge(env_df, on='Timestamp', how='inner')
        
        # Add derived features
        df['Vibration_Magnitude'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
        df['Vibration_Variance'] = df[['X', 'Y', 'Z']].var(axis=1)
        
        # Add anomaly label (1 if any anomaly detected)
        df['Anomaly'] = (
            (df['Aftershocks'] == 1) |
            (df['Vibration_Magnitude'] > df['Vibration_Magnitude'].quantile(0.95)) |
            (df['Temperature'].abs() > df['Temperature'].std() * 3) |
            (df['Humidity'].abs() > df['Humidity'].std() * 3)
        ).astype(int)
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = "iot_sensor_data.csv"):
        """Save generated data to CSV file."""
        # Remove Timestamp column for CSV (keep Date/Time)
        df_export = df.drop('Timestamp', axis=1, errors='ignore')
        df_export.to_csv(filename, index=False, sep=';')
        print(f"Saved {len(df)} samples to {filename}")
        return filename


def load_iot_sensor_data(filename: str = "1.exl.csv") -> pd.DataFrame:
    """
    Load existing IoT sensor data from CSV.
    
    Args:
        filename: Path to CSV file
        
    Returns:
        DataFrame with sensor data
    """
    df = pd.read_csv(filename, sep=';')
    
    # Parse timestamps
    df['Timestamp'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d.%m.%Y %H:%M:%S'
    )
    
    # Fix Aftershocks column - convert to binary (1 if > threshold, else 0)
    # Original data seems to have numeric values, convert to binary
    aftershock_threshold = df['Aftershocks'].quantile(0.95) if len(df) > 0 else 1000
    df['Aftershocks_Binary'] = (df['Aftershocks'] > aftershock_threshold).astype(int)
    
    # Add derived features
    df['Vibration_Magnitude'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    df['Vibration_Variance'] = df[['X', 'Y', 'Z']].var(axis=1)
    
    return df


if __name__ == "__main__":
    # Generate synthetic IoT sensor data
    print("Generating IoT Sensor Data for Building Monitoring...")
    generator = IoTSensorDataGenerator(n_samples=50000, anomaly_rate=0.05)
    
    # Generate complete dataset
    df = generator.generate_complete_dataset()
    
    print(f"\nGenerated Dataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Anomalies: {df['Anomaly'].sum()} ({df['Anomaly'].mean()*100:.1f}%)")
    print(f"  Aftershocks: {df['Aftershocks'].sum()}")
    print(f"\nFeature ranges:")
    print(f"  X: {df['X'].min()} - {df['X'].max()}")
    print(f"  Y: {df['Y'].min()} - {df['Y'].max()}")
    print(f"  Z: {df['Z'].min()} - {df['Z'].max()}")
    print(f"  Temperature: {df['Temperature'].min():.1f} - {df['Temperature'].max():.1f}°C")
    
    # Save to CSV
    generator.save_to_csv(df, "iot_sensor_data_synthetic.csv")
