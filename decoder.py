import pandas as pd
import numpy as np

def decode_features(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols = ['pitch_mean', 'pitch_std', 'spectral_centroid', 'spectral_bandwidth', 
                    'rms_energy', 'energy_mean', 'energy_std'] + [f'mfcc_{i+1}_mean' for i in range(13)]
    features = df[feature_cols].values.astype(np.float32)
    labels = df['emotion'].values
    features = np.nan_to_num(features, nan=0.0, posinf=1e5, neginf=-1e5)
    return features, labels