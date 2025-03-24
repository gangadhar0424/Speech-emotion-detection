import os
import numpy as np
import pandas as pd
import librosa
import glob

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        pitch_proxy = librosa.feature.zero_crossing_rate(y)
        pitch_mean = np.mean(pitch_proxy)
        pitch_std = np.std(pitch_proxy)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        rms = np.mean(librosa.feature.rms(y=y))
        energy_mean = np.mean(np.abs(y))
        energy_std = np.std(np.abs(y))
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        features = {
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'spectral_centroid': float(spectral_centroid),
            'spectral_bandwidth': float(spectral_bandwidth),
            'rms_energy': float(rms),
            'energy_mean': float(energy_mean),
            'energy_std': float(energy_std),
            **{f'mfcc_{i+1}_mean': float(mfcc_mean[i]) for i in range(13)}
        }
        for key in features:
            features[key] = np.nan_to_num(features[key], nan=0.0, posinf=1e5, neginf=-1e5)
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def prepare_dataset(audio_dir):
    features_list = []
    files = glob.glob(f"{audio_dir}/Actor_*/*.wav")
    print(f"Found {len(files)} audio files")
    if not files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}/Actor_*/*.wav")
    
    for i, file in enumerate(files):
        emotion_code = int(os.path.basename(file).split('-')[2])
        emotion_map = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 
                       5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
        emotion = emotion_map.get(emotion_code, 'unknown')
        if emotion == 'unknown': continue
        
        feature_dict = extract_features(file)
        if feature_dict:
            feature_dict['emotion'] = emotion
            features_list.append(feature_dict)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(files)} files")
    
    df = pd.DataFrame(features_list)
    df.to_csv('features.csv', index=False)
    print(f"Saved {len(df)} samples to features.csv")
    return df

if __name__ == "__main__":
    audio_dir = r'C:\Users\ganga\OneDrive\Desktop\Speech-emotion-detection\dataset'
    prepare_dataset(audio_dir)
