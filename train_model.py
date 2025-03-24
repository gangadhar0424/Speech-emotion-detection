import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from decoder import decode_features

# Disable oneDNN to potentially bypass signbit issue
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def train_lstm_model(features, labels):
    print("Features shape:", features.shape, "dtype:", features.dtype)
    print("Labels shape:", labels.shape, "dtype:", labels.dtype)
    
    le = LabelEncoder()
    labels_encoded = to_categorical(le.fit_transform(labels)).astype(np.float32)
    print("Labels encoded shape:", labels_encoded.shape, "dtype:", labels_encoded.dtype)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1])).astype(np.float32)
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1])).astype(np.float32)
    print("X_train reshaped:", X_train.shape, "dtype:", X_train.dtype)
    
    model = Sequential([
        Input(shape=(1, X_train.shape[2])),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(labels_encoded.shape[1], activation='softmax')
    ])
     
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    print("Final validation accuracy:", history.history['val_accuracy'][-1])
    
    return model, le

def load_model():
    features, labels = decode_features('features.csv')
    model, le = train_lstm_model(features, labels)
    return model, le

if __name__ == "__main__":
    model, label_encoder = load_model()
    model.save('emotion_model.h5')
    np.save('label_encoder.npy', label_encoder.classes_)

