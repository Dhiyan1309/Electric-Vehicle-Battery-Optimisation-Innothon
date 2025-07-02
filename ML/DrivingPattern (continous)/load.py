import pandas as pd 
import numpy as np 
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib

train_data = pd.read_csv('train_motion_data.csv')
encoder = LabelEncoder()

train_data['Class'] = encoder.fit_transform(train_data['Class'])
model = load_model('./lstm_driving_behavior.h5')
scaler = joblib.load('scaler.pkl')

data = pd.read_csv('test_motion_data.csv')

features = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
data[features] = scaler.fit_transform(data[features])

sequence_length = 10

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
    return np.array(X)

X_data = create_sequences(data[features], sequence_length);
y_pred = model.predict(X_data)
predicted_class_index = np.argmax(y_pred, axis=1)
predicted_label = encoder.inverse_transform(predicted_class_index)

x = np.random.randint(3000, size=(10))

for i in x:
    print(predicted_label[i])
