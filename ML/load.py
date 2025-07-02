import pandas as pd 
import numpy as np 
import joblib

SVC_model = joblib.load("SVC.joblib")
RFC_model = joblib.load("RFC.joblib")

test_data = pd.read_csv('./DrivingPattern (continous)/test_motion_data.csv')
features = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
test_data = test_data[features]

AccMeanX = test_data['AccX'].mean()
AccMeanY = test_data['AccY'].mean()
AccMeanZ = test_data['AccZ'].mean()
GyroMeanX = test_data['GyroX'].mean()
GyroMeanY = test_data['GyroY'].mean()
GyroMeanZ = test_data['GyroZ'].mean()

test_data['AccMeanX'] = test_data['AccX'].div(AccMeanX)
test_data['AccMeanY'] = test_data['AccY'].div(AccMeanY)
test_data['AccMeanZ'] = test_data['AccZ'].div(AccMeanZ)
test_data['GyroMeanX'] = test_data['GyroX'].div(GyroMeanX)
test_data['GyroMeanY'] = test_data['GyroY'].div(GyroMeanY)
test_data['GyroMeanZ'] = test_data['GyroZ'].div(GyroMeanZ)

filtered_features = ['AccMeanX', 'AccMeanY', 'AccMeanX', 'GyroMeanX', 'GyroMeanY', 'GyroMeanZ']
test_data_filtered = test_data[filtered_features]
print(test_data_filtered.head())
