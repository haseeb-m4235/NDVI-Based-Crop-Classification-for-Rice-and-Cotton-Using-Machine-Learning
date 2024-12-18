from Data_Processing import DataPreProcess

dp=DataPreProcess()
data=dp.apply_preprocessing()
print(data['X_train_3'].var())
print(data['y_train_3'].value_counts())