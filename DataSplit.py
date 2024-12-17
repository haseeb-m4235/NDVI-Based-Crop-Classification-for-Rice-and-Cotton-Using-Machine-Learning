import os
import pandas as pd

class DataSplit():
    def __init__(self, datasetDir):
        self.datasetDir = datasetDir
        self.combining_data()
        
    def combining_data(self):
        cotton_2021_path = os.path.join(self.datasetDir, "Cotton", "cotton2021.csv")
        cotton_2022_path = os.path.join(self.datasetDir, "Cotton", "cotton2022.csv")
        cotton_2023_path = os.path.join(self.datasetDir, "Cotton", "cotton2023.csv")
        rice_2021_path =  os.path.join(self.datasetDir, "Rice", "Rice2021.csv")
        rice_2022_path = os.path.join(self.datasetDir, "Rice", "Rice2022.csv")
        rice_2023_path = os.path.join(self.datasetDir, "Rice", "Rice2023.csv")

        cotton_2021_df = pd.read_csv(cotton_2021_path)
        cotton_2022_df = pd.read_csv(cotton_2022_path)
        cotton_2023_df = pd.read_csv(cotton_2023_path)
        rice_2021_df = pd.read_csv(rice_2021_path)
        rice_2022_df = pd.read_csv(rice_2022_path)
        rice_2023_df = pd.read_csv(rice_2023_path)

        cotton_2021_df['label'] = 1 
        cotton_2021_df['year'] = 2021 
        rice_2021_df['label'] = 0
        rice_2021_df['year'] = 2021
        cotton_2022_df['label'] = 1 
        cotton_2022_df['year'] = 2022 
        rice_2022_df['label'] = 0
        rice_2022_df['year'] = 2022
        cotton_2023_df['label'] = 1 
        cotton_2023_df['year'] = 2023
        rice_2023_df['label'] = 0
        rice_2023_df['year'] = 2023

        self.combined_data = pd.concat([cotton_2021_df, rice_2021_df, cotton_2022_df, rice_2022_df, cotton_2023_df, rice_2023_df], axis=0)

    def get_split(self, train_data_1, train_data_2, test_data):
        train_df = self.combined_data[(self.combined_data['year'] == train_data_1) | (self.combined_data['year'] == train_data_2)]
        test_df = self.combined_data[(self.combined_data['year'] == test_data)]
        
        X_train = train_df.drop(columns=['label', 'year'], axis=1)
        X_test = test_df.drop(columns=['label', 'year'], axis=1)

        y_train = train_df['label']
        y_test = test_df['label']

        # Count samples in training dataset  
        train_cotton_count = train_df[train_df['label'] == 1].shape[0]  
        train_rice_count = train_df[train_df['label'] == 0].shape[0]  

        # Count samples in tesdfng dftaset  
        test_cotton_count = test_df[test_df['label'] == 1].shape[0]  
        test_rice_count = test_df[test_df['label'] == 0].shape[0]  

        # Print results  
        print("Training Dataset:")  
        print(f"Cotton samples: {train_cotton_count}")  
        print(f"Rice samples: {train_rice_count}")  
        print("\nTesting Dataset:")  
        print(f"Cotton samples: {test_cotton_count}")  
        print(f"Rice samples: {test_rice_count}")  
        
        return X_train, y_train, X_test, y_test