import os
import pandas as pd

class DataSplit():
    def __init__(self, datasetDir):
        """
        Args:
            datasetDir (str): Path to the dataset directory.
        """
        self.datasetDir = datasetDir
        self.combining_data()
        
    def combining_data(self):
        """
        Combines the data from the different csv files into one dataframe.

        The cotton and rice data are combined into one dataframe with the 'label' column
        indicating whether it is a cotton (1) or rice (0) sample and the 'year' column indicating
        the year of the data (2021, 2022 or 2023). The combined dataframe is stored as an
        instance variable.

        Returns:
            None
        """
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
        """ 
        Splits the combined data into a training set and a testing set.

        Args:
            train_data_1 (int): The year of the first set of training data.
            train_data_2 (int): The year of the second set of training data.
            test_data (int): The year of the testing data.

        Returns:
            A tuple of four dataframes: X_train, y_train, X_test, y_test. X_train and X_test are the feature data for
            the training and testing sets, respectively. y_train and y_test are the labels for the training and testing
            sets, respectively.
        """

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