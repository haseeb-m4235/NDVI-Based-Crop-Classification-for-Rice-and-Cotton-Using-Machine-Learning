import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from DataSplit import DataSplit
import random
class DataPreProcess:
    def scalling(self, X_train, X_test, scaler=None):
        """
        This function performs scaling on the numeric columns of the given dataset.
        The default scaler is StandardScaler(). If you want to apply a different
        scaler, pass the scaler object in the `scaler` argument.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features dataset.
        X_test : pd.DataFrame
            Testing features dataset.
        scaler : object, optional
            Scaler object (default is StandardScaler()).
        
        Returns
        -------
        tuple
            Scaled training and testing datasets.
        """
        # Use StandardScaler as default if no scaler is provided
        if scaler is None:
            scaler = StandardScaler()
        
        # Identify columns to exclude from scaling
        exclude_columns = ['year', 'label']
        
        # Select columns to standardize
        columns_to_standardize = [
            col for col in X_train.columns 
            if col not in exclude_columns and 
               pd.api.types.is_numeric_dtype(X_train[col])
        ]
        
        # Create copies to avoid modifying original dataframes
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Fit the scaler on X_train numeric columns and transform both train and test
        if columns_to_standardize:
            X_train_scaled[columns_to_standardize] = scaler.fit_transform(
                X_train[columns_to_standardize]
            )
            X_test_scaled[columns_to_standardize] = scaler.transform(
                X_test[columns_to_standardize]
            )
        
        # Return scaled datasets
        return X_train_scaled, X_test_scaled
    def null_detector(self,data):
        print("The null in dataset are:\n",data.isna().sum())

    def imputer(self):
        pass

    def plot_correlation(self,data):
        """
    Plots a heatmap of the correlation matrix for the given dataset.

    Parameters:
    ----------
    data : pandas.DataFrame
        A DataFrame containing the dataset for which the correlation matrix 
        will be computed and visualized.

    Returns:
    -------
    None
        Displays a heatmap of the correlation matrix.

    Notes:
    -----
    - The function calculates the Pearson correlation coefficients for all
      numeric features in the dataset using `DataFrame.corr()`.
    - The heatmap is plotted using Seaborn's `heatmap()` function with annotations
      and a "coolwarm" color map for better visualization of positive and negative correlations.
    - This function is useful for understanding relationships between features 
      in the dataset and identifying multicollinearity.
        """
        # Step 1: Calculate correlation matrix
        correlation_matrix = data.corr()

        # Step 2: Visualize the correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Matrix")
        plt.show()

    def address_correlation(self,data,columns_list:list,X_train=None,X_test=None,drop_train_test=False):
        """
    Removes specified columns from the training and testing datasets to address multicollinearity.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training dataset containing feature columns.
    X_test : pd.DataFrame
        The testing dataset containing feature columns.
    columns_list : list
        A list of column names to be dropped from both the training and testing datasets.

    Returns
    -------
    tuple
        A tuple containing two DataFrames:
        - `new_X_train`: The training dataset after removing the specified columns.
        - `new_X_test`: The testing dataset after removing the specified columns.

    Notes
    -----
    - This function is useful when you have identified features with high correlation 
      or multicollinearity that need to be removed from your dataset.
    - The function creates copies of the input datasets to avoid modifying the original DataFrames.

    Examples
    --------
    dp = DataPreProcess()
    X_train, X_test = dp.address_correlation(X_train, X_test, ['NDVI01', 'NDVI02'])
    print(X_train.columns)  # NDVI01 and NDVI02 should no longer be in the columns
        """
        if drop_train_test:
            new_X_test=X_test.copy()
            new_X_train=X_train.copy()
            new_X_test.drop(columns=columns_list,inplace=True)
            new_X_train.drop(columns=columns_list,inplace=True)
            return new_X_train,new_X_test
        else:
            data.drop(columns=columns_list,inplace=True)
            return data

        
        
        
        
        return new_X_train,new_X_test
        
    def handle_imbalance(self,X_train,y_train,desc):
        """
    Handles class imbalance in the training dataset using the Synthetic Minority Oversampling Technique (SMOTE).

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        The training feature dataset.
    y_train : pd.Series or np.ndarray
        The training target labels with imbalanced classes.

    Returns
    -------
    tuple
        A tuple containing:
        - `X_train_sm` (pd.DataFrame or np.ndarray): The oversampled training feature dataset.
        - `y_train_sm` (pd.Series or np.ndarray): The oversampled training target labels.

    Notes
    -----
    - This function uses SMOTE to generate synthetic samples for the minority class, 
      thereby balancing the class distribution in the training dataset.
    - It prints the original and new class distributions for reference.
    - SMOTE works by creating synthetic samples between existing minority class data points.

    Examples
    --------
    dp = DataPreProcess()
    X_train_sm, y_train_sm = dp.handle_imbalance(X_train, y_train)
    Original class distribution: Counter({0: 1000, 1: 100})
    New class distribution after SMOTE: Counter({0: 1000, 1: 1000})
    Total samples after augmentation: 2000
        """
        # Define class label mapping
        label_mapping = {0: "rice", 1: "cotton"}

        # Map numeric labels to class names for plotting
        def map_labels(counter):
            return {label_mapping[k]: v for k, v in counter.items()}

        # Plot original class distribution
        original_distribution = Counter(y_train)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        mapped_original = map_labels(original_distribution)
        plt.bar(mapped_original.keys(), mapped_original.values(), color='skyblue')
        plt.title(desc)
        plt.xlabel("Class Labels")
        plt.ylabel("Count")

        # oversampling the train dataset using SMOTE
        smt = SMOTE()
        X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
        # Check the new class distribution
        new_distribution = Counter(y_train_sm)

        # Plot new class distribution
        plt.subplot(1, 2, 2)
        mapped_new = map_labels(new_distribution)
        plt.bar(mapped_new.keys(), mapped_new.values(), color='orange')
        plt.title(f"{desc} After SMOTE")
        plt.xlabel("Class Labels")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        # Print results
        print("Original class distribution:", Counter(y_train))
        print("New class distribution after SMOTE:", new_distribution)
        print("Total samples after augmentation:", len(y_train_sm))
        return X_train_sm, y_train_sm
    
    def apply_preprocessing(self):

        dataset_dir = 'Crop-dataset'
        data = DataSplit(datasetDir=dataset_dir)

        self.null_detector(data.combined_data)
        print("As there is no Null in Dataset so no removal us needed\n")

        # Plotting Correlation
        print("Now we will check for the correlation in dataset\n")
        self.plot_correlation(data.combined_data)

        # Addressing Correlation
        print("As the data is a time series data so we will not remove the correlated columns because it can lead to data loss")
        print(data.combined_data.head(5))

        print("Splitting the data:\n")
        data_split=data.get_combined_split()

        # Applyin TS Augmentation
        print("Now Applyin TS Augmentation (See DocString for details)")
        data_split["X_train_1"]=self.augment_time_series(data_split["X_train_1"],list(data_split["X_train_1"].columns))
        data_split["X_train_2"]=self.augment_time_series(data_split["X_train_2"],list(data_split["X_train_2"].columns))
        data_split["X_train_3"]=self.augment_time_series(data_split["X_train_3"],list(data_split["X_train_3"].columns))
        
        # Performing Scalling
        data_split["X_train_1"],data_split["X_test_1"]=self.scalling(data_split["X_train_1"],data_split["X_test_1"])
        data_split["X_train_2"],data_split["X_test_2"]=self.scalling(data_split["X_train_2"],data_split["X_test_2"])
        data_split["X_train_3"],data_split["X_test_3"]=self.scalling(data_split["X_train_3"],data_split["X_test_3"])

        # Handling Imbalance
        data_split["X_train_1"],data_split["y_train_1"]=self.handle_imbalance(data_split["X_train_1"],data_split["y_train_1"],desc="Class Distribution on 2022 and 2023 split")
        data_split["X_train_2"],data_split["y_train_2"]=self.handle_imbalance(data_split["X_train_2"],data_split["y_train_2"],desc="Class Distribution on 2021 and 2023 split")
        data_split["X_train_3"],data_split["y_train_3"]=self.handle_imbalance(data_split["X_train_3"],data_split["y_train_3"],desc="Class Distribution on 2021 and 2022 split")

        '''Add feature engineering like log transform or polynomial feature etc'''

        return data_split
    def scale_unsupervised(self,data):
        scaler = StandardScaler()
        data_scaled=data.copy()
        data_scaled=scaler.fit_transform(data_scaled)

        return data_scaled
    

    def augment_time_series(self,df, column_list, noise_factor=0.05, shift_factor=1):
        """
    Apply time-series data augmentation to simulate real-world variations in NDVI data.

    This function performs two augmentation techniques:
    1. **Adding Noise:** Random Gaussian noise is added to the specified columns to mimic real-world measurement variations.
    2. **Time-Series Shifting:** Each specified column is shifted by a random number of steps within the range [-shift_factor, shift_factor]. 
       Missing values introduced by the shift are filled with 0.

    Args:
        df (pd.DataFrame): Input dataset containing NDVI time-series data.
        column_list (list of str): List of column names to apply the augmentation on.
        noise_factor (float, optional): Standard deviation of the Gaussian noise to add. Default is 0.05.
        shift_factor (int, optional): Maximum number of steps to shift the time series forward or backward. Default is 1.

    Returns:
        pd.DataFrame: Augmented dataset with noise and time-series shifting applied to the specified columns.

    Notes:
        - Adding noise simulates slight variations in NDVI values, reflecting real-world data acquisition noise.
        - Time-series shifting adjusts the temporal alignment of the NDVI data, helping the model learn invariant features.
        - Ensure that the input `df` is a pandas DataFrame, and the `column_list` contains valid column names from `df`.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.DataFrame({
        ...     "NDVI1": np.random.rand(10),
        ...     "NDVI2": np.random.rand(10)
        ... })
        >>> processor = DataProcessor()  # Assuming this is a class with this method
        >>> augmented_data = processor.augment_time_series(data, ["NDVI1", "NDVI2"], noise_factor=0.1, shift_factor=2)
        >>> print(augmented_data)
    """
        # Add noise to the data (simulating real-world variations)
        df_new=df.copy()
        for col in column_list:
            noise = np.random.normal(0, noise_factor, df_new[col].shape)
            df_new[col] += noise

        # Time-series shifting (shifting the series by a random amount)
        for col in column_list:
            shift = random.randint(-shift_factor, shift_factor)
            df_new[col] = df_new[col].shift(shift, fill_value=0)

        return df_new
    
    def show_boxplot(self,df,string):
        df=df.copy()
        plt.rcParams['figure.figsize'] = [14,6]
        sns.boxplot(data = df, orient="v")
        plt.title(f"{string} Outliers Distribution", fontsize = 16)
        plt.ylabel("Range", fontweight = 'bold')
        plt.xlabel("Attributes", fontweight = 'bold')
   

    def remove_outliers(self,data):
        df = data.copy()

        for col in list(df.columns):
        
              Q1 = df[str(col)].quantile(0.05)
              Q3 = df[str(col)].quantile(0.95)
              IQR = Q3 - Q1
              lower_bound = Q1 - 1.5*IQR
              upper_bound = Q3 + 1.5*IQR

              df = df[(df[str(col)] >= lower_bound) & 

                (df[str(col)] <= upper_bound)]

        return df


    def apply_unsupervised_processing(self):
        """
    Perform preprocessing steps for unsupervised learning on NDVI-based crop data.

    This function performs the following preprocessing steps:
    1. **Null Detection:** Detects and reports any missing values in the dataset.
    2. **Feature Selection:** Drops irrelevant features, such as 'label' and 'year', from the dataset.
    3. **Time-Series Augmentation:** Applies augmentation techniques to the NDVI time-series data, including:
       - Adding Gaussian noise to simulate real-world measurement variations.
       - Random time-series shifting to enhance temporal robustness.
    4. **Feature Scaling:** Scales the augmented features to standardize the data.

    Args:
        None (relies on class attributes for dataset access and processing).

    Returns:
        tuple:
            - pd.DataFrame: Processed feature dataset with augmented and scaled NDVI values.
            - pd.Series: Corresponding labels for the data.

    Notes:
        - This function is designed for unsupervised learning tasks and assumes the presence of NDVI time-series data in the dataset.
        - Time-series augmentation is applied before scaling to ensure that noise and shifts are preserved during the preprocessing.
        - The labels are extracted before feature processing and returned unmodified.

    Example:
        >>> process = DataPreProcess()
        >>> features, labels = process.apply_unsupervised_processing()
        >>> print(features.shape)
        >>> print(labels.head())

    """
        print("Initializing Data Preprocessing....")
        dataset_dir = 'Crop-dataset'
        data = DataSplit(datasetDir=dataset_dir)
        print("")
        print("Detecting Nulls")
        self.null_detector(data.combined_data)
        print("As there is no Null in Dataset so no removal us needed\n")


        # Getting Features and Labels
        labels=data.combined_data['label']
        data.combined_data.drop(['label','year'],axis=1,inplace=True)

        # Applying Time series Augmentation
        # print("Applying Time series augmentation")
        # features=self.augment_time_series(data.combined_data, list(data.combined_data.columns))
        # print("After applying TS augmentatio")
        # print(features.head(5))

        print("Ploting BOX Plot for outliers")
        self.show_boxplot(data.combined_data,string="Before Removal")

        print("Adressing Outliers")
        features=self.remove_outliers(data.combined_data)
        print("Now verify the outliers")
        self.show_boxplot(features,string="After Removal")

        # Applying Scalling
        print("")
        print("Applying Scalling")
        features=self.scale_unsupervised(features)
        print("Data Processing is completed")
        

        return features,labels



















    



