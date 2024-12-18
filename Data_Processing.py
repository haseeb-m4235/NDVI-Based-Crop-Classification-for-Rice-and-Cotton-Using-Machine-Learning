import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from DataSplit import DataSplit
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
    def null_detector(self):
        pass

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

    def address_correlation(self,X_train,X_test,columns_list:list):
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
    >>> dp = DataPreProcess()
    >>> X_train, X_test = dp.address_correlation(X_train, X_test, ['NDVI01', 'NDVI02'])
    >>> print(X_train.columns)  # NDVI01 and NDVI02 should no longer be in the columns
        """
        new_X_train=X_train.copy()
        new_X_test=X_test.copy()
        new_X_train.drop(columns=columns_list,inplace=True)
        new_X_test.drop(columns=columns_list,inplace=True)
        return new_X_train,new_X_test
        
    def handle_imbalance(self,X_train,y_train):
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
    >>> dp = DataPreProcess()
    >>> X_train_sm, y_train_sm = dp.handle_imbalance(X_train, y_train)
    Original class distribution: Counter({0: 1000, 1: 100})
    New class distribution after SMOTE: Counter({0: 1000, 1: 1000})
    Total samples after augmentation: 2000
        """
        # oversampling the train dataset using SMOTE
        smt = SMOTE()
        X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
        # Check the new class distribution
        new_distribution = Counter(y_train_sm)

        # Print results
        print("Original class distribution:", Counter(y_train))
        print("New class distribution after SMOTE:", new_distribution)
        print("Total samples after augmentation:", len(y_train_sm))
        return X_train_sm, y_train_sm
    def apply_preprocessing(self):
        pass
