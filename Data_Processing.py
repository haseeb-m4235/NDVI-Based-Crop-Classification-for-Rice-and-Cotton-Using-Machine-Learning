import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    