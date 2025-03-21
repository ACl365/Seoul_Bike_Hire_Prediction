#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing functions for the Seoul Bike Demand project.
"""

import pandas as pd
import numpy as np


def clean_data(df):
    """
    Clean the raw data by handling missing values, correcting data types, etc.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Raw data
        
    Returns
    -------
    pandas.DataFrame
        Cleaned data
    """
    # Create a copy to avoid modifying the original data
    df_clean = df.copy()
    
    # Convert 'Date' to datetime objects
    if df_clean['Date'].dtype == 'object':
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d/%m/%Y', errors='coerce')
    
    # If the conversion failed (NaN values present), try a different format
    if df_clean['Date'].isna().any():
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
    
    # Ensure 'Hour' is an integer
    df_clean['Hour'] = df_clean['Hour'].astype(int)
    
    # Check for missing values in the dataset
    missing_values = df_clean.isnull().sum()
    
    # If there are missing values, handle them depending on the feature
    if missing_values.sum() > 0:
        # For numerical features, impute with median
        numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # For categorical features, impute with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    return df_clean


def create_temporal_features(df):
    """
    Create temporal features from the 'Date' and 'Hour' columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data with 'Date' and 'Hour' columns
        
    Returns
    -------
    pandas.DataFrame
        Data with additional temporal features
    """
    # Create a copy to avoid modifying the original data
    df_temp = df.copy()
    
    # Extract basic calendar features
    df_temp['Year'] = df_temp['Date'].dt.year
    df_temp['Month'] = df_temp['Date'].dt.month
    df_temp['Day'] = df_temp['Date'].dt.day
    df_temp['Day_of_Week'] = df_temp['Date'].dt.day_name()
    df_temp['Is_Weekend'] = df_temp['Date'].dt.dayofweek >= 5
    
    # Create cyclical features for time variables
    df_temp['Hour_sin'] = np.sin(2 * np.pi * df_temp['Hour']/24)
    df_temp['Hour_cos'] = np.cos(2 * np.pi * df_temp['Hour']/24)
    df_temp['Month_sin'] = np.sin(2 * np.pi * df_temp['Month']/12)
    df_temp['Month_cos'] = np.cos(2 * np.pi * df_temp['Month']/12)
    df_temp['Day_sin'] = np.sin(2 * np.pi * df_temp['Day']/31)
    df_temp['Day_cos'] = np.cos(2 * np.pi * df_temp['Day']/31)
    
    return df_temp


def encode_categorical_features(df):
    """
    Encode categorical features using one-hot encoding.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data with categorical features
        
    Returns
    -------
    pandas.DataFrame
        Data with encoded categorical features
    """
    # Create a copy to avoid modifying the original data
    df_encoded = df.copy()
    
    # One-hot encode categorical features
    categorical_cols = ['Seasons', 'Holiday', 'Functioning Day', 'Day_of_Week']
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col)
    
    return df_encoded


def preprocess_input(input_data):
    """
    Preprocess a single input record for prediction.
    
    Parameters
    ----------
    input_data : dict
        Dictionary containing input features
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed input data ready for model prediction
    """
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Convert datetime string to datetime object
    if 'datetime' in df.columns:
        df['Date'] = pd.to_datetime(df['datetime'])
        df['Hour'] = df['Date'].dt.hour
        df = df.drop('datetime', axis=1)
    
    # Clean the data
    df = clean_data(df)
    
    # Create temporal features
    df = create_temporal_features(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Drop Date column for prediction
    if 'Date' in df.columns:
        df = df.drop('Date', axis=1)
    
    # Ensure all expected columns are present
    # In a real implementation, we would load the expected columns from a config file
    # or a saved list of columns from the training data
    
    return df