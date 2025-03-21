#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering functions for the Seoul Bike Demand project.
This module contains functions to create advanced features from the basic dataset.
"""

import numpy as np
import pandas as pd


def create_temporal_features(df, datetime_col='Date'):
    """
    Create temporal features from datetime column.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the datetime column
    datetime_col : str, optional
        Name of the datetime column
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with temporal features added
    """
    # Create a copy to avoid modifying the original data
    df_temp = df.copy()
    
    # Ensure the datetime column is a datetime type
    if datetime_col in df_temp.columns and not pd.api.types.is_datetime64_any_dtype(df_temp[datetime_col]):
        df_temp[datetime_col] = pd.to_datetime(df_temp[datetime_col])
    
    # Extract basic calendar features
    if datetime_col in df_temp.columns:
        df_temp['Year'] = df_temp[datetime_col].dt.year
        df_temp['Month'] = df_temp[datetime_col].dt.month
        df_temp['Day'] = df_temp[datetime_col].dt.day
        df_temp['Day_of_Week'] = df_temp[datetime_col].dt.day_name()
        df_temp['Day_of_Week_num'] = df_temp[datetime_col].dt.dayofweek
        df_temp['Is_Weekend'] = (df_temp[datetime_col].dt.dayofweek >= 5).astype(int)
        df_temp['Day_of_Year'] = df_temp[datetime_col].dt.dayofyear
        df_temp['Week_of_Year'] = df_temp[datetime_col].dt.isocalendar().week
        df_temp['Quarter'] = df_temp[datetime_col].dt.quarter
        
        # Create cyclical features for time variables
        if 'Hour' in df_temp.columns:
            df_temp['Hour_sin'] = np.sin(2 * np.pi * df_temp['Hour']/24)
            df_temp['Hour_cos'] = np.cos(2 * np.pi * df_temp['Hour']/24)
        
        df_temp['Month_sin'] = np.sin(2 * np.pi * df_temp['Month']/12)
        df_temp['Month_cos'] = np.cos(2 * np.pi * df_temp['Month']/12)
        df_temp['Day_sin'] = np.sin(2 * np.pi * df_temp['Day']/31)
        df_temp['Day_cos'] = np.cos(2 * np.pi * df_temp['Day']/31)
        df_temp['Day_of_Week_sin'] = np.sin(2 * np.pi * df_temp['Day_of_Week_num']/7)
        df_temp['Day_of_Week_cos'] = np.cos(2 * np.pi * df_temp['Day_of_Week_num']/7)
        
        # Create features for special times of day
        if 'Hour' in df_temp.columns:
            df_temp['Is_Morning_Rush'] = ((df_temp['Hour'] >= 7) & (df_temp['Hour'] <= 9)).astype(int)
            df_temp['Is_Evening_Rush'] = ((df_temp['Hour'] >= 17) & (df_temp['Hour'] <= 19)).astype(int)
            df_temp['Is_Night'] = ((df_temp['Hour'] >= 22) | (df_temp['Hour'] <= 5)).astype(int)
            df_temp['Is_Working_Hour'] = ((df_temp['Hour'] >= 9) & (df_temp['Hour'] <= 17)).astype(int)
    
    return df_temp


def create_weather_features(df):
    """
    Create weather-related features.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing weather features
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with weather features added
    """
    # Create a copy to avoid modifying the original data
    df_weather = df.copy()
    
    # Temperature and humidity interaction (heat index proxy)
    if all(col in df_weather.columns for col in ['Temperature(°C)', 'Humidity(%)']):
        df_weather['Temp_Humidity_Interaction'] = df_weather['Temperature(°C)'] * df_weather['Humidity(%)']
    
    # Create feels-like temperature (simplified version)
    if all(col in df_weather.columns for col in ['Temperature(°C)', 'Wind speed (m/s)', 'Humidity(%)']):
        # Simple approximation of feels-like temperature considering wind chill and humidity
        df_weather['Feels_Like_Temp'] = df_weather['Temperature(°C)'] - (
            0.3 * df_weather['Wind speed (m/s)'] + 
            0.1 * (df_weather['Humidity(%)'] - 65) * np.where(df_weather['Temperature(°C)'] > 20, 1, 0)
        )
    
    # Weather severity index
    if all(col in df_weather.columns for col in ['Rainfall(mm)', 'Snowfall (cm)']):
        df_weather['Weather_Severity'] = df_weather['Rainfall(mm)'] + (5 * df_weather['Snowfall (cm)'])
    
    # Weather categories
    if 'Rainfall(mm)' in df_weather.columns:
        df_weather['Is_Rainy'] = (df_weather['Rainfall(mm)'] > 0).astype(int)
    if 'Snowfall (cm)' in df_weather.columns:
        df_weather['Is_Snowy'] = (df_weather['Snowfall (cm)'] > 0).astype(int)
    
    # Temperature bins for categorical analysis
    if 'Temperature(°C)' in df_weather.columns:
        df_weather['Temp_Category'] = pd.cut(
            df_weather['Temperature(°C)'], 
            bins=[-20, 0, 10, 20, 30, 40], 
            labels=['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot']
        )
    
    # Visibility categories
    if 'Visibility (10m)' in df_weather.columns:
        df_weather['Visibility_Category'] = pd.cut(
            df_weather['Visibility (10m)'], 
            bins=[0, 500, 1000, 2000, float('inf')], 
            labels=['Very Low', 'Low', 'Moderate', 'Good']
        )
    
    return df_weather


def create_interaction_features(df):
    """
    Create interaction features between different variables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the features
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with interaction features added
    """
    # Create a copy to avoid modifying the original data
    df_interact = df.copy()
    
    # Create interaction between hour and weekday/weekend
    if 'Is_Weekend' in df_interact.columns and 'Hour' in df_interact.columns:
        df_interact['Hour_Weekend_Interaction'] = df_interact['Hour'] * df_interact['Is_Weekend']
    
    # Create interaction between hour and temperature
    if 'Hour' in df_interact.columns and 'Temperature(°C)' in df_interact.columns:
        df_interact['Hour_Temp_Interaction'] = df_interact['Hour'] * df_interact['Temperature(°C)']
    
    # Create interaction between hour and weather features
    if 'Hour' in df_interact.columns:
        if 'Is_Rainy' in df_interact.columns:
            df_interact['Hour_Rain_Interaction'] = df_interact['Hour'] * df_interact['Is_Rainy']
        if 'Weather_Severity' in df_interact.columns:
            df_interact['Hour_Weather_Severity'] = df_interact['Hour'] * df_interact['Weather_Severity']
    
    # Create interaction between season and temperature
    if 'Seasons' in df_interact.columns and 'Temperature(°C)' in df_interact.columns:
        # Create dummy variables for seasons
        seasons_dummies = pd.get_dummies(df_interact['Seasons'], prefix='Season')
        # Create interactions with temperature
        for season in seasons_dummies.columns:
            df_interact[f'{season}_Temp_Interaction'] = seasons_dummies[season] * df_interact['Temperature(°C)']
    
    return df_interact


def encode_categorical_features(df):
    """
    One-hot encode categorical features.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing categorical features
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with one-hot encoded categorical features
    """
    # Create a copy to avoid modifying the original data
    df_encoded = df.copy()
    
    # Define categorical features
    categorical_cols = [
        'Seasons', 'Holiday', 'Functioning Day', 'Day_of_Week', 
        'Temp_Category', 'Visibility_Category'
    ]
    
    # Filter out columns that don't exist in the dataframe
    categorical_cols = [col for col in categorical_cols if col in df_encoded.columns]
    
    # Apply one-hot encoding
    for col in categorical_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col)
    
    return df_encoded


def build_features(df):
    """
    Apply all feature engineering steps to the input dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with all engineered features
    """
    # Apply feature engineering steps in sequence
    df_features = create_temporal_features(df)
    df_features = create_weather_features(df_features)
    df_features = create_interaction_features(df_features)
    df_features = encode_categorical_features(df_features)
    
    return df_features


if __name__ == "__main__":
    # Example usage
    import os
    import sys
    from pathlib import Path
    
    # Find the project root directory
    project_dir = Path(__file__).resolve().parents[2]
    
    # Add the src directory to the path
    sys.path.append(os.path.join(project_dir, 'src'))
    
    # Define input and output file paths
    input_file = os.path.join(project_dir, 'data', 'raw', 'SeoulBikeData.csv')
    output_file = os.path.join(project_dir, 'data', 'processed', 'seoul_bike_features.csv')
    
    # Load the data
    df = pd.read_csv(input_file, encoding='unicode_escape')
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    
    # Build features
    df_features = build_features(df)
    
    # Save the result
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_features.to_csv(output_file, index=False)
    
    print(f"Features built and saved to {output_file}")
    print(f"Original shape: {df.shape}, Features shape: {df_features.shape}")
