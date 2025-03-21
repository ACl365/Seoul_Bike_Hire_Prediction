#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate processed datasets from raw data.
This script handles all data preprocessing steps for the Seoul Bike Demand project.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the src directory to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import project modules
from src.data.preprocess import clean_data, create_temporal_features, encode_categorical_features


def main(input_filepath, output_filepath):
    """
    Run the data processing pipeline to turn raw data into processed data ready for analysis.
    
    Parameters
    ----------
    input_filepath : str
        Path to the raw data directory
    output_filepath : str
        Path to the processed data directory
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_filepath, exist_ok=True)
    
    # Load the raw data
    logger.info(f'Loading raw data from {input_filepath}')
    input_file = os.path.join(input_filepath, 'SeoulBikeData.csv')
    df = pd.read_csv(input_file, encoding='unicode_escape')
    
    # Clean the data
    logger.info('Cleaning data')
    df = clean_data(df)
    
    # Create temporal features
    logger.info('Creating temporal features')
    df = create_temporal_features(df)
    
    # Encode categorical features
    logger.info('Encoding categorical features')
    df = encode_categorical_features(df)
    
    # Define target variable and features
    target = 'Rented Bike Count'
    
    # Define numerical features
    numerical_features = [
        'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 
        'Visibility (10m)', 'Dew point temperature(°C)', 
        'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)'
    ]
    
    # Scale numerical features
    logger.info('Scaling numerical features')
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Save the scaler for later use in predictions
    logger.info('Saving scaler')
    import joblib
    joblib.dump(scaler, os.path.join(output_filepath, 'scaler.pkl'))
    
    # Save the processed data
    logger.info('Saving processed data')
    df.to_csv(os.path.join(output_filepath, 'seoul_bike_processed.csv'), index=False)
    
    # Split the data into training and test sets
    logger.info('Splitting data into training and test sets')
    X = df.drop([target, 'Date'], axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save the training and test sets
    logger.info('Saving training and test sets')
    train_df = pd.concat([y_train, X_train], axis=1)
    test_df = pd.concat([y_test, X_test], axis=1)
    
    train_df.to_csv(os.path.join(output_filepath, 'seoul_bike_train.csv'), index=False)
    test_df.to_csv(os.path.join(output_filepath, 'seoul_bike_test.csv'), index=False)
    
    logger.info('Data processing completed')


if __name__ == '__main__':
    # Set up logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    # Find the project root directory
    project_dir = Path(__file__).resolve().parents[2]
    
    # Define input and output file paths
    input_filepath = os.path.join(project_dir, 'data', 'raw')
    output_filepath = os.path.join(project_dir, 'data', 'processed')
    
    main(input_filepath, output_filepath)
