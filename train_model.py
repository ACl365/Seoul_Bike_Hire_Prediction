#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train models for the Seoul Bike Demand project.
This script trains and evaluates various regression models for bike demand prediction.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor, 
    RandomForestRegressor, 
    StackingRegressor, 
    HistGradientBoostingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add the src directory to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import project modules
from src.models.evaluate_model import evaluate_model


def train_models(X_train, y_train, X_test, y_test, output_filepath):
    """
    Train and evaluate multiple regression models, saving the best one.
    
    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    output_filepath : str
        Path to save trained models
        
    Returns
    -------
    dict
        Results of model evaluations
    """
    logger = logging.getLogger(__name__)
    logger.info('Training models')
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    
    # Add XGBoost and LightGBM if they are available
    try:
        models['XGBoost'] = XGBRegressor(random_state=42)
    except:
        logger.warning("XGBoost not available, skipping")
    
    try:
        models['LightGBM'] = LGBMRegressor(verbose=-1, random_state=42)
    except:
        logger.warning("LightGBM not available, skipping")
    
    # Define base models for stacking
    estimators = [
        ('ridge', Ridge(alpha=1.0)),
        ('gbr', GradientBoostingRegressor(random_state=42)),
        ('rf', RandomForestRegressor(random_state=42))
    ]
    
    # Add XGBoost and LightGBM to estimators if available
    if 'XGBoost' in models:
        estimators.append(('xgb', XGBRegressor(random_state=42)))
    if 'LightGBM' in models:
        estimators.append(('lgbm', LGBMRegressor(verbose=-1, random_state=42)))
    
    # Add Stacking Ensemble model
    models['Stacking Ensemble'] = StackingRegressor(
        estimators=estimators, 
        final_estimator=HistGradientBoostingRegressor(random_state=42)
    )
    
    # Evaluate all models
    results = []
    for name, model in models.items():
        try:
            logger.info(f'Training and evaluating {name}')
            result = evaluate_model(model, name, X_train, X_test, y_train, y_test)
            results.append(result)
            
            # Save the trained model
            joblib.dump(model, os.path.join(output_filepath, f'{name.lower().replace(" ", "_")}.pkl'))
            
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Save the results
    results_df.to_csv(os.path.join(output_filepath, 'model_comparison.csv'), index=False)
    
    # Identify the best model
    best_model_idx = results_df['r2'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'model_name']
    
    logger.info(f"Best performing model: {best_model_name}")
    
    # Make a copy of the best model as 'best_model.pkl'
    best_model_path = os.path.join(output_filepath, f'{best_model_name.lower().replace(" ", "_")}.pkl')
    joblib.dump(joblib.load(best_model_path), os.path.join(output_filepath, 'best_model.pkl'))
    
    return results


def main(input_filepath, output_filepath):
    """
    Run the model training pipeline.
    
    Parameters
    ----------
    input_filepath : str
        Path to the processed data directory
    output_filepath : str
        Path to save trained models
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting model training pipeline')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_filepath, exist_ok=True)
    
    # Load the processed training and test data
    logger.info(f'Loading processed data from {input_filepath}')
    train_file = os.path.join(input_filepath, 'seoul_bike_train.csv')
    test_file = os.path.join(input_filepath, 'seoul_bike_test.csv')
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Define target variable
    target = 'Rented Bike Count'
    
    # Split into features and target
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]
    
    # Train and evaluate models
    logger.info('Training and evaluating models')
    results = train_models(X_train, y_train, X_test, y_test, output_filepath)
    
    logger.info('Model training completed')


if __name__ == '__main__':
    # Set up logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    # Find the project root directory
    project_dir = Path(__file__).resolve().parents[2]
    
    # Define input and output file paths
    input_filepath = os.path.join(project_dir, 'data', 'processed')
    output_filepath = os.path.join(project_dir, 'models')
    
    main(input_filepath, output_filepath)