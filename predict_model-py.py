#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prediction functions for the Seoul Bike Demand models.
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def load_model(model_path):
    """
    Load a trained model from a file.
    
    Parameters
    ----------
    model_path : str
        Path to the model file
        
    Returns
    -------
    object
        Loaded model
    """
    return joblib.load(model_path)


def predict(model, X):
    """
    Make predictions using a trained model.
    
    Parameters
    ----------
    model : object
        Trained model
    X : pandas.DataFrame
        Features to make predictions on
        
    Returns
    -------
    array-like
        Predictions
    """
    return model.predict(X)


def main(input_filepath, model_filepath, output_filepath):
    """
    Run the prediction pipeline.
    
    Parameters
    ----------
    input_filepath : str
        Path to the input data
    model_filepath : str
        Path to the model file
    output_filepath : str
        Path to save the predictions
    """
    # Load the data
    if input_filepath.endswith('.csv'):
        df = pd.read_csv(input_filepath)
    else:
        raise ValueError(f"Unsupported file format: {input_filepath}")
    
    # Load the model
    model = load_model(model_filepath)
    
    # Prepare the features
    # Assuming the target column is present, we remove it for prediction
    if 'Rented Bike Count' in df.columns:
        X = df.drop('Rented Bike Count', axis=1)
        has_target = True
    else:
        X = df
        has_target = False
    
    # Make predictions
    predictions = predict(model, X)
    
    # Create a DataFrame with the predictions
    if has_target:
        result_df = pd.DataFrame({
            'Actual': df['Rented Bike Count'],
            'Predicted': predictions
        })
    else:
        result_df = pd.DataFrame({
            'Predicted': predictions
        })
    
    # Save the predictions
    result_df.to_csv(output_filepath, index=False)
    
    return result_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions with a trained model.')
    parser.add_argument('input_filepath', type=str, help='Path to the input data.')
    parser.add_argument('--model', dest='model_filepath', type=str, 
                        default='models/best_model.pkl',
                        help='Path to the model file. Default is models/best_model.pkl.')
    parser.add_argument('--output', dest='output_filepath', type=str, 
                        default='data/predictions.csv',
                        help='Path to save the predictions. Default is data/predictions.csv.')
    
    args = parser.parse_args()
    
    # Find the project root directory
    project_dir = Path(__file__).resolve().parents[2]
    
    # Convert relative paths to absolute paths
    input_filepath = os.path.join(project_dir, args.input_filepath) if not os.path.isabs(args.input_filepath) else args.input_filepath
    model_filepath = os.path.join(project_dir, args.model_filepath) if not os.path.isabs(args.model_filepath) else args.model_filepath
    output_filepath = os.path.join(project_dir, args.output_filepath) if not os.path.isabs(args.output_filepath) else args.output_filepath
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Run the prediction pipeline
    main(input_filepath, model_filepath, output_filepath)
