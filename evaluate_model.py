#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation functions for the Seoul Bike Demand models.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a model, returning performance metrics.
    
    Parameters
    ----------
    model : sklearn estimator
        Model to train and evaluate
    model_name : str
        Name of the model for reporting
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Test features
    y_train : pandas.Series
        Training target
    y_test : pandas.Series
        Test target
        
    Returns
    -------
    dict
        Dictionary containing performance metrics
    """
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    variance = np.var(y_test)
    mse_percentage_of_variance = (mse / variance) * 100
    
    # Cross-validation RMSE
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    cv_rmse_mean = cv_rmse.mean()
    cv_rmse_std = cv_rmse.std()
    
    # Print evaluation metrics
    print(f"Model: {model_name}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.3f}")
    print(f"MSE as percentage of variance: {mse_percentage_of_variance:.2f}%")
    print(f"Cross-Validation RMSE (Mean): {cv_rmse_mean:.2f}")
    print(f"Cross-Validation RMSE (Std): {cv_rmse_std:.2f}")
    print("-" * 30)
    
    # Return evaluation metrics
    return {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse_percentage_of_variance': mse_percentage_of_variance,
        'cv_rmse_mean': cv_rmse_mean,
        'cv_rmse_std': cv_rmse_std,
        'model': model,
        'predictions': y_pred
    }


def calculate_feature_importance(model, feature_names):
    """
    Calculate feature importance for tree-based models.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model
    feature_names : list
        List of feature names
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing feature names and their importance scores
    """
    import pandas as pd
    
    # Check if the model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame of feature importances
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return feature_importance
    else:
        return None


def plot_residuals(y_true, y_pred, title='Residual Plot'):
    """
    Plot residuals of a model.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    title : str, optional
        Title of the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the residual plot
    """
    import matplotlib.pyplot as plt
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot residuals vs predicted values
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    
    # Plot histogram of residuals
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    
    # Add overall title
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """
    Plot actual vs predicted values.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the actual vs predicted plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot actual vs predicted values
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels and title
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Actual vs Predicted ({model_name})')
    
    # Add performance metrics to the plot
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    ax.text(
        0.05, 0.95, 
        f'RMSE: {rmse:.2f}\nR²: {r2:.3f}', 
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    return fig
