#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization functions for the Seoul Bike Demand project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def setup_plotting_style():
    """
    Set up the plotting style for consistent visualizations.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 20


def save_figure(fig, filename, figures_dir, dpi=300):
    """
    Save a figure to the figures directory.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Name of the file (without extension)
    figures_dir : str
        Path to the figures directory
    dpi : int, optional
        DPI for the saved figure
    """
    # Create figures directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    fig_path = os.path.join(figures_dir, f"{filename}.png")
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved to {fig_path}")


def plot_bike_rental_distribution(df, figures_dir):
    """
    Plot the distribution of bike rentals.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing bike rental data
    figures_dir : str
        Path to save the figure
    """
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(df['Rented Bike Count'], bins=50, kde=True, ax=ax)
    
    # Add vertical line for mean
    mean_value = df['Rented Bike Count'].mean()
    ax.axvline(mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.1f}')
    
    # Add labels and title
    ax.set_xlabel('Number of Rented Bikes')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Bike Rentals')
    ax.legend()
    
    # Save the figure
    save_figure(fig, 'bike_rental_distribution', figures_dir)


def plot_correlation_matrix(df, figures_dir):
    """
    Plot the correlation matrix of numerical features.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing bike rental data
    figures_dir : str
        Path to save the figure
    """
    setup_plotting_style()
    
    # Select numerical columns for correlation analysis
    numerical_cols = [
        'Rented Bike Count', 'Hour', 'Temperature(°C)', 'Humidity(%)', 
        'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)', 
        'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)'
    ]
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    # Calculate correlation matrix
    corr = df[numerical_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, 
                linewidths=0.5, linecolor='black', cbar_kws={"shrink": 0.8})
    
    # Add title
    ax.set_title('Correlation Matrix of Numerical Features')
    
    # Save the figure
    save_figure(fig, 'correlation_matrix', figures_dir)


def plot_hourly_bike_rentals(df, figures_dir):
    """
    Plot the average bike rentals by hour.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing bike rental data
    figures_dir : str
        Path to save the figure
    """
    setup_plotting_style()
    
    # Calculate average rentals by hour
    hourly_rentals = df.groupby('Hour')['Rented Bike Count'].mean().reset_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot line chart
    sns.lineplot(x='Hour', y='Rented Bike Count', data=hourly_rentals, 
                marker='o', linewidth=2, markersize=8, ax=ax)
    
    # Highlight morning and evening peaks
    morning_peak = hourly_rentals.loc[hourly_rentals['Hour'] == 8]
    evening_peak = hourly_rentals.loc[hourly_rentals['Hour'] == 18]
    
    ax.plot(morning_peak['Hour'], morning_peak['Rented Bike Count'], 'ro', 
            markersize=10, label='Morning Peak')
    ax.plot(evening_peak['Hour'], evening_peak['Rented Bike Count'], 'go', 
            markersize=10, label='Evening Peak')
    
    # Add text annotations for peaks
    ax.annotate(f"Morning Peak: {morning_peak['Rented Bike Count'].values[0]:.0f}", 
                xy=(8, morning_peak['Rented Bike Count'].values[0]), 
                xytext=(8.3, morning_peak['Rented Bike Count'].values[0] + 50),
                arrowprops=dict(arrowstyle='->'))
    
    ax.annotate(f"Evening Peak: {evening_peak['Rented Bike Count'].values[0]:.0f}", 
                xy=(18, evening_peak['Rented Bike Count'].values[0]), 
                xytext=(18.3, evening_peak['Rented Bike Count'].values[0] + 50),
                arrowprops=dict(arrowstyle='->'))
    
    # Add labels and title
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Number of Rented Bikes')
    ax.set_title('Average Bike Rentals by Hour of Day')
    ax.set_xticks(range(0, 24))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend()
    
    # Save the figure
    save_figure(fig, 'hourly_bike_rentals', figures_dir)


def plot_seasonal_bike_rentals(df, figures_dir):
    """
    Plot the average bike rentals by season.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing bike rental data
    figures_dir : str
        Path to save the figure
    """
    setup_plotting_style()
    
    # Calculate average rentals by season
    if 'Seasons' in df.columns:
        seasonal_rentals = df.groupby('Seasons')['Rented Bike Count'].mean().reset_index()
        
        # Define the order of seasons
        season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
        seasonal_rentals['Seasons'] = pd.Categorical(
            seasonal_rentals['Seasons'], categories=season_order, ordered=True
        )
        seasonal_rentals = seasonal_rentals.sort_values('Seasons')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bar chart
        bars = sns.barplot(x='Seasons', y='Rented Bike Count', data=seasonal_rentals, 
                           palette='viridis', ax=ax)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars.patches):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 20,
                f'{bar.get_height():.1f}',
                ha='center',
                fontsize=10
            )
        
        # Add labels and title
        ax.set_xlabel('Season')
        ax.set_ylabel('Average Number of Rented Bikes')
        ax.set_title('Average Bike Rentals by Season')
        
        # Save the figure
        save_figure(fig, 'seasonal_bike_rentals', figures_dir)


def plot_weather_effects(df, figures_dir):
    """
    Plot the effects of weather on bike rentals.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing bike rental data
    figures_dir : str
        Path to save the figure
    """
    setup_plotting_style()
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Temperature effect
    if 'Temperature(°C)' in df.columns:
        sns.scatterplot(x='Temperature(°C)', y='Rented Bike Count', data=df, 
                        alpha=0.5, ax=axes[0, 0])
        sns.regplot(x='Temperature(°C)', y='Rented Bike Count', data=df, 
                    scatter=False, color='red', ax=axes[0, 0])
        axes[0, 0].set_title('Bike Rentals vs Temperature')
    
    # 2. Humidity effect
    if 'Humidity(%)' in df.columns:
        sns.scatterplot(x='Humidity(%)', y='Rented Bike Count', data=df, 
                        alpha=0.5, ax=axes[0, 1])
        sns.regplot(x='Humidity(%)', y='Rented Bike Count', data=df, 
                    scatter=False, color='red', ax=axes[0, 1])
        axes[0, 1].set_title('Bike Rentals vs Humidity')
    
    # 3. Rainfall effect
    if 'Rainfall(mm)' in df.columns:
        # Create a rain category
        df['Rain_Category'] = pd.cut(
            df['Rainfall(mm)'], 
            bins=[-0.001, 0, 5, 10, float('inf')], 
            labels=['No Rain', 'Light', 'Moderate', 'Heavy']
        )
        
        rain_effect = df.groupby('Rain_Category')['Rented Bike Count'].mean().reset_index()
        
        sns.barplot(x='Rain_Category', y='Rented Bike Count', data=rain_effect, 
                    palette='Blues_r', ax=axes[1, 0])
        axes[1, 0].set_title('Bike Rentals by Rainfall Category')
    
    # 4. Snowfall effect
    if 'Snowfall (cm)' in df.columns:
        # Create a snow category
        df['Snow_Category'] = pd.cut(
            df['Snowfall (cm)'], 
            bins=[-0.001, 0, 1, 3, float('inf')], 
            labels=['No Snow', 'Light', 'Moderate', 'Heavy']
        )
        
        snow_effect = df.groupby('Snow_Category')['Rented Bike Count'].mean().reset_index()
        
        sns.barplot(x='Snow_Category', y='Rented Bike Count', data=snow_effect, 
                    palette='Blues_r', ax=axes[1, 1])
        axes[1, 1].set_title('Bike Rentals by Snowfall Category')
    
    # Add overall title
    fig.suptitle('Weather Effects on Bike Rentals', fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    save_figure(fig, 'weather_effects', figures_dir)


def plot_model_comparison(results_df, figures_dir):
    """
    Plot a comparison of model performance.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing model evaluation results
    figures_dir : str
        Path to save the figure
    """
    setup_plotting_style()
    
    # Sort models by R2 score
    results_df_sorted = results_df.sort_values('r2', ascending=True)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Bar chart of RMSE values
    ax1.barh(results_df_sorted['model_name'], results_df_sorted['rmse'], color='skyblue')
    ax1.set_xlabel('RMSE (lower is better)')
    ax1.set_title('Model RMSE Comparison')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(results_df_sorted['rmse']):
        ax1.text(v + 5, i, f'{v:.2f}', va='center')
    
    # 2. Bar chart of R² scores
    ax2.barh(results_df_sorted['model_name'], results_df_sorted['r2'], color='lightgreen')
    ax2.set_xlabel('R² Score (higher is better)')
    ax2.set_title('Model R² Comparison')
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(results_df_sorted['r2']):
        ax2.text(v - 0.12, i, f'{v:.3f}', va='center')
    
    # Add overall title
    fig.suptitle('Model Performance Comparison', fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    save_figure(fig, 'model_comparison', figures_dir)


def plot_residuals(y_true, y_pred, model_name, figures_dir):
    """
    Plot residual analysis for a model.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
    figures_dir : str
        Path to save the figure
    """
    setup_plotting_style()
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Residuals vs. predicted values
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Bike Rentals')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs. Predicted Values')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    fig.suptitle(f'Residual Analysis for {model_name}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    save_figure(fig, 'residual_analysis', figures_dir)


def plot_actual_vs_predicted(y_true, y_pred, model_name, figures_dir):
    """
    Plot actual vs. predicted values.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
    figures_dir : str
        Path to save the figure
    """
    setup_plotting_style()
    
    # Calculate performance metrics
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot actual vs predicted values
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels and title
    ax.set_xlabel('Actual Bike Rentals')
    ax.set_ylabel('Predicted Bike Rentals')
    ax.set_title(f'Actual vs Predicted Bike Rentals ({model_name})')
    
    # Add performance metrics to the plot
    ax.text(
        0.05, 0.95,
        f'RMSE: {rmse:.2f}\nR²: {r2:.3f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Save the figure
    save_figure(fig, f'actual_vs_predicted_{model_name.lower().replace(" ", "_")}', figures_dir)


def main(data_filepath, models_filepath, figures_dir):
    """
    Run the visualization pipeline.
    
    Parameters
    ----------
    data_filepath : str
        Path to the processed data directory
    models_filepath : str
        Path to the models directory
    figures_dir : str
        Path to save the figures
    """
    import pandas as pd
    import joblib
    
    # Load the processed data
    processed_data = pd.read_csv(os.path.join(data_filepath, 'seoul_bike_processed.csv'))
    test_data = pd.read_csv(os.path.join(data_filepath, 'seoul_bike_test.csv'))
    
    # Generate basic visualizations
    plot_bike_rental_distribution(processed_data, figures_dir)
    plot_correlation_matrix(processed_data, figures_dir)
    plot_hourly_bike_rentals(processed_data, figures_dir)
    plot_seasonal_bike_rentals(processed_data, figures_dir)
    plot_weather_effects(processed_data, figures_dir)
    
    # Load model comparison results if available
    model_comparison_path = os.path.join(models_filepath, 'model_comparison.csv')
    if os.path.exists(model_comparison_path):
        model_results = pd.read_csv(model_comparison_path)
        plot_model_comparison(model_results, figures_dir)
    
    # Load the best model and generate prediction visualizations
    best_model_path = os.path.join(models_filepath, 'best_model.pkl')
    if os.path.exists(best_model_path):
        best_model = joblib.load(best_model_path)
        
        # Get features and target from test data
        X_test = test_data.drop('Rented Bike Count', axis=1)
        y_test = test_data['Rented Bike Count']
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Plot residuals and actual vs predicted
        plot_residuals(y_test, y_pred, 'Best Model', figures_dir)
        plot_actual_vs_predicted(y_test, y_pred, 'Best Model', figures_dir)


if __name__ == '__main__':
    # Set up logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    # Find the project root directory
    project_dir = Path(__file__).resolve().parents[2]
    
    # Define input and output file paths
    data_filepath = os.path.join(project_dir, 'data', 'processed')
    models_filepath = os.path.join(project_dir, 'models')
    figures_dir = os.path.join(project_dir, 'reports', 'figures')
    
    main(data_filepath, models_filepath, figures_dir)