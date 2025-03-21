#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the Seoul Bike Demand Prediction pipeline.
"""

import os
import sys
import logging
import click
from pathlib import Path

# Add the src directory to the path
project_dir = Path(__file__).resolve().parents[0]
sys.path.append(os.path.join(project_dir, 'src'))

# Import project modules
from src.data.make_dataset import main as make_dataset
from src.features.build_features import build_features
from src.models.train_model import main as train_model
from src.models.predict_model import main as predict_model
from src.visualization.visualize import main as visualize


@click.group()
def cli():
    """Run the Seoul Bike Demand Prediction pipeline."""
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


@cli.command()
@click.option('--input-filepath', type=click.Path(exists=True), default='data/raw',
              help='Path to the raw data directory.')
@click.option('--output-filepath', type=click.Path(), default='data/processed',
              help='Path to save the processed data.')
def process_data(input_filepath, output_filepath):
    """Process the raw data and create the processed dataset."""
    click.echo(f"Processing data from {input_filepath} to {output_filepath}")
    
    # Convert paths to absolute paths
    input_filepath = os.path.join(project_dir, input_filepath) if not os.path.isabs(input_filepath) else input_filepath
    output_filepath = os.path.join(project_dir, output_filepath) if not os.path.isabs(output_filepath) else output_filepath
    
    # Run the data processing pipeline
    make_dataset(input_filepath, output_filepath)
    click.echo("Data processing completed.")


@cli.command()
@click.option('--input-filepath', type=click.Path(exists=True), default='data/processed',
              help='Path to the processed data directory.')
@click.option('--output-filepath', type=click.Path(), default='models',
              help='Path to save the trained models.')
def train(input_filepath, output_filepath):
    """Train models on the processed data."""
    click.echo(f"Training models using data from {input_filepath}")
    
    # Convert paths to absolute paths
    input_filepath = os.path.join(project_dir, input_filepath) if not os.path.isabs(input_filepath) else input_filepath
    output_filepath = os.path.join(project_dir, output_filepath) if not os.path.isabs(output_filepath) else output_filepath
    
    # Run the model training pipeline
    train_model(input_filepath, output_filepath)
    click.echo("Model training completed.")


@cli.command()
@click.argument('input-filepath', type=click.Path(exists=True))
@click.option('--model', type=click.Path(exists=True), default='models/best_model.pkl',
              help='Path to the model file. Default is models/best_model.pkl.')
@click.option('--output', type=click.Path(), default='data/predictions.csv',
              help='Path to save the predictions. Default is data/predictions.csv.')
def predict(input_filepath, model, output):
    """Make predictions using a trained model."""
    click.echo(f"Making predictions on {input_filepath} using model {model}")
    
    # Convert paths to absolute paths
    input_filepath = os.path.join(project_dir, input_filepath) if not os.path.isabs(input_filepath) else input_filepath
    model = os.path.join(project_dir, model) if not os.path.isabs(model) else model
    output = os.path.join(project_dir, output) if not os.path.isabs(output) else output
    
    # Run the prediction pipeline
    predict_model(input_filepath, model, output)
    click.echo(f"Predictions saved to {output}")


@cli.command()
@click.option('--data-filepath', type=click.Path(exists=True), default='data/processed',
              help='Path to the processed data directory.')
@click.option('--models-filepath', type=click.Path(exists=True), default='models',
              help='Path to the models directory.')
@click.option('--figures-dir', type=click.Path(), default='reports/figures',
              help='Path to save the figures.')
def visualize_results(data_filepath, models_filepath, figures_dir):
    """Generate visualizations from the processed data and model results."""
    click.echo("Generating visualizations")
    
    # Convert paths to absolute paths
    data_filepath = os.path.join(project_dir, data_filepath) if not os.path.isabs(data_filepath) else data_filepath
    models_filepath = os.path.join(project_dir, models_filepath) if not os.path.isabs(models_filepath) else models_filepath
    figures_dir = os.path.join(project_dir, figures_dir) if not os.path.isabs(figures_dir) else figures_dir
    
    # Run the visualization pipeline
    visualize(data_filepath, models_filepath, figures_dir)
    click.echo("Visualization completed.")


@cli.command()
def run_all():
    """Run the entire pipeline: process data, train models, and visualize results."""
    click.echo("Running the entire pipeline")
    
    # Process data
    process_data.callback(
        input_filepath='data/raw',
        output_filepath='data/processed'
    )
    
    # Train models
    train.callback(
        input_filepath='data/processed',
        output_filepath='models'
    )
    
    # Generate visualizations
    visualize_results.callback(
        data_filepath='data/processed',
        models_filepath='models',
        figures_dir='reports/figures'
    )
    
    click.echo("Pipeline completed successfully!")


if __name__ == '__main__':
    cli()