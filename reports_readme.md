# Reports Directory

This directory contains reports and visualisations generated from the Seoul Bike Demand project analysis.

## Directory Structure

```
reports/
├── figures/                          # Directory containing all visualisations
│   ├── actual_vs_predicted.png       # Comparison of actual vs. predicted values
│   ├── bike_rental_distribution.png  # Distribution of bike rentals
│   ├── correlation_matrix.png        # Correlation matrix of features
│   ├── error_by_hour.png             # Analysis of prediction errors by hour
│   ├── feature_importance.png        # Feature importance from the best model
│   ├── hourly_bike_rentals.png       # Average bike rentals by hour
│   ├── model_comparison.png          # Comparison of model performance
│   ├── operational_dashboard.png     # Dashboard for operational insights
│   ├── residual_analysis.png         # Analysis of model residuals
│   ├── seasonal_bike_rentals.png     # Average bike rentals by season
│   └── weather_effects.png           # Impact of weather conditions on rentals
└── seoul_bike_demand_analysis.md     # Comprehensive analysis report
```

## Key Visualisations

### Exploratory Analysis

- **Bike Rental Distribution**: Shows the distribution of hourly bike rentals across the dataset.
- **Correlation Matrix**: Visualises the correlation between different features and the target variable.
- **Hourly Bike Rentals**: Displays the average bike rentals by hour of day, highlighting morning and evening peaks.
- **Seasonal Bike Rentals**: Shows how bike rentals vary across seasons.
- **Weather Effects**: Illustrates the impact of temperature, humidity, rainfall, and snowfall on bike rentals.

### Model Evaluation

- **Model Comparison**: Compares the performance of different models using RMSE and R² metrics.
- **Actual vs. Predicted**: Scatter plot of actual vs. predicted bike rental counts.
- **Residual Analysis**: Examines the distribution of residuals and their relationship with predicted values.
- **Error by Hour**: Analyzes how prediction errors vary by hour of day.
- **Feature Importance**: Ranks features by their importance in the best performing model.

### Operational Insights

- **Operational Dashboard**: A comprehensive dashboard providing key insights for bike-sharing operation management.

## Main Report

The main report (`seoul_bike_demand_analysis.md`) provides a comprehensive analysis of the Seoul bike rental data, including:

1. Introduction and problem statement
2. Data description and preprocessing steps
3. Exploratory data analysis findings
4. Feature engineering approach
5. Model development and evaluation
6. Operational insights and recommendations
7. Conclusions and future work

## Generating Reports

To regenerate all visualisations and reports, run:

```bash
python src/visualization/visualize.py
```

This will create all figures in the `reports/figures` directory.
