# Model Directory

This directory contains all trained models and model evaluation results for the Seoul Bike Demand project.

## Directory Structure

```
models/
├── best_model.pkl                 # Copy of the best performing model
├── linear_regression.pkl          # Linear Regression model
├── ridge_regression.pkl           # Ridge Regression model
├── gradient_boosting.pkl          # Gradient Boosting Regressor model
├── random_forest.pkl              # Random Forest Regressor model
├── xgboost.pkl                    # XGBoost Regressor model
├── lightgbm.pkl                   # LightGBM Regressor model
├── stacking_ensemble.pkl          # Stacking Ensemble model
├── model_comparison.csv           # CSV file containing model evaluation metrics
└── model_summary.json             # JSON file with summary of the best model performance
```

## Model Descriptions

### Linear Models
- **Linear Regression**: Basic linear regression model.
- **Ridge Regression**: Linear regression with L2 regularisation.

### Tree-Based Models
- **Random Forest**: Ensemble of decision trees trained via bagging.
- **Gradient Boosting**: Sequential ensemble of weak learners.
- **XGBoost**: Optimised gradient boosting implementation.
- **LightGBM**: Gradient boosting framework that uses tree-based learning algorithms.

### Ensemble Model
- **Stacking Ensemble**: Meta-model that combines predictions from multiple base models.

## Model Performance

The table below shows the performance metrics for all models on the test set:

| Model               | RMSE    | MAE    | R²     | MSE as % of Variance | CV RMSE (Mean) |
|---------------------|---------|--------|--------|----------------------|----------------|
| Linear Regression   | 428.42  | 323.19 | 0.559  | 44.05%               | 412.50         |
| Ridge Regression    | 428.32  | 323.06 | 0.560  | 44.03%               | 412.49         |
| Gradient Boosting   | 235.46  | 155.58 | 0.867  | 13.31%               | 228.55         |
| Random Forest       | 169.51  | 94.03  | 0.931  | 6.90%                | 168.64         |
| XGBoost             | 150.76  | 88.38  | 0.945  | 5.46%                | 152.47         |
| LightGBM            | 149.16  | 87.56  | 0.947  | 5.34%                | 149.43         |
| Stacking Ensemble   | 144.75  | 79.70  | 0.950  | 5.03%                | 143.64         |

## Best Model

The best performing model is the **Stacking Ensemble** with:
- R² score of 0.950
- RMSE of 144.75
- MAE of 79.70

This model combines the strengths of multiple base models (Ridge, Gradient Boosting, Random Forest, XGBoost, and LightGBM) with a Histogram Gradient Boosting Regressor as the meta-learner.

## Loading and Using Models

To load and use a saved model:

```python
import joblib

# Load the model
model = joblib.load('models/best_model.pkl')

# Make predictions
predictions = model.predict(X_test)
```

## Model Summary

The `model_summary.json` file contains a summary of the best model's performance and key findings, including peak hours for bike rentals, temperature correlation, and rainfall effect.
