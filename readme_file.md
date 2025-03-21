# Seoul Bike Demand Prediction

![Seoul Bikes](https://images.unsplash.com/photo-1529339276202-354ef6f58097?q=80&w=1000&auto=format&fit=crop)

## Project Overview

This project develops and evaluates machine learning models to predict hourly bike rental demand in Seoul's bike-sharing system using weather and temporal data. Accurate demand forecasting is critical for operational efficiency, resource allocation, and enhanced user experience in urban mobility systems.

### Key Achievements
- **Superior Predictive Performance**: Developed a stacking ensemble model achieving 95% variance explanation (R² = 0.950)
- **Comprehensive Model Comparison**: Systematically evaluated 7 different algorithms to identify optimal approach
- **Actionable Insights**: Translated technical findings into practical operational recommendations
- **Weather Impact Quantification**: Measured precise effects of temperature, precipitation and seasonality on bike demand

## Data Description

The dataset contains hourly bike rental records from Seoul's bike-sharing system spanning December 2017 to November 2018 (8,760 observations), including:

- **Target variable**: Rented Bike Count
- **Temporal features**: Date, Hour
- **Meteorological variables**: Temperature, Humidity, Wind speed, Visibility, Dew point temperature, Solar Radiation, Rainfall, Snowfall
- **Categorical features**: Seasons, Holiday status, Functioning Day status

## Repository Structure

```
├── README.md                  # Project overview and documentation
├── data/                      # Data directory
│   ├── raw/                   # Original, immutable data
│   ├── processed/             # Cleaned, transformed data ready for modeling
│   └── README.md              # Data documentation
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # Initial EDA and insights
│   ├── 02_feature_engineering.ipynb # Feature creation and transformation
│   └── 03_modeling.ipynb            # Model development and evaluation
├── src/                       # Source code
│   ├── data/                  # Code for data processing
│   │   ├── __init__.py
│   │   ├── make_dataset.py    # Script to generate processed dataset
│   │   └── preprocess.py      # Data cleaning and preprocessing functions
│   ├── features/              # Code for feature engineering
│   │   ├── __init__.py
│   │   └── build_features.py  # Feature creation functions
│   ├── models/                # Code for model training and evaluation
│   │   ├── __init__.py
│   │   ├── train_model.py     # Model training
│   │   ├── predict_model.py   # Prediction using trained model
│   │   └── evaluate_model.py  # Model evaluation functions
│   └── visualization/         # Code for creating figures and plots
│       ├── __init__.py
│       └── visualize.py       # Visualization functions
├── reports/                   # Project reports and findings
│   ├── figures/               # Generated graphics and figures
│   └── seoul_bike_demand_analysis.md  # Final analysis report
├── models/                    # Saved model files and model metadata
│   └── model_comparison.csv   # Performance metrics for all models
├── requirements.txt           # Dependencies
├── setup.py                   # Make project pip installable
└── LICENSE                    # Project license
```

## Key Findings

![Model Performance Comparison](reports/figures/model_comparison.png)

### Model Performance Summary

| Model               | RMSE    | MAE    | R²     | MSE as % of Variance | CV RMSE (Mean) |
|---------------------|---------|--------|--------|----------------------|----------------|
| Linear Regression   | 428.42  | 323.19 | 0.559  | 44.05%               | 412.50         |
| Ridge Regression    | 428.32  | 323.06 | 0.560  | 44.03%               | 412.49         |
| Gradient Boosting   | 235.46  | 155.58 | 0.867  | 13.31%               | 228.55         |
| Random Forest       | 169.51  | 94.03  | 0.931  | 6.90%                | 168.64         |
| XGBoost             | 150.76  | 88.38  | 0.945  | 5.46%                | 152.47         |
| LightGBM            | 149.16  | 87.56  | 0.947  | 5.34%                | 149.43         |
| Stacking Ensemble   | 144.75  | 79.70  | 0.950  | 5.03%                | 143.64         |

### Feature Importance

Temperature (29.1%) and hour of day (22.1%) emerged as the most influential predictors, followed by solar radiation (6.9%), humidity (5.7%), and cyclic hour encoding (5.5%).

### Temporal Patterns

- **Peak demand hours**: 18:00, 19:00, 17:00
- **Seasonal variation**: Summer rentals (avg. 1,034/hr) nearly 5× higher than winter (avg. 226/hr)
- **Day type impact**: Weekday rentals 7.8% higher than weekends

### Weather Effects

- Temperature correlation: +0.54
- Rainfall impact: 78% reduction during precipitation
- Humidity correlation: -0.20

## Operational Applications

The model enables several practical applications:

1. **Dynamic Fleet Management**: Optimise bike distribution based on predicted demand
2. **Demand-Based Pricing**: Implement surge pricing during peak hours
3. **Maintenance Scheduling**: Plan maintenance during predicted low-demand periods
4. **Infrastructure Planning**: Guide station expansion decisions
5. **User Experience Enhancement**: Provide availability forecasts to users

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/seoul-bike-demand.git
cd seoul-bike-demand

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the project in development mode
pip install -e .
```

## Usage Example

```python
from src.models.predict_model import load_model, predict
from src.data.preprocess import preprocess_input

# Load the trained model
model = load_model('models/stacking_ensemble.pkl')

# Example input data
input_data = {
    'datetime': '2023-06-15 18:00:00',
    'temperature': 25.5,
    'humidity': 60,
    'wind_speed': 1.5,
    'visibility': 2000,
    'dew_point': 15.2,
    'solar_radiation': 0.5,
    'rainfall': 0,
    'snowfall': 0,
    'season': 'Summer',
    'holiday': 'No Holiday',
    'functioning_day': 'Yes'
}

# Preprocess the input
processed_input = preprocess_input(input_data)

# Make prediction
predicted_demand = predict(model, processed_input)
print(f"Predicted bike demand: {predicted_demand} bikes")
```

## Tech Stack

- **Python**: Core programming language
- **pandas & NumPy**: Data manipulation and numerical operations
- **scikit-learn**: Model training, evaluation and preprocessing
- **XGBoost & LightGBM**: Gradient boosting frameworks
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebooks**: Interactive development and documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/seoul-bike-demand](https://github.com/yourusername/seoul-bike-demand)

## Acknowledgments

- Seoul Bike Sharing Dataset: [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/seoul-bike-sharing/data)
