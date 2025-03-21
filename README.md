# Seoul Bike Demand Prediction

This project aims to predict hourly bike rental demand in Seoul's bike sharing system using machine learning techniques.

## Project Structure

The project follows a standard data science project structure:

```
Seoul_Bikes/
├── data/
│   ├── raw/             # Raw data
│   └── processed/       # Processed data ready for modeling
├── models/              # Trained models
├── reports/
│   └── figures/         # Generated visualisations
└── src/                 # Source code
    ├── data/            # Data processing scripts
    ├── features/        # Feature engineering scripts
    ├── models/          # Model training and evaluation scripts
    └── visualization/   # Visualisation scripts
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Required packages listed in `setup.py`

### Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -e .
```

## Usage

The project provides a command-line interface for running the entire pipeline or individual steps:

### Process Data

```bash
python run.py process-data
```

### Train Models

```bash
python run.py train
```

### Make Predictions

```bash
python run.py predict [INPUT_FILEPATH]
```

### Generate Visualizations

```bash
python run.py visualize-results
```

### Run the Entire Pipeline

```bash
python run.py run-all
```

## Data

The dataset contains hourly bike rental counts along with weather information for Seoul's bike sharing system. Features include:

- Date
- Hour
- Temperature
- Humidity
- Wind speed
- Visibility
- Dew point temperature
- Solar radiation
- Rainfall
- Snowfall
- Seasons
- Holiday
- Functioning Day

## Models

The project trains and evaluates several regression models:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Stacking Ensemble

## Results

The model performance is evaluated using metrics such as RMSE, MAE, and R². Visualizations are generated to analyze the data and model results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
