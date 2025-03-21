# Data Directory

This directory contains all data files used in the Seoul Bike Demand project.

## Directory Structure

```
data/
├── raw/                    # Original, immutable data
│   └── SeoulBikeData.csv   # Original Seoul bike sharing dataset
├── processed/              # Cleaned, transformed data ready for modeling
│   ├── seoul_bike_processed.csv          # Preprocessed dataset with feature engineering
│   ├── seoul_bike_train.csv              # Training dataset (80%)
│   └── seoul_bike_test.csv               # Test dataset (20%)
└── external/               # External data sources that supplement the analysis
    └── seoul_weather_stations.csv        # Additional weather station data (if applicable)
```

## Dataset Description

The primary dataset (`SeoulBikeData.csv`) contains hourly bike rental records from Seoul's bike-sharing system spanning December 2017 to November 2018, with the following features:

### Target Variable
- **Rented Bike Count**: Number of bikes rented per hour

### Features

#### Temporal Features
- **Date**: Date in DD/MM/YYYY format
- **Hour**: Hour of the day (0-23)

#### Meteorological Variables
- **Temperature(°C)**: Temperature in Celsius
- **Humidity(%)**: Humidity percentage
- **Wind speed (m/s)**: Wind speed in meters per second
- **Visibility (10m)**: Visibility in 10m units
- **Dew point temperature(°C)**: Dew point temperature in Celsius
- **Solar Radiation (MJ/m2)**: Solar radiation in MJ/m²
- **Rainfall(mm)**: Rainfall in millimetres
- **Snowfall (cm)**: Snowfall in centimetres

#### Categorical Features
- **Seasons**: Season of the year (Winter, Spring, Summer, Autumn)
- **Holiday**: Whether it is a holiday or not (Holiday, No Holiday)
- **Functioning Day**: Whether it is a functioning day or not (Yes, No)

## Dataset Statistics

- **Number of rows**: 8,760 (365 days × 24 hours)
- **Number of columns**: 14
- **Time period**: 1st December 2017 to 30th November 2018
- **Missing values**: None

### Summary Statistics for Target Variable

| Statistic | Value |
|-----------|-------|
| Mean      | 704.60 |
| Std Dev   | 644.99 |
| Min       | 0      |
| 25%       | 191.00 |
| 50%       | 504.50 |
| 75%       | 1065.25 |
| Max       | 3556.00 |

## Processed Data

The processed data includes several engineered features:

- **Cyclical temporal features**: Hour_sin, Hour_cos, Month_sin, Month_cos, Day_sin, Day_cos
- **Calendar features**: Year, Month, Day, Day_of_Week, Is_Weekend
- **One-hot encoded categorical features**: Seasons, Holiday, Functioning Day, Day_of_Week

## Data Source

The original dataset was sourced from the Seoul city bike rental system:
- **Kaggle link**: [Seoul Bike Sharing Demand Dataset](https://www.kaggle.com/datasets/joebeachcapital/seoul-bike-sharing/data)
- **Original source**: Seoul Metropolitan Government

## Data Processing Steps

The `make_dataset.py` script performs the following operations:

1. Loads the raw data
2. Converts date strings to datetime objects
3. Extracts calendar features
4. Creates cyclical encodings for periodic features
5. One-hot encodes categorical variables
6. Standardises numerical features
7. Splits data into training and test sets
8. Saves the processed datasets

## Usage

To regenerate the processed datasets:

```bash
# From the project root directory
python src/data/make_dataset.py
```
