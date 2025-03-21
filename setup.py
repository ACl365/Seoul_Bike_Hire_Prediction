from setuptools import find_packages, setup

setup(
    name='seoul_bike_demand',
    packages=find_packages(),
    version='0.1.0',
    description='Prediction of hourly bike rental demand in Seoul bike sharing system',
    author='Your Name',
    license='MIT',
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'xgboost>=1.7.0',
        'lightgbm>=4.0.0',
        'plotly>=5.15.0',
        'shap>=0.42.0',
        'joblib>=1.3.0',
        'click>=8.0.0',
    ],
    python_requires='>=3.9',
)
