# AI-Powered Climate Risk Analysis

## Overview

Ireland faces a growing need to address climate change and its impacts. This project uses machine learning to analyze historical climate data and predict extreme weather events. By combining data preprocessing, time-series forecasting, and visualization, the project demonstrates practical AI/ML skills in a highly relevant domain.

## Features

- **Data Preprocessing:** Clean and transform raw climate data.
- **Time-Series Forecasting:** Build an LSTM-based model to forecast key climate indicators and detect extreme weather events.
- **Visualization:** Plot historical data, forecast results, and highlight potential extreme events.
- **Extendable Architecture:** Easily update with more parameters (e.g., wind speed, humidity) or alternative forecasting methods.

## Repository Structure

```
climate-risk-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── climate_clean.csv
│   └── climate_data.csv
├── output/
│   ├── climate_model.h5
│   ├── scaler.pkl
│   └── plots/
│       ├── temperature_forecast.png
│       └── training_loss.png
└── scripts/
    ├── forecast.py
    ├── preprocess.py
    └── train_model.py
```
