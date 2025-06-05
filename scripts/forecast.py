import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

LOOKBACK = 30  # Must match training
EXTREME_THRESHOLD = 0.9  # Normalized threshold for extremes, adapt as needed

def create_dataset(data, lookback):
    X = []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
    return np.array(X)

def main():
    # Load cleaned data (using the most recent portion of the data)
    df = pd.read_csv("data/climate_clean.csv", parse_dates=['date'])
    values = df['temperature'].values.reshape(-1, 1)
    
    # Load scaler and model
    scaler = pickle.load(open("output/scaler.pkl", "rb"))
    model = load_model("output/climate_model.h5")
    
    scaled_values = scaler.transform(values)
    X = create_dataset(scaled_values, LOOKBACK)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Forecast on the available dataset (or you can use new incoming data)
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    # Append predictions to df (trimming the first LOOKBACK dates as they have no prediction)
    df_predictions = df.iloc[LOOKBACK:].copy()
    df_predictions['predicted_temperature'] = predictions
    
    # Flag extremes: here we simply flag if the normalized prediction is high (customize your logic)
    normalized_preds = scaler.transform(df_predictions['predicted_temperature'].values.reshape(-1, 1))
    df_predictions['extreme_event'] = normalized_preds > EXTREME_THRESHOLD
    
    # Plot actual vs forecasted temperatures
    plt.figure(figsize=(10, 6))
    plt.plot(df_predictions['date'], df_predictions['temperature'], label='Actual Temperature')
    plt.plot(df_predictions['date'], df_predictions['predicted_temperature'], label='Predicted Temperature', linestyle='--')
    plt.axhline(y=scaler.inverse_transform([[EXTREME_THRESHOLD]])[0][0], color='red', linestyle=':', label='Extreme Threshold')
    plt.title("Temperature Forecast and Extreme Event Prediction")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.legend()
    plt.savefig("output/plots/temperature_forecast.png")
    plt.show()
    
    # Print identified extreme events
    extreme_events = df_predictions[df_predictions['extreme_event']]
    print("Potential Extreme Weather Events Detected:")
    print(extreme_events[['date', 'temperature', 'predicted_temperature']])
    
if __name__ == '__main__':
    main()