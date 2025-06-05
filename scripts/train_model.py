import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Parameters
LOOKBACK = 30  # days to look back for forecasting
EPOCHS = 50
BATCH_SIZE = 16

def create_dataset(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def main():
    # Load cleaned data
    df = pd.read_csv("data/climate_clean.csv", parse_dates=['date'])
    
    # Assume we're forecasting 'temperature' as an example
    # You can change the column for different forecasts
    values = df['temperature'].values.reshape(-1, 1)
    
    # Normalize the values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    
    # Create time-series dataset
    X, y = create_dataset(scaled_values, LOOKBACK)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(LOOKBACK, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='loss', patience=5)
    
    # Train the model
    history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop])
    
    # Save the trained model and scaler
    model.save("output/climate_model.h5")
    pd.to_pickle(scaler, "output/scaler.pkl")
    print("Model and scaler saved to the output folder.")
    
    # Optionally, visualize the training loss
    plt.plot(history.history['loss'])
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("output/plots/training_loss.png")
    plt.show()

if __name__ == '__main__':
    main()