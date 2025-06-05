import pandas as pd
import numpy as np

def preprocess_data(input_file, output_file):
    # Load the data
    df = pd.read_csv(input_file, parse_dates=['date'])
    
    # Sort by date
    df.sort_values('date', inplace=True)
    
    # Fill missing values (simple forward fill)
    df.fillna(method='ffill', inplace=True)
    
    # Create additional features, e.g., day of year (could be useful for seasonality)
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Optionally, filter or aggregate data if needed
    # For example, computing daily averages if you have multiple readings per day
    df = df.groupby('date').mean().reset_index()
    
    # Save the cleaned data
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == '__main__':
    preprocess_data("data/climate_data.csv", "data/climate_clean.csv")