import pandas as pd
import numpy as np
from pipeline import load_data, preprocess  

# Test load_data
def test_load_data():
    data = load_data()
    # Ensure data is not None and is a DataFrame
    assert data is not None, "Data should not be None"
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame"

    # Check that essential columns exist
    required_columns = ['SO2', 'PM10', 'Air Quality']
    for col in required_columns:
        assert col in data.columns, f"Missing required column: {col}"

    print("load_data passed all checks.")

# Test preprocess
def test_preprocess():
    # Create a mock dataset
    mock_data = pd.DataFrame({
    'Temperature': np.random.randint(20, 35, size=5),  # Example: temperatures between 20 and 35
    'Humidity': np.random.randint(40, 80, size=5),    # Example: humidity between 40 and 80
    'PM2.5': np.random.randint(10, 50, size=5),       # Example: PM2.5 values between 10 and 50
    'PM10': np.random.randint(30, 80, size=5),        # Example: PM10 values between 30 and 80
    'NO2': np.random.randint(5, 30, size=5),          # Example: NO2 values between 5 and 30
    'SO2': np.random.randint(2, 15, size=5),           # Example: SO2 values between 2 and 15
    'CO': np.random.randint(1, 10, size=5),            # Example: CO values between 1 and 10
    'Proximity_to_Industrial_Areas': np.random.randint(0, 2, size=5),  # Example: 0 or 1 (near or far)
    'Population_Density': np.random.randint(100, 1000, size=5),  # Example: population density between 100 and 1000
})


    processed_data = preprocess(mock_data)

    # Assert rows with negative values are removed
    assert (processed_data['SO2'] >= 0).all(), "Rows with negative SO2 values should be removed"
    assert (processed_data['PM10'] >= 0).all(), "Rows with negative PM10 values should be removed"

    # Assert numerical columns are transformed correctly
    numerical_columns = processed_data.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        assert (processed_data[col] >= 0).all(), f"Column {col} should be transformed with sqrt"

    print("preprocess passed all checks.")

# Run tests
if __name__ == "__main__":
    test_load_data()
    test_preprocess()
