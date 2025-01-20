import pandas as pd
import numpy as np
import joblib
from pipeline import load_data, preprocess

def test_integration():
    """
    This test verifies the full pipeline from data loading,
    preprocessing, to model prediction.
    """
    # Step 1: Load data
    data = load_data()
    assert isinstance(data, pd.DataFrame), "Failed to load a valid DataFrame"

    #Preprocessing data
    processed_data = preprocess(data)
    assert isinstance(processed_data, pd.DataFrame), "Preprocessing did not return a DataFrame"
    assert (processed_data['SO2'] >= 0).all(), "SO2 column contains invalid values after preprocessing"
    assert (processed_data['PM10'] >= 0).all(), "PM10 column contains invalid values after preprocessing"

    # Dropping target column and prepare for prediction
    X = processed_data.drop('Air Quality', axis=1)
    assert 'Air Quality' not in X.columns, "Target column not removed from feature set"

    # Loading model
    try:
        with open('07-project/artifacts/RFC-v5/random_forest_model.joblib', 'rb') as f_in:
            model = joblib.load(f_in)
    except Exception as e:
        assert False, f"Model loading failed: {str(e)}"

    # Making predictions
    try:
        y_pred = model.predict(X)
    except Exception as e:
        assert False, f"Model prediction failed: {str(e)}"

    assert len(y_pred) == len(X), "Mismatch in number of predictions and input samples"

    print("Integration test passed: Full pipeline works as expected.")

if __name__ == "__main__":
    test_integration()
