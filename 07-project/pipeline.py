import pandas as pd
import kagglehub
import numpy as np
import joblib


def load_data():
    try:
        path = kagglehub.dataset_download("mujtabamatin/air-quality-and-pollution-assessment")
        path += '/updated_pollution_dataset.csv'
    except Exception as e:
        return f'error: {str(e)}'

    data = pd.read_csv(path)


    return data

def preprocess(data):
    for i in ['SO2', 'PM10']:
      data = data[data[i] >= 0]

    numerical_columns = data.select_dtypes(exclude= 'object').columns

    #normalising the skewed features with sqrt transformation
    for i in numerical_columns:
        data[i] = np.sqrt(data[i])

    return data

def main():
    data = load_data()
    data = preprocess(data)
    X = data.drop('Air Quality', axis=1)

    with open('07-project\artifacts\RFC-v5\random_forest_model.joblib', 'rb') as f_in:
        model = joblib.load(f_in)
    
    y_pred = model.predict(X)
    print(y_pred)