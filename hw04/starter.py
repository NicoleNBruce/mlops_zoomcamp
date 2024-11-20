# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import os
import sys


def read_data(filename, categorical):
    """This function reads and loads the data"""
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def apply_model(year, month, input_file, output_file):
    """This function loads and applies the model on the data"""
    try:
        with open('model.bin', 'rb') as f_in:
            dv, model = pickle.load(f_in)
    except FileNotFoundError:
        print("Error: model.bin not found.")
    except Exception as e:
        print(f"Error while loading the model: {e}")

    categorical = ['PULocationID', 'DOLocationID']
    df = read_data(input_file, categorical)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    print(df_result['predicted_duration'].mean())
    try:
        df_result.to_parquet(
            output_file,
            engine='pyarrow',
            compression=None,
            index=False
        )
    except Exception as e:
        print(f"Something went wrong: {e}")


def run():
    #taking the cmd arguments
    try:
        year = int(sys.argv[1])
        month = int(sys.argv[2])
    except IndexError:
        print("Error: Please provide year and month as command-line arguments.")
    except ValueError:
        print("Error: Year and month should be valid integers.")


    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-0{month}.parquet'
    output_file = f'output/yellow/{year}-{month}.parquet'

    # creating the directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except Exception as e:
        print(f"Error creating output directory: {e}")
    
    apply_model(year, month, input_file, output_file)


run()
