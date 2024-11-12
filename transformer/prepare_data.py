if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd

@transformer
def transform(data, *args, **kwargs):
    """
    This function converts some string columns into datetime format
    and generates a new column derived by subtracting the two
    formatted columns

    Args: 
        data: output from load_data.py

    Returns:
        The DictVectorizer object and model
    """

    data.tpep_dropoff_datetime = pd.to_datetime(data.tpep_dropoff_datetime)
    data.tpep_pickup_datetime = pd.to_datetime(data.tpep_pickup_datetime)

    data['duration'] = data.tpep_dropoff_datetime - data.tpep_pickup_datetime
    data.duration = data.duration.dt.total_seconds() / 60

    data = data[(data.duration >= 1) & (data.duration <= 60)]

    #converting the IDs into strings
    categorical = ['PULocationID', 'DOLocationID']
    data[categorical] = data[categorical].astype(str)
    
    return data
