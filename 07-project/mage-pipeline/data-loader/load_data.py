if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import kagglehub
import pandas as pd

@data_loader
def load_data(*args, **kwargs):
     """
    This function gets the parquet data file and converts it into
    a Pandas dataframe

    Returns:
        The Pandas dataframe
    """
    try:
        path = kagglehub.dataset_download("mujtabamatin/air-quality-and-pollution-assessment")
        path += '/updated_pollution_dataset.csv'
    except Exception as e:
        return f{'error': str(e)}

    data = pd.read_csv(path)


    return data
