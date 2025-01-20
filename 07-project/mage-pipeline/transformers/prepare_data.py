if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import numpy as np



@transformer
def transform(data, *args, **kwargs):
    """
    This function drops invalid rows from the data and normalises 
    skewed features.

    Args:
        data: The output from load_data.py

    Returns:
        The dataframe and target feature
    """

    #removing rows that have negative values in columns

    for i in ['SO2', 'PM10']:
      data = data[data[i] >= 0]

    numerical_columns = data.select_dtypes(exclude= 'object').columns

    #normalising the skewed features with sqrt transformation
    for i in numerical_columns:
        data[i] = np.sqrt(data[i])

    #encoding the target feature
    target = data['Air Quality'].map({'Moderate': 0, 'Good': 1, 'Hazardous': 2, 'Poor': 3})


    return data, target
