if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@transformer
def transform(data, *args, **kwargs):
    """
    This function trains a random forest classifier with the data
    Args:
        data: The dataframe and target feature

    Returns:
        The model
    """
    df, target = data

    X = df.drop(columns=['Air Quality'])
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

    model = RandomForestClassifier(random_state=8, n_estimators=100, min_samples_split=2)
    model.fit(X_train, y_train)

    return model
