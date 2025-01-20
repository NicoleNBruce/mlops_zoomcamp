import datetime
import logging
import pandas as pd
import joblib
import psycopg  # For PostgreSQL database interaction
import kagglehub

from evidently.report import Report
from evidently.metrics import DatasetMissingValuesMetric, ColumnDriftMetric

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

with open('07-project\artifacts\RFC-v5\random_forest_model.joblib', 'rb') as f_in:
	model = joblib.load(f_in)

path = kagglehub.dataset_download("mujtabamatin/air-quality-and-pollution-assessment")
path += '/updated_pollution_dataset.csv'
dataset = pd.read_csv(path)

# Select feature columns
feature_columns = [
    'Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO',
    'Proximity_to_Industrial_Areas', 'Population_Density'
]

# Evidently report setup
report = Report(metrics=[
    DatasetMissingValuesMetric(),
    ColumnDriftMetric(column_name='Air Quality')  
])

# Database connection settings
DB_CONNECTION_STRING = "host=localhost port=5432 dbname=air_quality user=postgres password=example"

def prep_db():
    """Prepare the database table for storing metrics."""
    create_table_statement = """
    CREATE TABLE IF NOT EXISTS air_quality_metrics (
        timestamp TIMESTAMP,
        share_missing_values FLOAT,
        air_quality_drift FLOAT
    );
    """
    with psycopg.connect(DB_CONNECTION_STRING, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_statement)

def calculate_metrics_and_log_to_db():
    """Calculate metrics and store them in the database."""
    # Predict using the model
    dataset['Predictions'] = model.predict(dataset[feature_columns])

    # Run Evidently report
    report.run(reference_data=dataset, current_data=dataset, column_mapping=None)
    results = report.as_dict()

    # Extract metrics
    share_missing_values = results['metrics'][0]['result']['current']['share_of_missing_values']
    air_quality_drift = results['metrics'][1]['result']['drift_score']

    # Log metrics to database
    timestamp = datetime.datetime.now()
    with psycopg.connect(DB_CONNECTION_STRING, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO air_quality_metrics (timestamp, share_missing_values, air_quality_drift)
                VALUES (%s, %s, %s);
                """,
                (timestamp, share_missing_values, air_quality_drift)
            )
    logging.info(f"Metrics logged: {timestamp}, Missing Values: {share_missing_values}, PM2.5 Drift: {air_quality_drift}")


def air_quality_monitoring_with_db():
    """Main monitoring flow."""
    prep_db()
    calculate_metrics_and_log_to_db()
    logging.info("Monitoring and logging to database complete.")

if __name__ == '__main__':
    air_quality_monitoring_with_db()
