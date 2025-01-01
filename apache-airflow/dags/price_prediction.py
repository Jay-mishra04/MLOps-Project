from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
# Import Libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Define the functions for each task

def extract_data():
    try:
        file_path = '/opt/airflow/dags/data/flights.csv'
        df = pd.read_csv(file_path)
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df.to_pickle('/opt/airflow/dags/data/raw_data.pkl')
        print("Data extracted and saved as pickle")
    except Exception as e:
        print(f"Error in extract_data: {e}")

def transform_data():
    try:
        df = pd.read_pickle('/opt/airflow/dags/data/raw_data.pkl')
        # Drop unnecessary columns
        df = df.drop(columns=['travelCode', 'userCode', 'date', 'time', 'distance'])
        # One-hot encode categorical columns
        encoded_df = pd.get_dummies(df, columns=['from', 'to', 'flightType', 'agency'], drop_first=True)
        encoded_df.to_pickle('/opt/airflow/dags/data/transformed_data.pkl')
        print("Data transformed and saved as pickle")
    except Exception as e:
        print(f"Error in transform_data: {e}")

def train_model():
    try:
        df = pd.read_pickle('/opt/airflow/dags/data/transformed_data.pkl')
        X = df.drop(columns=['price'])  # Features
        y = df['price']  # Target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Save the model
        with open('/opt/airflow/dags/data/model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)

        print("Model trained and saved")
    except Exception as e:
        print(f"Error in train_model: {e}")

def evaluate_model():
    try:
        df = pd.read_pickle('/opt/airflow/dags/data/transformed_data.pkl')
        X = df.drop(columns=['price'])  # Features
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with open('/opt/airflow/dags/data/model.pkl', 'rb') as f:
            model = pickle.load(f)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Evaluation Metrics:\nMSE: {mse:.2f}\nMAE: {mae:.2f}\nR² Score: {r2:.2f}")
    except Exception as e:
        print(f"Error in evaluate_model: {e}")

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    dag_id='flight_price_prediction',
    default_args=default_args,
    description='Automated workflow for managing travel data and regression model',
    schedule_interval='@daily',  
    start_date=datetime(2023, 12, 20),
    catchup=False,
) as dag:

    task_extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
    )

    task_transform = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
    )

    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    task_evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    # Define task dependencies
    task_extract >> task_transform >> task_train >> task_evaluate
