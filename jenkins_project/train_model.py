import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def train_model(input_file="cleaned_df.csv", model_file="trained_pipeline.pkl"):
    try:
        # Load preprocessed data
        logging.info(f"Loading cleaned data from '{input_file}'...")
        df = pd.read_csv(input_file)
        
        # Separate features (X) and target variable (y)
        logging.info("Separating features and target variable...")
        X = df.drop(columns="price")
        y = df["price"]
        
        # Identify categorical columns for one-hot encoding
        categorical_columns = ['from', 'to', 'flightType', 'agency']
        
        # Define preprocessing: one-hot encode categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
            ],
            remainder="passthrough"  # Leave other columns (if any) unchanged
        )
        
        # Create a pipeline: preprocessing + model
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train the pipeline
        logging.info("Training the pipeline...")
        pipeline.fit(X, y)
        
        # Save the entire pipeline to a file
        logging.info(f"Saving the trained pipeline to '{model_file}'...")
        joblib.dump(pipeline, model_file)
        logging.info("Pipeline training completed successfully!")
    
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")

if __name__ == "__main__":
    train_model(input_file="cleaned_df.csv", model_file="trained_pipeline.pkl")
