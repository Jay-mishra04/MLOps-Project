import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_model(input_file="cleaned_df.csv", model_file="trained_pipeline.pkl", test_size=0.2):
    try:  
        # Load the cleaned dataset
        logging.info(f"Loading cleaned data from '{input_file}'...")
        df = pd.read_csv(input_file)
        
        # Separate features (X) and target variable (y)
        logging.info("Separating features and target variable...")
        X = df.drop(columns="price")
        y = df["price"]
        
        # Split the data into training and testing sets
        logging.info(f"Splitting the data into training and testing sets (test size = {test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Load the trained model
        logging.info(f"Loading the trained model from '{model_file}'...")
        model = joblib.load(model_file)
        
        # Make predictions on the test set
        logging.info("Making predictions on the test data...")
        predictions = model.predict(X_test)
        
        # Evaluate the model
        logging.info("Evaluating model performance...")
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logging.info(f"Mean Squared Error (MSE): {mse:.2f}")
        logging.info(f"Mean Absolute Error (MAE): {mae:.2f}")
        logging.info(f"R-Squared (R2 Score): {r2:.2f}")
        
        print("\nModel Performance Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R-Squared (R2 Score): {r2:.2f}")
    
    except Exception as e:
        logging.error(f"An error occurred during model testing: {e}")

if __name__ == "__main__":
    # Default arguments for input CSV and model file
    test_model(input_file="cleaned_df.csv", model_file="trained_pipeline.pkl", test_size=0.2)
