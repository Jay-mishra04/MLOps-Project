import pandas as pd
import logging

def preprocess_data(input_file, output_file):
    logging.basicConfig(level=logging.INFO)

    # Load dataset
    logging.info("Loading dataset...")
    df = pd.read_csv(input_file)

    # Check for missing values
    logging.info("Dropping missing values...")
    df.dropna(inplace=True)

    # Drop duplicate rows
    logging.info("Dropping duplicate rows...")
    df.drop_duplicates(inplace=True)

    # Drop unnecessary columns
    columns_to_drop = ["travelCode", "userCode", "time", "distance", "date"]
    logging.info(f"Dropping columns: {columns_to_drop}")
    for col in columns_to_drop:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df.drop(columns=columns_to_drop, inplace=True)

    # Save cleaned data
    logging.info(f"Saving cleaned dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    logging.info("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data("flights.csv", "cleaned_df.csv")
