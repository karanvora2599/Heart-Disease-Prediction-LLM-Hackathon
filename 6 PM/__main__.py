import argparse
import os
import pandas as pd
import numpy as np
import joblib
import torch
import random

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

def preprocess_test_data(df_input):
    """
    Preprocess the test input data to match the training data format.
    """
    # Which columns to keep (ensure these match the training columns except 'Had Heart Attack')
    columns_to_keep = [
        'PatientID',
        'Sex',
        'HIVTesting',
        'ECigaretteUsage',
        'DifficultyConcentrating',
        'HadAsthma',
        'HadDepressiveDisorder',
        'CovidPos',
        'FluVaxLast12',
        'RaceEthnicityCategory',
        'HadDiabetes',
        'DifficultyDressingBathing',
        'ChestScan',
        'HadCOPD',
        'BlindOrVisionDifficulty',
        'HighRiskLastYear',
        'HadAngina',
        'PneumoVaxEver',
        'HadSkinCancer',
        'HadArthritis',
        'DeafOrHardOfHearing',
        'AlcoholDrinkers',
        'HadKidneyDisease',
        'TetanusLast10Tdap',
        'SmokerStatus',
        'HeightInMeters',
        'BMI'
    ]
    df = df_input[columns_to_keep]

    # Turn to bool
    columns_to_transform = [
        'DifficultyConcentrating',
        'HadAsthma',
        'HadDepressiveDisorder',
        'CovidPos',
        'FluVaxLast12',
        'DifficultyDressingBathing',
        'ChestScan',
        'HadCOPD',
        'BlindOrVisionDifficulty',
        'HighRiskLastYear',
        'HadAngina',
        'PneumoVaxEver',
        'HadSkinCancer',
        'HadArthritis',
        'DeafOrHardOfHearing',
        'AlcoholDrinkers',
        'HadKidneyDisease'
    ]
    df[columns_to_transform] = df[columns_to_transform].astype(bool)

    # Rounding
    df['BMI'] = df['BMI'].round(2)
    df['HeightInMeters'] = df['HeightInMeters'].round(2)

    ### Fix Column Names
    new_columns = ['Patient ID', 'Sex', 'HIV Testing', 'E-Cigarette Usage',
               'Difficulty Concentrating', 'Had Asthma', 'Had Depressive Disorder',
               'Covid Positive', 'Flu Vaccine Last 12 Months', 'Race/Ethnicity Category', 'Had Diabetes',
               'Difficulty Dressing/Bathing', 'Chest Scan', 'Had COPD',
               'Blind or Vision Difficulty', 'High Risk Last Year', 'Had Angina',
               'Pneumonia Vaccine Ever', 'Had Skin Cancer', 'Had Arthritis', 'Deaf or Hard of Hearing',
               'Alcohol Drinkers', 'Had Kidney Disease', 'Tetanus Last 10 Years (Tdap)',
               'Smoker Status', 'Height in Meters', 'BMI']

    df.columns = new_columns

    return df

def main(test_set_dir: str, results_dir: str):
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load test set data
    input_csv_path = os.path.join(test_set_dir, "inputs.csv")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Test inputs file not found at {input_csv_path}")

    input_df = pd.read_csv(input_csv_path)

    # Preprocess test data
    processed_df = preprocess_test_data(input_df)

    # Generate feature strings
    features = [", ".join([f"{col}: {row[col]}" for col in processed_df.columns if col != 'Patient ID']) for _, row in processed_df.iterrows()]
    patient_ids = processed_df['Patient ID'].tolist()

    # Initialize the SentenceTransformer
    model_transformer = SentenceTransformer("all-distilroberta-v1")

    # Encode features
    X_test = model_transformer.encode(features, convert_to_numpy=True)

    # Load the trained Logistic Regression model
    model_path = os.path.join(os.path.dirname(__file__), "logistic_regression_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    model = joblib.load(model_path)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = y_pred.astype(int)  # Ensure integer type (0 or 1)

    # Prepare results DataFrame
    results_df = pd.DataFrame({
        "PatientID": patient_ids,
        "HadHeartAttack": y_pred
    })

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Save results to results.csv
    results_csv_path = os.path.join(results_dir, "results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease Prediction Inference Script")
    parser.add_argument(
        "--bth_test_set",
        type=str,
        required=True,
        help="Path to the directory containing the test set inputs (inputs.csv)"
    )
    parser.add_argument(
        "--bth_results",
        type=str,
        required=True,
        help="Path to the directory where results.csv will be saved"
    )

    args = parser.parse_args()
    main(args.bth_test_set, args.bth_results)