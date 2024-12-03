import pandas as pd
import json

def preprocess(df_input, df_outputs):
    """ 
    Example Usage:
    ```
    df_input = pd.read_csv('dataset/inputs.csv')
    df_outputs = pd.read_csv('dataset/labels.csv')
    
    df = preprocess(df_input, df_outputs)
    ```
    """
    df = pd.merge(df_input, df_outputs, on='PatientID', how='inner')

    # Which columns to keep
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
        'BMI',
        'HadHeartAttack'
    ]
    df = df[columns_to_keep]

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
        'HadKidneyDisease',
        'HadHeartAttack'
    ]
    df[columns_to_transform] = df[columns_to_transform].astype(bool)

    # Rounding
    df['BMI'] = df['BMI'].round(2)

    return df

def patiend_detail(df):
    """
    Converts a DataFrame to a JSON file where each row is a JSON object.
    """
    
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    # Convert DataFrame to dictionary
    data = df.to_dict(orient='records')
    
    return data

def json_process(input_json):
    instruction = "Given the information below, classify whether the patient has a heart attack or not based on the medical details provided. Respond with 'Yes' if the patient has had a heart attack and 'No' if not."

    # Convert to Alpaca format
    alpaca_data = []
    for patient in input_json:
        # Format patient details into a multi-line string
        patient_details = "\n".join([f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in patient.items()])
        output = "Yes" if patient["hadheartattack"] else "No"
        alpaca_prompt = {
            "instruction": instruction,
            "input": f"{patient_details}",
            "response": output
        }
        alpaca_data.append(alpaca_prompt)

    # Save to a new JSON file
    output_file = "alpaca_formatted_test2.json"
    with open(output_file, "w") as f:
        json.dump(alpaca_data, f, indent=4)

if __name__ == '__main__':
    print("Start processing csv into json....")
    df_input = pd.read_csv('dataset/inputs.csv')
    df_outputs = pd.read_csv('dataset/labels.csv')

    df = preprocess(df_input, df_outputs)
    data = patiend_detail(df)
    print("Saved patient details in json.")
    print("Start formatting to alpaca format....")

    json_process(data)
    print("Completed.")