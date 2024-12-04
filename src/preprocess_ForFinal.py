import pandas as pd

def preprocess(df_input):
    """ 
    Example Usage:
    ```
    df_input = pd.read_csv('dataset/inputs.csv')
    
    df = preprocess(df_input)
    ```
    """
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
        'BMI'
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
    ]
    df[columns_to_transform] = df[columns_to_transform].astype(bool)

    # Rounding
    df['BMI'] = df['BMI'].round(2)

    return df