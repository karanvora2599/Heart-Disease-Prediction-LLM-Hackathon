You'll need to preprocess your csv file, then run it on our logistic regression.
```
# Preprocessing
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
    df['HeightInMeters'] = df['HeightInMeters'].round(2)

    ### Fix Column Names
    new_columns = ['Patient ID', 'Sex', 'HIV Testing', 'E-Cigarette Usage',
               'Difficulty Concentrating', 'Had Asthma', 'Had Depressive Disorder',
               'Covid Positive', 'Flu Vaccine Last 12 Months', 'Race/Ethnicity Category', 'Had Diabetes',
               'Difficulty Dressing/Bathing', 'Chest Scan', 'Had COPD',
               'Blind or Vision Difficulty', 'High Risk Last Year', 'Had Angina',
               'Pneumonia Vaccine Ever', 'Had Skin Cancer', 'Had Arthritis', 'Deaf or Hard of Hearing',
               'Alcohol Drinkers', 'Had Kidney Disease', 'Tetanus Last 10 Years (Tdap)',
               'Smoker Status', 'Height in Meters', 'BMI', 'Had Heart Attack']

    df.columns = new_columns

    return df

df = preprocess(df_input, df_outputs)

class_0 = df[df['Had Heart Attack'] == 0]
class_1 = df[df['Had Heart Attack'] == 1]

class_0 = class_0.drop(columns=["Had Heart Attack", "Patient ID"])
class_1 = class_1.drop(columns=["Had Heart Attack", "Patient ID"])

def batch(size: int):
    # Generate features for class 0 and class 1
    feature1 = [", ".join([f"{col}: {row[col]}" for col in class_0.columns]) for _, row in class_0.head(size).iterrows()]
    feature2 = [", ".join([f"{col}: {row[col]}" for col in class_1.columns]) for _, row in class_1.head(size).iterrows()]

    # Combine features and labels
    features = feature1 + feature2
    labels = [0] * size + [1] * size

    # Shuffle features and labels together
    combined = list(zip(features, labels))
    random.shuffle(combined)
    shuffled_features, shuffled_labels = zip(*combined)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(shuffled_labels, dtype=torch.float32)

    return list(shuffled_features), labels_tensor



X = batch(1) # single datapoint
model = ... # Load `logistic_regression_model.joblib`
prediction = model.predict(X)
```
