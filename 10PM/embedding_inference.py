import pandas as pd
import re
import random
from tqdm import tqdm

device = 'cuda'

def load_Transformer(path: str):
    model_Transformer = SentenceTransformer(path).to(device)
    return model_Transformer

def load_Regression(path: str):
    class Regression(nn.Module):
        def __init__(self):
            super(Regression, self).__init__()
            self.linear = nn.LazyLinear(1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            return self.sigmoid(self.linear(x))
    
    model = Regression(Regression).to(device)
    model.load_state_dict(torch.load(path))
    
    return model

def predict(model_Transformer, model_Regression, X):
    embedding = torch.tensor(model_Transformer.encode(X)).to(device)
    pred = model_Regression(embedding).detach().cpu().numpy()
    pred = np.round(pred)

    return pred # Numpy array of number of different patients 


def preprocess(df_input):
    """
    Example Usage:
    ```
    df_input = pd.read_csv('dataset/inputs.csv')
    df_outputs = pd.read_csv('dataset/labels.csv')

    df = preprocess(df_input, df_outputs)
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
        'BMI',
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
        'HadKidneyDisease',
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


def format_string(row):
    # row is a row in dataframe
    features = ", ".join([f"{col}: {row[col]}" for col in row.keys()])

def format_output(df, pred):

    new_df = pd.DataFrame({
        'PatientID': df['Patient ID'],
        'HadHeartAttack': pred.tolist()
    })

    return new_df


if __name__ == '__main__.py':
    path_Transformer = ''
    path_Regression = ''
    data_path = ''
    
    # Load data
    df = pd.read_csv(data_path)
    df = preprocess(df)
    features = [format_string(row) for _, row in df.iterrows()]

    # Model
    model_Transformer = load_Transformer(path_Transformer)
    model_Regression = load_Regression(path_Regression)

    # Prediction
    pred = predict(model_Transformer, model_Regression, features)

    # Convert to Output
    final = format_output(df, pred)
    new_df.to_csv('patient_heart_attack_data.csv', index=False)




