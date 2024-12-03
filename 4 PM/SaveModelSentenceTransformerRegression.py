import pandas as pd
import re
import random
from tqdm import tqdm

path = 'dataset/processed_inputs.csv'
df = pd.read_csv(path)

import pandas as pd

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

df_input =  pd.read_csv('dataset/inputs.csv')
df_outputs = pd.read_csv('dataset/labels.csv')
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


from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample, losses
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

model_Transformer = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

features, labels = batch(1000)
X = model_Transformer.encode(features)
y = np.array(labels)

type(X), type(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(penalty=None, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained Logistic Regression model
import joblib
joblib.dump(model, 'logistic_regression_model.joblib')
print("Model saved to logistic_regression_model.joblib")