from unsloth import FastLanguageModel
from tqdm import tqdm

import pandas as pd
import json
import csv

def preprocess(df_input):
    print("Start processing csv into json....")
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
    # Keep only specified columns
    df = df_input[columns_to_keep]

    # Transform specified columns to boolean
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
    # Ensure columns exist before attempting transformation
    columns_to_transform = [col for col in columns_to_transform if col in df.columns]
    df[columns_to_transform] = df[columns_to_transform].astype(bool)

    # Round BMI to 2 decimal places if it exists
    if 'BMI' in df.columns:
        df['BMI'] = df['BMI'].round(2)

    return df

def patient_detail(df):
    """
    Converts a DataFrame to a JSON file where each row is a JSON object.
    """
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    # Convert DataFrame to dictionary
    data = df.to_dict(orient='records')

    return data

def json_process(input_json):
    print("Start formatting to alpaca format....")
    # Convert to Alpaca format
    alpaca_data = []
    for patient in input_json:
        
        # Format patient details into a multi-line string
        patient_details = "\n".join([f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in patient.items()])
        # output = "True" if patient["hadheartattack"] else "False"

        # print(patient_details)
        alpaca_prompt = {
            # "instruction": instruction,
            "input": f"{patient_details}"
            # "response": output
        }
        alpaca_data.append(alpaca_prompt)

    # Save to a new JSON file
    output_file = "input_text.json"
    with open(output_file, "w") as f:
        json.dump(alpaca_data, f, indent=4)

    print("Json loaded completed.")

    return output_file

def load_model(model_path):
    global tokenizer, model, alpaca_prompt
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True 

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    # alpaca_prompt = Copied from above
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

def evaluate(input_text):
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Given the information below, classify whether the patient has a heart attack or not based on the medical details provided. Respond with 'True' if the patient has had a heart attack and 'False' if not.", # instruction
            input_text,
            "",
        )
    ], return_tensors = "pt").to("cuda")

    # Generate response
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)

    # Decode and extract the response
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Extract the response part
    for output in decoded_output:
        # Look for the part after "### Response:" marker
        response = output.split("### Response:")[1].strip()
    
    return response
    
def main(df, eval_file):
    df.columns = df.columns.str.strip().str.lower()
    with open(eval_file, "r") as file:
        data = json.load(file)

    patient_ids = df['patientid'].tolist()

    # Open the file once in write mode to initialize it and add headers if necessary
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

    # Open the file in append mode to add rows
    with open('output.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        for patient_id, entry in tqdm(zip(patient_ids, data), total=len(patient_ids)):
            input_text = entry.get("input")

            response = evaluate(input_text)
            if response == "True":
                response = 1
            else:
                response = 0

            # Write each row
            writer.writerow([patient_id, response])

    print("output.csv is saved.")


if __name__ == '__main__':

    ########################Input here########################
    model_path = ''
    csv_path = ''
    ########################Input here########################

    df_input = pd.read_csv(csv_path)

    df = preprocess(df_input)

    data = patient_detail(df)

    output_file = json_process(data)

    load_model(model_path)
    main(df, output_file)