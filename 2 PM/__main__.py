import argparse
import os

import pandas as pd
import numpy as np


PREDICTION_WINDOW_MONTHS = [3, 6, 9, 12]  # Constant for this charge-off prediction task.


def main(test_set_dir: str, results_dir: str):
    
    # Load test set data.
    input_df = pd.read_csv(os.path.join(test_set_dir, "inputs.csv"))

    # ---------------------------------
    # START PROCESSING TEST SET INPUTS
    # Beep boop bop you should do something with test inputs unlike this script.

    # In lieu of doing something test inputs, maybe you "learned" from training data that
    #  20% of patients have heart attacks, so you randomly guess with that percentage.
    patients = list(input_df.PatientID)
    output_df = pd.DataFrame(columns=["PatientID", "HadHeartAttack"])
    output_df["PatientID"] = patients
    heart_attack_percent = 0.2
    had_heart_attack = np.random.random(len(patients)) < heart_attack_percent
    output_df["HadHeartAttack"] = had_heart_attack

    # END PROCESSING TEST SET INPUTS
    # ---------------------------------

    # NOTE: name "results.csv" is a must.
    output_df.to_csv(os.path.join(results_dir, "results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bth_test_set",
        type=str,
        required=True
    )
    parser.add_argument(
        "--bth_results",
        type=str,
        required=True
    )

    args = parser.parse_args()
    main(args.bth_test_set, args.bth_results)