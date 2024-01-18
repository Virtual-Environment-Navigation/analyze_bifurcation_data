from config.config import FINAL_HEADING_DATA
import numpy as np
import pandas as pd
from utilities.uni_vs_bi_model_bics import uni_vs_bi_model_bics
import json

def main():
    df = pd.read_csv(FINAL_HEADING_DATA)

    all_alpha_values = sorted(list(set(df['Alpha'].values.tolist())))

    bic_results = {}

    for alpha_value in all_alpha_values:

        ALPHA_CONDITION = float(alpha_value)  # the condition that we're assessing
        alpha_heading = ALPHA_CONDITION / 2  # the heading that the subgroups are moving towards
        ALPHA_INPUT = float(20)  # the condition that is being used to create the statistical model

        assessed_condition = df[df['Alpha'] == ALPHA_CONDITION]['Final_Heading'].tolist()
        input_condition = df[df['Alpha'] == ALPHA_INPUT]['Final_Heading'].tolist()

        std_dev = np.std(input_condition)

        [bic_uni, bic_bi] = uni_vs_bi_model_bics(assessed_condition, std_dev, alpha_heading)

        bic_results[alpha_value] = [bic_uni, bic_bi]

    # NOTE: must manually create data directory in main project if it doesn't exist
    with open('data/bifurcation_model_fit_results.json', 'w') as f:
        json.dump(bic_results, f)


    f = 0
