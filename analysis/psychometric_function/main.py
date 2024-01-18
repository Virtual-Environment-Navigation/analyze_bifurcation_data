from config.config import SUBGROUP_BINARY_DATA
from utilities.logistic_regression import perform_logistic_regression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_logistic_regression(df, x_col, y_col):
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot logistic regression for all subjects
    sns.regplot(x=x_col, y=y_col, data=df, logistic=True, ax=ax, line_kws={'color': 'blue'})
    plt.axhline(y=0.5, color='r', linestyle='--')  # Add dashed line at y=0.5
    plt.title('Logistic Regression for All Subjects')

    subjects = df['Subject'].unique()

    # Plot logistic regression for each subject
    for subject in subjects:
        subject_df = df[df['Subject'] == subject]
        sns.regplot(x=x_col, y=y_col, data=subject_df, logistic=True, ax=ax, ci=None, line_kws={'color': 'grey', 'alpha': 0.5})

    plt.show()


def main():
    print("Calculating the Psychometric Function(s)")

    df = pd.read_csv(SUBGROUP_BINARY_DATA)

    df['Alpha'] = df['Alpha'].replace(12, 120) # replace values of 12 with 120!! 12 isn't an alpha level

    logistic_regression_model = perform_logistic_regression(df, 'Alpha', 'Followed_Subgroup')

    plot_logistic_regression(df, 'Alpha', 'Followed_Subgroup')

    # plot_logistic_regression_by_subject(df, 'Alpha', 'Followed_Subgroup')

    f = 10
