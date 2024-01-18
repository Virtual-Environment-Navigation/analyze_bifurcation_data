from sklearn.linear_model import LogisticRegression
import numpy as np

def perform_logistic_regression(df, predictor_var, target_var):
    X = df[[predictor_var]]
    y = df[target_var]

    model = LogisticRegression()
    model.fit(X, y)

    return model