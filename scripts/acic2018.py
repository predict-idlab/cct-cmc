import numpy as np
import pandas as pd
from evaluation import evaluate_distribution
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    df_data = pd.read_csv("../data/ACIC2018/00ea30e866f141d9880d5824a361a76a.csv")
    W = df_data.iloc[:, 0].values
    Y0 = df_data.iloc[:, 1].values
    Y1 = df_data.iloc[:, 2].values
    Y = W * Y1 + (1 - W) * Y0
    X = df_data.iloc[:, 5:].values
    ps = LogisticRegression(solver="lbfgs").fit(X, W).predict_proba(X)[:, 1]
    df = pd.DataFrame(
        np.column_stack((X, W, Y, Y0, Y1, ps)),
        columns=[f"X{i}" for i in range(1, X.shape[1] + 1)] + ["W", "Y", "Y0", "Y1", "ps"],
    )
    df_train, df_test = train_test_split(df, test_size=0.2)
    evaluate_distribution(
        df_train=df_train,
        df_test=df_test,
        num_cov=X.shape[1],
        output_path="../results/outputs/acic2018/eval_dist/acic2018_results.csv",
        dataset_name="acic2018",
        alpha=0.1,
        verbose=True,
        store_p_values=False,
    )


if __name__ == "__main__":
    main()
