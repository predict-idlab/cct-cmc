import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

NLM_path = "../../data/NLSM/NLSM_data.csv"


def get_nlsm_sim_data():
    df = pd.read_csv(NLM_path)
    X = df.drop(columns=["Z", "Y"])
    W = df["Z"].to_numpy()
    Y = df["Y"].to_numpy()

    gamma = np.random.normal(0, 1, size=76) * 0.105

    # Based on Section 2 of Carvalho et al. (2019)
    ite = (
        0.228
        + 0.05 * (X["X1"] < 0.07).astype(int).to_numpy()
        - 0.05 * (X["X2"] < -0.69).astype(int).to_numpy()
        - 0.08 * X["C3"].isin(["1", "13", "14"]).astype(int).to_numpy()
        + gamma[X["schoolid"] - 1]
    )

    Yc = W * (Y - ite) + (1 - W) * ite

    Y1 = np.zeros_like(Y)
    Y1[W == 1] = Y[W == 1]
    Y1[W == 0] = Yc[W == 0]
    Y0 = np.zeros_like(Y)
    Y0[W == 1] = Yc[W == 1]
    Y0[W == 0] = Y[W == 0]

    ps = LogisticRegression(solver="lbfgs").fit(X, W).predict_proba(X)[:, 1]

    return Y, X.to_numpy(), W, ite, ps, Y0, Y1


acic2016_path_dir = "../data/ACIC2016"


def get_acic_2016(setting=1, sim_nb=1, all_sim=False):
    # There are 77 settings in total
    # 1 <= setting <= 77
    acic2016_files = os.listdir(f"{acic2016_path_dir}/{setting}")
    X = pd.read_csv(f"{acic2016_path_dir}/x.csv")
    X = pd.get_dummies(
        X,
        columns=[
            "x_2",
            "x_21",
            "x_24",
        ],
    ).to_numpy()
    if all_sim:
        X = []
        y = []
        W = []
        y0 = []
        y1 = []
        mu0 = []
        mu1 = []
        ite = []
        cate = []
        ps = []
        for file in acic2016_files:
            X.append(X.copy())
            df_sim = pd.read_csv(f"{acic2016_path_dir}/{setting}/{file}")
            W.append(df_sim["z"].to_numpy())
            y0.append(df_sim["y0"].to_numpy())
            y1.append(df_sim["y1"].to_numpy())
            mu0.append(df_sim["mu0"].to_numpy())
            mu1.append(df_sim["mu1"].to_numpy())
            y.append(W[-1] * y1[-1] + (1 - W[-1]) * y0[-1])
            ite.append(y1[-1] - y0[-1])
            cate.append(mu1[-1] - mu0[-1])
            ps.append(
                LogisticRegression(solver="lbfgs").fit(X[-1], W[-1]).predict_proba(X[-1])[:, 1]
            )
    else:
        df_sim = pd.read_csv(f"{acic2016_path_dir}/{setting}/{acic2016_files[sim_nb]}")
        W = df_sim["z"].to_numpy()
        y0 = df_sim["y0"].to_numpy()
        y1 = df_sim["y1"].to_numpy()
        mu0 = df_sim["mu0"].to_numpy()
        mu1 = df_sim["mu1"].to_numpy()
        y = W * y1 + (1 - W) * y0
        ite = y1 - y0
        cate = mu1 - mu0
        ps = LogisticRegression(solver="lbfgs").fit(X, W).predict_proba(X)[:, 1]
    return y, X, W, ps, ite, cate, mu0, mu1, y0, y1
