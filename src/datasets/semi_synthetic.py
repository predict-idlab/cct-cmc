"""
Inspired by the code from Alaa et al. (2023)
source: https://github.com/AlaaLab/conformal-metalearners
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

IHDP_TRAIN_DATASET = "../../data/IHDP/ihdp_npci_1-100.train.npz"
IHDP_TEST_DATASET = "../../data/IHDP/ihdp_npci_1-100.test.npz"
IHDP_TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
IHDP_TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"
PATH_dir = "../../data/NLSM/sim_data"
NLM_path = "../../data/NLSM/NLSM_data.csv"
acic2016_path_dir = "../../data/ACIC2016"


def convert(npz_train_file, npz_test_file, scale=None):
    npz_train_data = np.load(npz_train_file)
    npz_test_data = np.load(npz_test_file)

    scales = []

    x_train, t_train, yf_train, ycf_train, mu0_train, mu1_train = (
        npz_train_data["x"],
        npz_train_data["t"],
        npz_train_data["yf"],
        npz_train_data["ycf"],
        npz_train_data["mu0"],
        npz_train_data["mu1"],
    )

    x_test, t_test, yf_test, ycf_test, mu0_test, mu1_test = (
        npz_test_data["x"],
        npz_test_data["t"],
        npz_test_data["yf"],
        npz_test_data["ycf"],
        npz_test_data["mu0"],
        npz_test_data["mu1"],
    )

    num_realizations = x_train.shape[2]

    dataframes_train = []
    dataframes_test = []

    for i in range(num_realizations):
        x_realization_train, x_realization_test = x_train[:, :, i], x_test[:, :, i]
        t_realization_train, t_realization_test = t_train[:, i], t_test[:, i]
        yf_realization_train, yf_realization_test = yf_train[:, i], yf_test[:, i]
        ycf_realization_train, ycf_realization_test = ycf_train[:, i], ycf_test[:, i]
        mu1_realization_train, mu1_realization_test = mu1_train[:, i], mu1_test[:, i]
        mu0_realization_train, mu0_realization_test = mu0_train[:, i], mu0_test[:, i]

        model = LogisticRegression()
        model.fit(
            np.concatenate([x_realization_train, x_realization_test]),
            np.concatenate([t_realization_train, t_realization_test]),
        )
        df_train = pd.DataFrame(
            x_realization_train, columns=[f"X{j + 1}" for j in range(x_realization_train.shape[1])]
        )
        df_train["T"] = t_realization_train
        df_train["Y"] = yf_realization_train
        df_train["Y_cf"] = ycf_realization_train
        df_train["Y1"] = yf_realization_train * t_realization_train + ycf_realization_train * (
            1 - t_realization_train
        )
        df_train["Y0"] = ycf_realization_train * t_realization_train + yf_realization_train * (
            1 - t_realization_train
        )
        df_train["ITE"] = df_train["Y1"] - df_train["Y0"]
        df_train["ps"] = model.predict_proba(x_realization_train)[:, 1]
        df_train["CATE"] = mu1_realization_train - mu0_realization_train
        df_train["mu0"] = mu0_realization_train
        df_train["mu1"] = mu1_realization_train

        df_test = pd.DataFrame(
            x_realization_test, columns=[f"X{j + 1}" for j in range(x_realization_test.shape[1])]
        )
        df_test["T"] = t_realization_test
        df_test["Y"] = yf_realization_test
        df_test["Y_cf"] = ycf_realization_test
        df_test["Y1"] = yf_realization_test * t_realization_test + ycf_realization_test * (
            1 - t_realization_test
        )
        df_test["Y0"] = ycf_realization_test * t_realization_test + yf_realization_test * (
            1 - t_realization_test
        )
        df_test["ITE"] = df_test["Y1"] - df_test["Y0"]
        df_test["ps"] = model.predict_proba(x_realization_test)[:, 1]
        df_test["CATE"] = mu1_realization_test - mu0_realization_test
        df_test["mu0"] = mu0_realization_test
        df_test["mu1"] = mu1_realization_test

        df_train["train_set"] = True
        df_test["train_set"] = False
        df = pd.concat([df_train, df_test])

        sd_cate = np.sqrt((np.array(df["CATE"])).var())

        if scale is None:
            if sd_cate > 1:
                error_0 = np.array(df["Y0"]) - df["mu0"]
                error_1 = np.array(df["Y1"]) - df["mu1"]

                mu0_ = df["mu0"] / sd_cate
                mu1_ = df["mu1"] / sd_cate

                scales.append(sd_cate)

                df["Y0"] = mu0_ + error_0
                df["Y1"] = mu1_ + error_1
                df["ITE"] = df["Y1"] - df["Y0"]
                df["CATE"] = mu1_ - mu0_

            else:
                scales.append(1)

        elif scale is not None:
            # test data
            error_0 = np.array(df["Y0"]) - df["mu0"]
            error_1 = np.array(df["Y1"]) - df["mu1"]

            mu0_ = df["mu0"] / scale[i]
            mu1_ = df["mu1"] / scale[i]

            df["Y0"] = mu0_ + error_0
            df["Y1"] = mu1_ + error_1
            df["ITE"] = df["Y1"] - df["Y0"]
            df["CATE"] = mu1_ - mu0_

        dataframes_train.append(df[df["train_set"]])
        dataframes_test.append(df[~df["train_set"]])
    return dataframes_train, dataframes_test, scales


def get_ihdp_data():
    train = IHDP_TRAIN_DATASET
    test = IHDP_TEST_DATASET

    train_data, test_data, scale = convert(train, test)

    return train_data, test_data


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
