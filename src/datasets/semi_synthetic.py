"""
Inspired by the code from Alaa et al. (2023)
source: https://github.com/AlaaLab/conformal-metalearners
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn, optim

IHDP_TRAIN_DATASET = "data/IHDP/ihdp_npci_1-100.train.npz"
IHDP_TEST_DATASET = "data/IHDP/ihdp_npci_1-100.test.npz"
IHDP_TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
IHDP_TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"
PATH_dir = "data/NLSM/sim_data"
NLM_path = "data/NLSM/NLSM_data.csv"
acic2016_path_dir = "data/ACIC2016"
edu_dir = "data/EDU"


def convert(npz_train_file, npz_test_file, sim_nb=0, scale=None):
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

    x_realization_train, x_realization_test = x_train[:, :, sim_nb], x_test[:, :, sim_nb]
    t_realization_train, t_realization_test = t_train[:, sim_nb], t_test[:, sim_nb]
    yf_realization_train, yf_realization_test = yf_train[:, sim_nb], yf_test[:, sim_nb]
    ycf_realization_train, ycf_realization_test = ycf_train[:, sim_nb], ycf_test[:, sim_nb]
    mu1_realization_train, mu1_realization_test = mu1_train[:, sim_nb], mu1_test[:, sim_nb]
    mu0_realization_train, mu0_realization_test = mu0_train[:, sim_nb], mu0_test[:, sim_nb]

    model = LogisticRegression()
    model.fit(
        np.concatenate([x_realization_train, x_realization_test]),
        np.concatenate([t_realization_train, t_realization_test]),
    )
    df_train = pd.DataFrame(
        x_realization_train, columns=[f"X{j + 1}" for j in range(x_realization_train.shape[1])]
    )
    df_train["W"] = t_realization_train
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
    df_test["W"] = t_realization_test
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

        mu0_ = df["mu0"] / scale
        mu1_ = df["mu1"] / scale

        df["Y0"] = mu0_ + error_0
        df["Y1"] = mu1_ + error_1
        df["ITE"] = df["Y1"] - df["Y0"]
        df["CATE"] = mu1_ - mu0_

    return df[df["train_set"]].drop(columns=["train_set"]), df[~df["train_set"]].drop(
        columns=["train_set"]
    )


def get_ihdp_data(data_dir="../../", sim_nb=0):
    train = data_dir + IHDP_TRAIN_DATASET
    test = data_dir + IHDP_TEST_DATASET

    return convert(train, test, sim_nb=sim_nb)


def get_nlsm_sim_data(data_dir="../../"):
    df = pd.read_csv(data_dir + NLM_path)
    X = df.drop(columns=["Z", "Y"]).astype(float)
    W = df["Z"].to_numpy().astype(int)
    Y = df["Y"].to_numpy().astype(float)

    gamma = np.random.normal(0, 1, size=76) * 0.105

    # Based on Section 2 of Carvalho et al. (2019)
    ite = (
        0.228
        + 0.05 * (X["X1"] < 0.07).astype(int).to_numpy()
        - 0.05 * (X["X2"] < -0.69).astype(int).to_numpy()
        - 0.08 * X["C3"].isin(["1", "13", "14"]).astype(int).to_numpy()
        + gamma[X["schoolid"].astype(int) - 1]
    )

    Yc = W * (Y - ite) + (1 - W) * ite

    Y1 = np.zeros_like(Y)
    Y1[W == 1] = Y[W == 1]
    Y1[W == 0] = Yc[W == 0]
    Y0 = np.zeros_like(Y)
    Y0[W == 1] = Yc[W == 1]
    Y0[W == 0] = Y[W == 0]

    ps = LogisticRegression(solver="lbfgs").fit(X, W).predict_proba(X)[:, 1]
    df = pd.DataFrame(
        np.column_stack((X, W, Y, Y0, Y1, ps)),
        columns=[f"X{i}" for i in range(1, X.shape[1] + 1)] + ["W", "Y", "Y0", "Y1", "ps"],
    )
    # split the data into train and test
    df_train = df.sample(frac=0.5, random_state=42)
    df_test = df.drop(df_train.index)
    return df_train, df_test


def get_acic_2016(
    data_dir="../../",
    setting=1,
    sim_nb=1,
    all_sim=False,
):
    # There are 77 settings in total
    # 1 <= setting <= 77
    acic2016_files = os.listdir(f"{data_dir+acic2016_path_dir}/{setting}")
    X = pd.read_csv(f"{data_dir+acic2016_path_dir}/x.csv")
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
            df_sim = pd.read_csv(f"{data_dir+acic2016_path_dir}/{setting}/{file}")
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
        df_sim = pd.read_csv(f"{data_dir+acic2016_path_dir}/{setting}/{acic2016_files[sim_nb]}")
        W = df_sim["z"].to_numpy()
        y0 = df_sim["y0"].to_numpy()
        y1 = df_sim["y1"].to_numpy()
        mu0 = df_sim["mu0"].to_numpy()
        mu1 = df_sim["mu1"].to_numpy()
        y = W * y1 + (1 - W) * y0
        ite = y1 - y0
        cate = mu1 - mu0
        ps = LogisticRegression(solver="lbfgs").fit(X, W).predict_proba(X)[:, 1]

    df = pd.DataFrame(
        np.column_stack((X, W, y, y0, y1, ps)),
        columns=[f"X{i}" for i in range(1, X.shape[1] + 1)] + ["W", "Y", "Y0", "Y1", "ps"],
    )
    # split the data into train and test
    df_train = df.sample(frac=0.8, random_state=42)
    df_test = df.drop(df_train.index)
    return df_train, df_test


def get_edu(data_dir="../../", seed=1):
    """
    From: Zhou et al. (2022) "Estimating Potential Outcome Distributions
        with Collaborating Causal Networks"
    """
    df = pd.read_csv(f"{data_dir+edu_dir}/edu.csv")
    # covariance matrix
    xnew = np.array(df.drop(["treatment", "final"], axis=1))
    np.random.seed(seed)

    ori_trt = np.array(df["treatment"])

    # based on the observed outcome
    n1 = sum(ori_trt)
    x1 = torch.from_numpy(xnew[ori_trt == 1,]).float()
    y1 = torch.from_numpy(np.array(df["final"])[ori_trt == 1]).float().reshape(-1, 1)

    n0 = sum(1 - ori_trt)
    x0 = torch.from_numpy(xnew[ori_trt == 0,]).float()
    y0 = torch.from_numpy(np.array(df["final"])[ori_trt == 0]).float().reshape(-1, 1)

    device = torch.device("cpu")

    # The network
    class out_reg(nn.Module):
        def __init__(self):
            super().__init__()
            self.k1 = 32
            self.fc1 = nn.Linear(32, self.k1)
            self.fc2 = nn.Linear(self.k1, 1)
            self.relu = nn.Sigmoid()

        def forward(self, x):
            h1 = self.relu(self.fc1(x))
            out = self.fc2(h1)
            return out

    out0 = out_reg().to(device)
    out1 = out_reg().to(device)

    out0loss = nn.MSELoss()
    out1loss = nn.MSELoss()

    optimizer_0 = optim.Adam(out0.parameters())
    optimizer_1 = optim.Adam(out1.parameters())

    batch_size = 64

    loss0 = []
    loss1 = []

    for _ in range(1000):
        optimizer_0.zero_grad()
        optimizer_1.zero_grad()

        # grp0
        i0 = np.random.choice(n0, size=batch_size, replace=False)
        ys0 = y0[i0, :].to(device)
        xs0 = x0[i0, :].to(device)
        y0pred = out0(xs0)
        y0loss = out0loss(ys0, y0pred)
        y0loss.backward(retain_graph=False)
        optimizer_0.step()
        loss0.append(y0loss.cpu().item())

        # grp1
        i1 = np.random.choice(n1, size=batch_size, replace=False)
        ys1 = y1[i1, :].to(device)
        xs1 = x1[i1, :].to(device)
        y1pred = out1(xs1)
        y1loss = out1loss(ys1, y1pred)
        y1loss.backward(retain_graph=False)
        optimizer_1.step()
        loss1.append(y1loss.cpu().item())

    # prediction
    out0.eval()
    out1.eval()

    xnewtorch = torch.from_numpy(xnew).float()

    y0pred = out0(xnewtorch).detach().numpy()
    y1pred = out1(xnewtorch).detach().numpy()

    # create more variability to the ITE
    inflate = 1.5 / np.std((y1pred - y0pred).ravel())
    y0pred = y0pred * inflate
    y1pred = y1pred * inflate

    # generate the propensity model
    coef = np.random.uniform(-0.8, 0.8, 32)
    prop = 1 / (1 + np.exp(-xnew @ coef))
    # limit the propensity score to be between 0.05 and 0.95
    prop = np.clip(prop, 0.05, 0.95)
    trt = np.random.binomial(1, prop, len(prop))

    # generate potential outcome based on exponential distribution
    S = xnew[:, 23]  # label for whether attended school
    y0 = y0pred.ravel() + (2 - S) * np.random.normal(scale=0.5, size=len(y0pred))
    y1 = y1pred.ravel() + (2 - S) * np.random.exponential(scale=0.5, size=len(y1pred))
    y = y0 * (1 - trt) + y1 * trt

    ## Use the Beta Distribution to generate y(1) and Gaussian to generate y(0)
    y0 = y0pred.ravel() + (2 - S) * np.random.normal(scale=0.5, size=len(y0pred))
    y1 = y1pred.ravel() + (2 - S) * np.random.exponential(scale=0.5, size=len(y1pred))
    y = y0 * (1 - trt) + y1 * trt

    mu0 = y0pred.ravel()
    mu1 = y1pred.ravel() + (2 - S) * 0.5

    # the full data
    dt = np.c_[y, y0, y1, trt, mu0, mu1, S, xnew]

    # scramble the data
    newidx = np.random.choice(len(dt), len(dt), replace=False)
    prop = prop[newidx]
    dt = dt[newidx]

    # reserve 1,000 for testing
    testtmp = dt[:1000]

    y = testtmp[:, 0]
    y0 = testtmp[:, 1]
    y1 = testtmp[:, 2]
    W = testtmp[:, 3]
    xnew = testtmp[:, 7:]
    ps = prop[:1000]

    df_test = pd.DataFrame(
        np.column_stack((xnew, W, y, y0, y1, ps)),
        columns=[f"X{i}" for i in range(1, xnew.shape[1] + 1)] + ["W", "Y", "Y0", "Y1", "ps"],
    )

    # remove training data with propensity between 0.3-0.7 and keep only 4,500 training size
    # This was done in the paper however, we omit this step in the code because it
    # viloates the assumption of exchangeability
    traintmp = dt[1000:]
    proptmp = prop[1000:]
    # rmid=np.where(np.logical_and(proptmp<0.7,proptmp>0.3)) #id of removed observation
    # traintmp=np.delete(traintmp,rmid,axis=0)
    traintmp = traintmp[:4500]
    # proptmp=np.delete(proptmp,rmid)
    proptmp = proptmp[:4500]

    y = traintmp[:, 0]
    y0 = traintmp[:, 1]
    y1 = traintmp[:, 2]
    W = traintmp[:, 3]
    xnew = traintmp[:, 7:]
    ps = proptmp

    df_train = pd.DataFrame(
        np.column_stack((xnew, W, y, y0, y1, ps)),
        columns=[f"X{i}" for i in range(1, xnew.shape[1] + 1)] + ["W", "Y", "Y0", "Y1", "ps"],
    )
    return df_train, df_test
