"""Synthetic data generation for the Alaa et al. 2023 paper.
source: https://github.com/AlaaLab/conformal-metalearners
"""

import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy.special import erfinv
from scipy.stats import beta, norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def generate_data(n, d, gamma, alpha, nexps, correlated=False, heteroscedastic=True):
    def correlated_covariates(n, d):
        rho = 0.9
        X = np.random.randn(n, d)
        fac = np.random.randn(n, d)
        X = X * np.sqrt(1 - rho) + fac * np.sqrt(rho)

        return norm.cdf(X)

    datasets = []

    for _ in range(nexps):
        if correlated == False and heteroscedastic == False:
            X = np.random.uniform(0, 1, (n, d))
            tau = (2 / (1 + np.exp(-12 * (X[:, 0] - 0.5)))) * (
                2 / (1 + np.exp(-12 * (X[:, 1] - 0.5)))
            )
            tau = tau.reshape((-1,))
            tau_0 = gamma * tau

            std = np.ones(X.shape[0])
            ps = (1 + beta.cdf(X[:, 0], 2, 4)) / 4
            errdist = np.random.normal(0, 1, n)

            err_0 = np.random.normal(0, 1, n)

            Y0 = tau_0 + std * err_0  # np.zeros(n)
            Y1 = tau + std * errdist
            T = np.random.uniform(size=n) < ps
            Y = Y0.copy()
            Y[T] = Y1[T]

            # Pseudolabel calculation
            A = T
            pi = ps
            xi = (A - pi) * Y / (pi * (1 - pi))

            data = np.column_stack((X, T, Y))

            column_names = [f"X{i}" for i in range(1, d + 1)] + ["W", "Y"]
            df = pd.DataFrame(data, columns=column_names)
            df["xi"] = xi
            df["ps"] = np.array(ps).reshape((-1,))
            df["Y1"] = Y1.reshape((-1,))
            df["Y0"] = Y0.reshape((-1,))
            df["CATE"] = tau - tau_0
            df["width"] = np.mean(
                np.sqrt(2) * (np.sqrt(2) * std) * erfinv(2 * (1 - (alpha / 2)) - 1) * 2
            )

            datasets.append(df)

        elif correlated == False and heteroscedastic == True:
            # Generate dataset with heteroscedastic errors and independent covariates
            X = np.random.uniform(0, 1, (n, d))

            tau = (2 / (1 + np.exp(-12 * (X[:, 0] - 0.5)))) * (
                2 / (1 + np.exp(-12 * (X[:, 1] - 0.5)))
            )
            tau = tau.reshape((-1,))

            tau_0 = gamma * tau
            std = -np.log(X[:, 0] + 1e-9)
            ps = (1 + beta.cdf(X[:, 0], 2, 4)) / 4

            errdist = np.random.normal(0, 1, n)
            err_0 = np.random.normal(0, 1, n)

            Y0 = tau_0 + 1 * err_0
            Y1 = tau + np.sqrt(std) * errdist
            T = np.random.uniform(size=n) < ps
            Y = Y0.copy()
            Y[T] = Y1[T]

            # Pseudolabel calculation
            A = T
            pi = ps
            xi = (A - pi) * Y / (pi * (1 - pi))

            # Stratify by conditional variance, CATE
            n_percentiles = 100
            cate = Y1 - Y0
            conditional_var = std**2 * (1 - pi) + pi * std**2 * (1 + tau) ** 2
            cate_percentiles = np.zeros(n)
            var_percentiles = np.zeros(n)

            for j in range(n):
                cate_percentiles[j] = np.searchsorted(
                    np.percentile(cate, np.linspace(0, 100, n_percentiles + 1)), cate[j]
                )
                var_percentiles[j] = np.searchsorted(
                    np.percentile(conditional_var, np.linspace(0, 100, n_percentiles + 1)),
                    conditional_var[j],
                )

            data = np.column_stack((X, T, Y))
            column_names = [f"X{i}" for i in range(1, d + 1)] + ["W", "Y"]
            df = pd.DataFrame(data, columns=column_names)
            df["xi"] = xi
            df["cate_percentile"] = cate_percentiles / n_percentiles
            df["var_percentile"] = var_percentiles / n_percentiles

            df["ps"] = np.array(ps).reshape((-1,))
            df["Y1"] = Y1.reshape((-1,))
            df["Y0"] = Y0.reshape((-1,))
            df["CATE"] = tau - tau_0
            df["width"] = np.mean(
                np.sqrt(2) * (np.sqrt(2) * std) * erfinv(2 * (1 - (alpha / 2)) - 1) * 2
            )

            datasets.append(df)

        elif correlated == True and heteroscedastic == False:
            # Generate dataset with homoscedastic errors and correlated covariates
            X = correlated_covariates(n, d)
            tau = (2 / (1 + np.exp(-12 * (X[:, 0] - 0.5)))) * (
                2 / (1 + np.exp(-12 * (X[:, 1] - 0.5)))
            )
            std = np.ones(X.shape[0])
            ps = (1 + beta.cdf(X[:, 0], 2, 4)) / 4
            errdist = np.random.normal(0, 1, n)
            Y0 = np.zeros(n)
            Y1 = tau + std * errdist
            T = np.random.uniform(size=n) < ps
            Y = Y0.copy()
            Y[T] = Y1[T]

            # Pseudolabel calculation
            A = T
            pi = ps
            xi = (A - pi) * Y / (pi * (1 - pi))

            data = np.column_stack((X, T, Y))
            column_names = [f"X{i}" for i in range(1, d + 1)] + ["W", "Y"]
            df = pd.DataFrame(data, columns=column_names)
            df["xi"] = xi
            df["ps"] = np.array(ps).reshape((-1,))
            df["Y1"] = Y1.reshape((-1,))
            df["Y0"] = Y0.reshape((-1,))

            datasets.append(df)

        elif correlated == True and heteroscedastic == True:
            # Generate dataset with heteroscedastic errors and correlated covariates
            X = correlated_covariates(n, d)
            tau = (2 / (1 + np.exp(-12 * (X[:, 0] - 0.5)))) * (
                2 / (1 + np.exp(-12 * (X[:, 1] - 0.5)))
            )

            tau_0 = 2 / (1 + np.exp(-12 * (X[:, 1] - 0.5)))

            std = -np.log(X[:, 0] + 1e-9)
            ps = (1 + beta.cdf(X[:, 0], 2, 4)) / 4
            errdist = np.random.normal(0, 1, n)

            Y0 = tau_0  # np.zeros(n)
            Y1 = tau + std * errdist
            T = np.random.uniform(size=n) < ps
            Y = Y0.copy()
            Y[T] = Y1[T]

            # Pseudolabel calculation
            A = T
            pi = ps
            xi = (A - pi) * Y / (pi * (1 - pi))

            # Stratify by conditional variance, CATE
            n_percentiles = 100
            cate = Y1 - Y0
            conditional_var = std**2 * (1 - pi) + pi * std**2 * (1 + tau) ** 2
            cate_percentiles = np.zeros(n)
            var_percentiles = np.zeros(n)

            for j in range(n):
                cate_percentiles[j] = np.searchsorted(
                    np.percentile(cate, np.linspace(0, 100, n_percentiles + 1)), cate[j]
                )
                var_percentiles[j] = np.searchsorted(
                    np.percentile(conditional_var, np.linspace(0, 100, n_percentiles + 1)),
                    conditional_var[j],
                )

            data = np.column_stack((X, T, Y))
            column_names = [f"X{i}" for i in range(1, d + 1)] + ["W", "Y"]
            df = pd.DataFrame(data, columns=column_names)
            df["xi"] = xi
            df["cate_percentile"] = cate_percentiles / n_percentiles
            df["var_percentile"] = var_percentiles / n_percentiles
            df["ps"] = np.array(ps).reshape((-1,))
            df["Y1"] = Y1.reshape((-1,))
            df["Y0"] = Y0.reshape((-1,))
            df["CATE"] = tau - tau_0

            datasets.append(df)

    return datasets


def get_data(n, d, gamma, alpha, correlated=False, heteroscedastic=True, test_size=0.6):
    df = generate_data(n, d, gamma, alpha, 1, correlated, heteroscedastic)[0]
    df_train, df_test = train_test_split(df, test_size=test_size)
    return df_train, df_test