import logging
import os
import warnings

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.metrics import calculate_dispersion_normal, crps_normal

# Suppress warnings and logging
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class DKLITE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_hidden=50,
        num_layers=2,
        learning_rate=0.001,
        reg_var=1.0,
        reg_rec=1.0,
    ):
        super(DKLITE, self).__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.size_z = num_hidden
        self.reg_var = reg_var
        self.reg_rec = reg_rec

        # Encoder
        self.encoder = self._build_network(
            input_dim, num_hidden, num_layers, self.size_z, final_activation=True
        )
        # Decoder
        self.decoder = self._build_network(
            self.size_z, num_hidden, num_layers, input_dim, final_activation=False
        )

        # Initialize weights
        self._initialize_weights()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Stored parameters
        self.ker_inv_0 = None
        self.ker_inv_1 = None
        self.mean_gp_0 = None
        self.mean_gp_1 = None
        self.mean_0 = None
        self.mean_1 = None
        self.Z_train = None

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"DKLITE is using device: {self.device}")
        self.to(self.device)

    def _build_network(self, in_dim, hidden, layers, out_dim, final_activation):
        modules = []
        modules.append(nn.Linear(in_dim, hidden))
        modules.append(nn.ELU())

        for _ in range(layers):
            modules.append(nn.Linear(hidden, hidden))
            modules.append(nn.ELU())

        modules.append(nn.Linear(hidden, out_dim))
        if final_activation:
            modules.append(nn.ELU())

        return nn.Sequential(*modules)

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=1.0)
            elif "bias" in name:
                nn.init.zeros_(param)

    def GP_NN(self, Y_f, Z_f):
        beta = torch.ones(1, 1, dtype=torch.float32, device=self.device)
        lam = 1000 * torch.ones(1, 1, dtype=torch.float32, device=self.device)
        r = beta / lam

        phi_phi = torch.matmul(Z_f.T, Z_f)
        Ker = r * phi_phi + torch.eye(Z_f.size(1), dtype=torch.float32, device=self.device)
        L_matrix = torch.linalg.cholesky(Ker)
        L_inv_reduce = torch.linalg.solve_triangular(
            L_matrix,
            torch.eye(L_matrix.size(0), dtype=torch.float32, device=self.device),
            upper=False,
        )
        L_y = torch.matmul(L_inv_reduce, torch.matmul(Z_f.T, Y_f))
        ker_inv = torch.matmul(L_inv_reduce.T, L_inv_reduce) / lam
        mean = r * torch.matmul(L_inv_reduce.T, L_y)
        term1 = -torch.mean(L_y**2)
        return term1, ker_inv, mean

    def fit(self, X, Y, T, num_iteration=1000):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32, device=self.device)
        T = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=self.device)

        loss_list = []
        for _ in tqdm(range(num_iteration)):
            self.optimizer.zero_grad()

            # Forward pass
            Z_train = self.encoder(X)
            mask_0 = (T < 0.5).flatten()
            mask_1 = (T >= 0.5).flatten()

            Z_0 = Z_train[mask_0]
            Y_0 = Y[mask_0]
            Z_1 = Z_train[mask_1]
            Y_1 = Y[mask_1]

            if Y_0.size(0) == 0 or Y_1.size(0) == 0:
                continue

            mean_0 = torch.mean(Y_0)
            mean_1 = torch.mean(Y_1)
            Y_0c = Y_0 - mean_0
            Y_1c = Y_1 - mean_1

            ml0, ker_inv_0, mean_gp_0 = self.GP_NN(Y_0c, Z_0)
            ml1, ker_inv_1, mean_gp_1 = self.GP_NN(Y_1c, Z_1)

            var_0 = torch.diagonal(torch.matmul(Z_1, torch.matmul(ker_inv_0, Z_1.T))).mean()
            var_1 = torch.diagonal(torch.matmul(Z_0, torch.matmul(ker_inv_1, Z_0.T))).mean()

            X_recon = self.decoder(Z_train)
            loss_rec = torch.mean(torch.sum((X - X_recon) ** 2, dim=1))

            total_loss = ml0 + ml1 + self.reg_var * (var_0 + var_1) + self.reg_rec * loss_rec
            total_loss.backward()
            self.optimizer.step()
            loss_list.append(total_loss.item())

            if len(loss_list) > 50 and np.abs(
                np.mean(loss_list[-10:]) - np.mean(loss_list[-50:-10])
            ) < np.std(loss_list[-50:-10]):
                break

        # Store final parameters
        with torch.no_grad():
            Z_train = self.encoder(X)
            mask_0 = (T < 0.5).flatten()
            mask_1 = (T >= 0.5).flatten()

            Y_0 = Y[mask_0]
            Y_1 = Y[mask_1]
            self.mean_0 = torch.mean(Y_0) if Y_0.size(0) > 0 else torch.tensor(0.0)
            self.mean_1 = torch.mean(Y_1) if Y_1.size(0) > 0 else torch.tensor(0.0)

            Z_0 = Z_train[mask_0]
            Z_1 = Z_train[mask_1]
            Y_0c = Y_0 - self.mean_0 if Y_0.size(0) > 0 else Y_0
            Y_1c = Y_1 - self.mean_1 if Y_1.size(0) > 0 else Y_1

            if Y_0c.size(0) > 0:
                _, self.ker_inv_0, self.mean_gp_0 = self.GP_NN(Y_0c, Z_0)
            if Y_1c.size(0) > 0:
                _, self.ker_inv_1, self.mean_gp_1 = self.GP_NN(Y_1c, Z_1)

            self.Z_train = Z_train

    def pred(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            Z_test = self.encoder(X)

            pred0 = Z_test @ (self.mean_gp_0 if self.mean_gp_0 is not None else 0.0) + self.mean_0
            pred1 = Z_test @ (self.mean_gp_1 if self.mean_gp_1 is not None else 0.0) + self.mean_1

            return torch.cat([pred0, pred1], dim=1).cpu().numpy()

    def element_var(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            Z_test = self.encoder(X)

            var0 = torch.diagonal(
                Z_test @ (self.ker_inv_0 if self.ker_inv_0 is not None else 0.0) @ Z_test.T
            )
            var1 = torch.diagonal(
                Z_test @ (self.ker_inv_1 if self.ker_inv_1 is not None else 0.0) @ Z_test.T
            )

            return var0.cpu().numpy(), var1.cpu().numpy()

    def evaluate(self, X, Y0, Y1, alpha=0.05, return_p_values=False):
        Y_hat = self.pred(X)
        mu0 = Y_hat[:, 0:1]
        mu1 = Y_hat[:, 1:2]
        var0, var1 = self.element_var(X)

        mu_ite = mu1 - mu0
        var_ite = var0 + var1

        # Point prediction metrics
        rmse_y0 = np.sqrt(np.mean((mu0 - Y0) ** 2))
        rmse_y1 = np.sqrt(np.mean((mu1 - Y1) ** 2))
        rmse_ite = np.sqrt(np.mean((mu_ite - (Y1 - Y0)) ** 2))

        # Interval prediction
        def get_coverage(mu, var, y):
            std = np.sqrt(var)
            upper = mu + std * scipy.stats.norm.ppf(1 - alpha / 2)
            lower = mu - std * scipy.stats.norm.ppf(1 - alpha / 2)
            coverage = np.mean((y >= lower) & (y <= upper))
            efficiency = np.mean(upper - lower)
            return coverage, efficiency

        coverage_y0, efficiency_y0 = get_coverage(mu0, var0, Y0)
        coverage_y1, efficiency_y1 = get_coverage(mu1, var1, Y1)
        coverage_ite, efficiency_ite = get_coverage(mu_ite, var_ite, Y1 - Y0)

        # Distribution prediction
        crps_y0 = crps_normal(Y0, mu=mu0, sigma=np.sqrt(var0))
        crps_y1 = crps_normal(Y1, mu=mu1, sigma=np.sqrt(var1))
        crps_ite = crps_normal(Y1 - Y0, mu=mu_ite, sigma=np.sqrt(var_ite))

        ll_y0 = np.mean(scipy.stats.norm.logpdf(Y0, loc=mu0, scale=np.sqrt(var0)))
        ll_y1 = np.mean(scipy.stats.norm.logpdf(Y1, loc=mu1, scale=np.sqrt(var1)))
        ll_ite = np.mean(scipy.stats.norm.logpdf(Y1 - Y0, loc=mu_ite, scale=np.sqrt(var_ite)))

        # Dispersion
        dispersion_y0, p_values_y0 = calculate_dispersion_normal(
            Y0, mu=mu0, sigma=np.sqrt(var0), return_p_values=True
        )
        dispersion_y1, p_values_y1 = calculate_dispersion_normal(
            Y1, mu=mu1, sigma=np.sqrt(var1), return_p_values=True
        )
        dispersion_ite, p_values_ite = calculate_dispersion_normal(
            Y1 - Y0, mu=mu_ite, sigma=np.sqrt(var_ite), return_p_values=True
        )

        results = {
            "rmse_y0": rmse_y0,
            "rmse_y1": rmse_y1,
            "rmse_ite": rmse_ite,
            "coverage_y0": coverage_y0,
            "coverage_y1": coverage_y1,
            "coverage_ite": coverage_ite,
            "efficiency_y0": efficiency_y0,
            "efficiency_y1": efficiency_y1,
            "efficiency_ite": efficiency_ite,
            "crps_y0": crps_y0,
            "crps_y1": crps_y1,
            "crps_ite": crps_ite,
            "ll_y0": ll_y0,
            "ll_y1": ll_y1,
            "ll_ite": ll_ite,
            "dispersion_y0": dispersion_y0,
            "dispersion_y1": dispersion_y1,
            "dispersion_ite": dispersion_ite,
        }
        if return_p_values:
            results["p_values_y0"] = [p_values_y0]
            results["p_values_y1"] = [p_values_y1]
            results["p_values_ite"] = [p_values_ite]
        return results
