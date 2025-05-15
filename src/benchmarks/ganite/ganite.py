"""GANITE Codebase (PyTorch Version)."""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.metrics import calculate_dispersion, crps, loglikelihood

torch.manual_seed(42)


class GcfModel(nn.Module):
    """Counterfactual generator model."""

    def __init__(self, input_size, h_dim):
        super().__init__()
        self.dense1 = nn.Linear(input_size + 1 + 1 + 2, h_dim)  # x(20) + t(1) + yf(1) + z(2)
        self.dense2 = nn.Linear(h_dim, h_dim)
        self.dense20 = nn.Linear(h_dim, h_dim)
        self.dense21 = nn.Linear(h_dim, h_dim)
        self.ycf0 = nn.Linear(h_dim, 1)
        self.ycf1 = nn.Linear(h_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t, yf, z):
        inputcf = torch.cat([x, t, yf, z], dim=1)
        hidden = F.relu(self.dense1(inputcf))
        hidden2 = F.relu(self.dense2(hidden))
        hidden20 = F.relu(self.dense20(hidden2))
        hidden21 = F.relu(self.dense21(hidden2))
        ycf0 = self.ycf0(hidden20)
        ycf1 = self.ycf1(hidden21)
        return torch.cat([ycf0, ycf1], dim=1)


class DcfModel(nn.Module):
    """Counterfactual discriminator model."""

    def __init__(self, input_size, h_dim):
        super().__init__()
        self.dense1 = nn.Linear(input_size + 1 + 1, h_dim)  # x(20) + ycf0(1) + ycf1(1)
        self.dense2 = nn.Linear(h_dim, h_dim)
        self.dlogit = nn.Linear(h_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, ycf0, ycf1):
        inputd = torch.cat([x, ycf0, ycf1], dim=1)
        hidden = F.relu(self.dense1(inputd))
        hidden2 = F.relu(self.dense2(hidden))
        return self.dlogit(hidden2)


class GiteModel(nn.Module):
    """ITE generator model."""

    def __init__(self, input_size, h_dim):
        super().__init__()
        self.dense1 = nn.Linear(input_size + 2, h_dim)  # x(20) + z(2)
        self.dense2 = nn.Linear(h_dim, h_dim)
        self.dense20 = nn.Linear(h_dim, h_dim)
        self.dense21 = nn.Linear(h_dim, h_dim)
        self.yite0 = nn.Linear(h_dim, 1)
        self.yite1 = nn.Linear(h_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, z):
        inputite = torch.cat([x, z], dim=1)
        hidden = F.relu(self.dense1(inputite))
        hidden2 = F.relu(self.dense2(hidden))
        hidden20 = F.relu(self.dense20(hidden2))
        hidden21 = F.relu(self.dense21(hidden2))
        yite0 = self.yite0(hidden20)
        yite1 = self.yite1(hidden21)
        return torch.cat([yite0, yite1], dim=1)


class DiteModel(nn.Module):
    """ITE discriminator model."""

    def __init__(self, input_size, h_dim):
        super().__init__()
        self.dense1 = nn.Linear(input_size + 2, h_dim)  # x(20) + ypair(2)
        self.dense2 = nn.Linear(h_dim, h_dim)
        self.dite = nn.Linear(h_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, ypair):
        inputdite = torch.cat([x, ypair], dim=1)
        hidden = F.relu(self.dense1(inputdite))
        hidden2 = F.relu(self.dense2(hidden))
        return self.dite(hidden2)


class GANITE:
    """GANITE model for individualized treatment effect estimation."""

    def __init__(self, input_size, h_dim, batch_size, iterations, alpha, beta):
        self.input_size = input_size
        self.h_dim = h_dim
        self.batch_size = batch_size
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"GANITE is using device: {self.device}")

        # Initialize models
        self.Gcf = GcfModel(input_size, h_dim).to(self.device)
        self.Dcf = DcfModel(input_size, h_dim).to(self.device)
        self.Gite = GiteModel(input_size, h_dim).to(self.device)
        self.Dite = DiteModel(input_size, h_dim).to(self.device)

        # Initialize optimizers
        self.G_optimizer = torch.optim.Adam(self.Gcf.parameters())
        self.D_optimizer = torch.optim.Adam(self.Dcf.parameters())
        self.Gite_optimizer = torch.optim.Adam(self.Gite.parameters())
        self.Dite_optimizer = torch.optim.Adam(self.Dite.parameters())

    def fit(self, X, Y, T):
        """Train GANITE model."""
        X = torch.FloatTensor(X).to(self.device)
        Y = torch.FloatTensor(Y.reshape(-1, 1)).to(self.device)
        T = torch.FloatTensor(T.reshape(-1, 1)).to(self.device)
        dataset_size = X.size(0)

        # Training counterfactual GAN
        for _ in tqdm(range(self.iterations), desc="Training Counterfactual GAN"):
            # Train Discriminator
            for _ in range(2):
                idx = np.random.randint(0, dataset_size, self.batch_size)
                x_batch = X[idx]
                t_batch = T[idx]
                yf_batch = Y[idx]
                zcf_batch = (
                    torch.rand(self.batch_size, 2, device=self.device) * 2 - 1
                )  # Uniform [-1,1]

                # Generate counterfactuals
                ycf = self.Gcf(x_batch, t_batch, yf_batch, zcf_batch)
                ycf0, ycf1 = ycf[:, 0:1], ycf[:, 1:2]
                ycf0_ = ycf0 * t_batch + yf_batch * (1 - t_batch)
                ycf1_ = ycf1 * (1 - t_batch) + yf_batch * t_batch

                # Discriminator loss
                self.D_optimizer.zero_grad()
                d_logits = self.Dcf(x_batch, ycf0_, ycf1_)
                d_loss = F.binary_cross_entropy_with_logits(d_logits, t_batch, reduction="mean")
                d_loss.backward()
                self.D_optimizer.step()

            # Train Generator
            idx = np.random.randint(0, dataset_size, self.batch_size)
            x_batch = X[idx]
            t_batch = T[idx]
            yf_batch = Y[idx]
            zcf_batch = torch.rand(self.batch_size, 2, device=self.device) * 2 - 1

            self.G_optimizer.zero_grad()
            ycf = self.Gcf(x_batch, t_batch, yf_batch, zcf_batch)
            ycf0, ycf1 = ycf[:, 0:1], ycf[:, 1:2]
            ycf0_ = ycf0 * t_batch + yf_batch * (1 - t_batch)
            ycf1_ = ycf1 * (1 - t_batch) + yf_batch * t_batch
            d_logits = self.Dcf(x_batch, ycf0_, ycf1_)

            # Generator loss components
            g_loss_gan = -F.binary_cross_entropy_with_logits(d_logits, t_batch, reduction="mean")
            yf_pred = ycf0 * (1 - t_batch) + ycf1 * t_batch
            mse_loss = F.mse_loss(yf_pred, yf_batch)
            g_loss = g_loss_gan + self.alpha * mse_loss

            g_loss.backward()
            self.G_optimizer.step()

        # Training ITE GAN
        for _ in tqdm(range(self.iterations), desc="Training ITE GAN"):
            # Train Discriminator
            for _ in range(2):
                idx = np.random.randint(0, dataset_size, self.batch_size)
                x_batch = X[idx]
                t_batch = T[idx]
                yf_batch = Y[idx]
                zcf_batch = torch.rand(self.batch_size, 2, device=self.device) * 2 - 1
                zite_batch = torch.rand(self.batch_size, 2, device=self.device) * 2 - 1

                # Generate real and fake samples
                with torch.no_grad():
                    ycf_real = self.Gcf(x_batch, t_batch, yf_batch, zcf_batch)
                ycf_fake = self.Gite(x_batch, zite_batch)

                # Discriminator loss
                self.Dite_optimizer.zero_grad()
                d_real = self.Dite(x_batch, ycf_real)
                d_fake = self.Dite(x_batch, ycf_fake)
                d_loss_real = F.binary_cross_entropy_with_logits(
                    d_real, torch.ones_like(d_real), reduction="mean"
                )
                d_loss_fake = F.binary_cross_entropy_with_logits(
                    d_fake, torch.zeros_like(d_fake), reduction="mean"
                )
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.Dite_optimizer.step()

            # Train Generator
            idx = np.random.randint(0, dataset_size, self.batch_size)
            x_batch = X[idx]
            zite_batch = torch.rand(self.batch_size, 2, device=self.device) * 2 - 1
            yf_batch = Y[idx]
            t_batch = T[idx]
            zcf_batch = torch.rand(self.batch_size, 2, device=self.device) * 2 - 1

            self.Gite_optimizer.zero_grad()
            ycf_fake = self.Gite(x_batch, zite_batch)
            d_fake = self.Dite(x_batch, ycf_fake)

            # Generator loss components
            g_loss_gan = -F.binary_cross_entropy_with_logits(
                d_fake, torch.zeros_like(d_fake, device=self.device), reduction="mean"
            )
            with torch.no_grad():
                ycf_real = self.Gcf(x_batch, t_batch, yf_batch, zcf_batch)
            mse_loss = F.mse_loss(ycf_fake, ycf_real)
            g_loss = g_loss_gan + self.beta * mse_loss

            g_loss.backward()
            self.Gite_optimizer.step()

    def sample_y0_y1(self, X, n_samples=500):
        """Generate potential outcomes samples."""
        X = torch.FloatTensor(X).to(self.device)
        pred_y0 = np.zeros((len(X), n_samples))
        pred_y1 = np.zeros((len(X), n_samples))

        with torch.no_grad():
            for i in range(len(X)):
                x_tiled = X[i : i + 1].repeat(n_samples, 1)
                zite = torch.rand(n_samples, 2, device=self.device) * 2 - 1
                samples = self.Gite(x_tiled, zite)
                pred_y0[i] = samples[:, 0].cpu().numpy().ravel()
                pred_y1[i] = samples[:, 1].cpu().numpy().ravel()
        return pred_y0, pred_y1

    def evaluate(self, X, Y0, Y1, alpha=0.05, n_samples=500, return_p_values=False):
        """Evaluate model performance."""
        y0_pred, y1_pred = self.sample_y0_y1(X, n_samples)
        ite_pred = y1_pred - y0_pred

        # Point prediction metrics
        rmse_y0 = np.sqrt(np.mean((y0_pred.mean(axis=1) - Y0) ** 2))
        rmse_y1 = np.sqrt(np.mean((y1_pred.mean(axis=1) - Y1) ** 2))
        rmse_ite = np.sqrt(np.mean((ite_pred.mean(axis=1) - (Y1 - Y0)) ** 2))

        # Interval prediction metrics
        y0_upper = np.percentile(y0_pred, 100 * (1 - alpha / 2), axis=1)
        y0_lower = np.percentile(y0_pred, 100 * alpha / 2, axis=1)
        coverage_y0 = np.mean((Y0 >= y0_lower) & (Y0 <= y0_upper))
        efficiency_y0 = np.mean(y0_upper - y0_lower)

        y1_upper = np.percentile(y1_pred, 100 * (1 - alpha / 2), axis=1)
        y1_lower = np.percentile(y1_pred, 100 * alpha / 2, axis=1)
        coverage_y1 = np.mean((Y1 >= y1_lower) & (Y1 <= y1_upper))
        efficiency_y1 = np.mean(y1_upper - y1_lower)

        ite_upper = np.percentile(ite_pred, 100 * (1 - alpha / 2), axis=1)
        ite_lower = np.percentile(ite_pred, 100 * alpha / 2, axis=1)
        coverage_ite = np.mean(((Y1 - Y0) >= ite_lower) & ((Y1 - Y0) <= ite_upper))
        efficiency_ite = np.mean(ite_upper - ite_lower)

        # Distribution prediction metrics
        crps_y0 = crps(Y0, y0_pred)
        crps_y1 = crps(Y1, y1_pred)
        crps_ite = crps(Y1 - Y0, ite_pred)

        ll_y0 = loglikelihood(Y0, y0_pred)
        ll_y1 = loglikelihood(Y1, y1_pred)
        ll_ite = loglikelihood(Y1 - Y0, ite_pred)

        # Dispersion metrics
        dispersion_y0, p_values_y0 = calculate_dispersion(Y0, y0_pred, return_p_values=True)
        dispersion_y1, p_values_y1 = calculate_dispersion(Y1, y1_pred, return_p_values=True)
        dispersion_ite, p_values_ite = calculate_dispersion(Y1 - Y0, ite_pred, return_p_values=True)

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
