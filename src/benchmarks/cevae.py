# Inspired by: https://github.com/toonvds/NOFLITE

import logging
import sys
from collections import defaultdict

import matplotlib.pyplot as plt  # this is used for the plot the graph
import numpy as np  # linear algebra
import torch
import torch.distributions
import torch.nn.functional as F
from scipy.stats import randint
from torch import nn, optim
from torch.distributions import bernoulli, normal
from torch.utils.data import DataLoader
from tqdm import notebook, tqdm

from src.metrics import calculate_dispersion, crps, loglikelihood


def init_qz(qz, pz, y, t, x):
    """
    Initialize qz towards outputting standard normal distributions
    - with standard torch init of weights the gradients tend to explode after first update step
    """
    idx = list(range(x.shape[0]))
    np.random.shuffle(idx)

    optimizer = optim.Adam(qz.parameters(), lr=0.001)

    for i in range(50):
        batch = np.random.choice(idx, 1)
        x_train, y_train, t_train = (x[batch]), (y[batch]), (t[batch])
        xy = torch.cat((x_train, y_train), 1)

        z_infer = qz(xy=xy, t=t_train)

        # KL(q_z|p_z) mean approx, to be minimized
        # KLqp = (z_infer.log_prob(z_infer.mean) - pz.log_prob(z_infer.mean)).sum(1)
        # Analytic KL
        KLqp = (
            -torch.log(z_infer.stddev) + 1 / 2 * (z_infer.variance + z_infer.mean**2 - 1)
        ).sum(1)

        objective = KLqp
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        if KLqp != KLqp:
            raise ValueError("KL(pz,qz) contains NaN during init")

    return qz


class p_x_z(nn.Module):

    def __init__(self, dim_in=20, nh=3, dim_h=20, dim_out_bin=19, dim_out_con=6):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out_bin = dim_out_bin
        self.dim_out_con = dim_out_con

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh - 1)])
        # output layer defined separate for continuous and binary outputs
        self.output_bin = nn.Linear(dim_h, dim_out_bin)
        # for each output an mu and sigma are estimated
        self.output_con_mu = nn.Linear(dim_h, dim_out_con)
        self.output_con_sigma = nn.Linear(dim_h, dim_out_con)
        self.softplus = nn.Softplus()

    def forward(self, z_input):
        z = F.elu(self.input(z_input))
        for i in range(self.nh - 1):
            z = F.elu(self.hidden[i](z))
        # for binary outputs:
        x_bin_p = torch.sigmoid(self.output_bin(z))
        x_bin = bernoulli.Bernoulli(x_bin_p)
        # for continuous outputs
        mu, sigma = self.output_con_mu(z), self.softplus(self.output_con_sigma(z))
        x_con = normal.Normal(mu, sigma)

        if (z != z).all():
            raise ValueError("p(x|z) forward contains NaN")

        return x_con, x_bin


class p_t_z(nn.Module):

    def __init__(self, dim_in=20, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # for binary outputs:
        out_p = torch.sigmoid(self.output(x))

        out = bernoulli.Bernoulli(out_p)
        return out


class p_y_zt(nn.Module):

    def __init__(self, dim_in=20, nh=3, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # Separated forwards for different t values, TAR

        self.input_t0 = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden_t0 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.mu_t0 = nn.Linear(dim_h, dim_out)

        self.input_t1 = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden_t1 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.mu_t1 = nn.Linear(dim_h, dim_out)

    def forward(self, z, t):
        # Separated forwards for different t values, TAR

        x_t0 = F.elu(self.input_t0(z))
        for i in range(self.nh):
            x_t0 = F.elu(self.hidden_t0[i](x_t0))
        mu_t0 = F.elu(self.mu_t0(x_t0))

        x_t1 = F.elu(self.input_t1(z))
        for i in range(self.nh):
            x_t1 = F.elu(self.hidden_t1[i](x_t1))
        mu_t1 = F.elu(self.mu_t1(x_t1))
        # set mu according to t value
        y = normal.Normal((1 - t) * mu_t0 + t * mu_t1, 1)

        return y


####### Inference model / Encoder #######
class q_t_x(nn.Module):

    def __init__(self, dim_in=25, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # for binary outputs:
        out_p = torch.sigmoid(self.output(x))
        out = bernoulli.Bernoulli(out_p)

        return out


class q_y_xt(nn.Module):

    def __init__(self, dim_in=25, nh=3, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        # separate outputs for different values of t
        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)

    def forward(self, x, t):
        # Unlike model network, shared parameters with separated heads
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # only output weights separated
        mu_t0 = self.mu_t0(x)
        mu_t1 = self.mu_t1(x)
        # set mu according to t, sigma set to 1
        y = normal.Normal((1 - t) * mu_t0 + t * mu_t1, 1)
        return y


class q_z_tyx(nn.Module):

    def __init__(self, dim_in=25 + 1, nh=3, dim_h=20, dim_out=20):
        super().__init__()
        # dim in is dim of x + dim of y
        # dim_out is dim of latent space z
        # save required vars
        self.nh = nh

        # Shared layers with separated output layers

        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])

        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)
        self.sigma_t0 = nn.Linear(dim_h, dim_out)
        self.sigma_t1 = nn.Linear(dim_h, dim_out)
        self.softplus = nn.Softplus()

    def forward(self, xy, t):
        # Shared layers with separated output layers
        # print('before first linear z_infer')
        # print(xy)
        x = F.elu(self.input(xy))
        # print('first linear z_infer')
        # print(x)
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))

        mu_t0 = self.mu_t0(x)
        mu_t1 = self.mu_t1(x)
        sigma_t0 = self.softplus(self.sigma_t0(x))
        sigma_t1 = self.softplus(self.sigma_t1(x))

        # Set mu and sigma according to t
        z = normal.Normal((1 - t) * mu_t0 + t * mu_t1, (1 - t) * sigma_t0 + t * sigma_t1)
        return z


class CEVAE:
    def __init__(self, dim_bin, dim_cont, lr=1e-4, batch_size=100, iters=7000, n_h=64):
        self.dim_bin = dim_bin
        self.dim_cont = dim_cont
        self.lr = lr
        self.batch_size = batch_size
        self.iters = iters
        self.n_h = n_h
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"CEVAE is using {self.device}")

        # init networks (overwritten per replication)
        self.p_x_z_dist = p_x_z(
            dim_in=20, nh=3, dim_h=n_h, dim_out_bin=dim_bin, dim_out_con=dim_cont
        ).to(self.device)
        self.p_t_z_dist = p_t_z(dim_in=20, nh=1, dim_h=n_h, dim_out=1).to(self.device)
        self.p_y_zt_dist = p_y_zt(dim_in=20, nh=3, dim_h=n_h, dim_out=1).to(self.device)
        self.q_t_x_dist = q_t_x(dim_in=dim_bin + dim_cont, nh=1, dim_h=n_h, dim_out=1).to(
            self.device
        )

        # t is not feed into network, therefore not increasing input size (y is fed).
        self.q_y_xt_dist = q_y_xt(dim_in=dim_bin + dim_cont, nh=3, dim_h=n_h, dim_out=1).to(
            self.device
        )
        self.q_z_tyx_dist = q_z_tyx(dim_in=dim_bin + dim_cont + 1, nh=3, dim_h=n_h, dim_out=20).to(
            self.device
        )
        self.p_z_dist = normal.Normal(
            torch.zeros(20).to(self.device), torch.ones(20).to(self.device)
        )

        # Create optimizer
        params = (
            list(self.p_x_z_dist.parameters())
            + list(self.p_t_z_dist.parameters())
            + list(self.p_y_zt_dist.parameters())
            + list(self.q_t_x_dist.parameters())
            + list(self.q_y_xt_dist.parameters())
            + list(self.q_z_tyx_dist.parameters())
        )

        self.optimizer = optim.Adamax(params, lr=lr, weight_decay=1e-4)

    def fit(self, X, W, Y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(W, torch.Tensor):
            W = torch.tensor(W, dtype=torch.float32)
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32)
        X = X.to(self.device)
        W = W.to(self.device).view(-1, 1)
        Y = Y.to(self.device).view(-1, 1)
        self.Y_mean, self.Y_std = torch.mean(Y), torch.std(Y)
        Y_prep = (Y - self.Y_mean) / self.Y_std

        # init q_z inference
        self.q_z_tyx_dist = init_qz(self.q_z_tyx_dist, self.p_z_dist, Y_prep, W, X)
        #####training################################################################
        loss = []
        for _ in tqdm(range(self.iters)):
            i = np.random.choice(X.shape[0], size=self.batch_size, replace=False)
            y_train = Y_prep[i, :]
            x_train = X[i, :]
            trt_train = W[i, :]

            # inferred distribution over z
            xy = torch.cat((x_train, y_train), 1)
            z_infer = self.q_z_tyx_dist(xy=xy, t=trt_train)
            # use a single sample to approximate expectation in lowerbound
            z_infer_sample = z_infer.sample()

            # RECONSTRUCTION LOSS
            # p(x|z)
            x_con, x_bin = self.p_x_z_dist(z_infer_sample)
            if self.dim_bin > 0:
                l1 = x_bin.log_prob(x_train[:, self.dim_cont :]).sum(1)
            else:
                l1 = 0
            if self.dim_cont > 0:
                l2 = x_con.log_prob(x_train[:, : self.dim_cont]).sum(1)
            else:
                l2 = 0

            # p(t|z)
            t = self.p_t_z_dist(z_infer_sample)
            l3 = t.log_prob(trt_train).squeeze()

            # p(y|t,z)
            # for training use trt_train, in out-of-sample prediction this becomes t_infer
            y = self.p_y_zt_dist(z_infer_sample, trt_train)
            l4 = y.log_prob(y_train).squeeze()

            # REGULARIZATION LOSS
            # p(z) - q(z|x,t,y)
            # approximate KL
            l5 = (self.p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(1)

            # AUXILIARY LOSS
            # q(t|x)
            self.t_infer = self.q_t_x_dist(x_train)
            l6 = self.t_infer.log_prob(trt_train).squeeze()

            # q(y|x,t)
            y_infer = self.q_y_xt_dist(x_train, trt_train)
            l7 = y_infer.log_prob(y_train).squeeze()

            # Total objective
            # inner sum to calculate loss per item, torch.mean over batch
            loss_mean = torch.mean(l1 + l2 + l3 + l4 + l5 + l6 + l7)
            loss.append(loss_mean.cpu().detach().numpy())
            objective = -loss_mean

            self.optimizer.zero_grad()
            # Calculate gradients
            objective.backward()
            # Update step
            self.optimizer.step()

    def sample_y0_y1(self, X, n_samples=500):
        """sample from the model"""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        out0 = []
        out1 = []

        t_infer = self.q_t_x_dist(X)
        for q in range(n_samples):
            ttmp = t_infer.sample()
            y_infer = self.q_y_xt_dist(X, ttmp)

            xy = torch.cat((X, y_infer.sample()), 1)
            z_infer = self.q_z_tyx_dist(xy=xy, t=ttmp).sample()
            # Manually input zeros and ones
            y0 = self.p_y_zt_dist(
                z_infer, torch.zeros(z_infer.shape[0], 1).to(self.device)
            ).sample()
            y1 = self.p_y_zt_dist(z_infer, torch.ones(z_infer.shape[0], 1).to(self.device)).sample()
            y0, y1 = y0 * self.Y_std + self.Y_mean, y1 * self.Y_std + self.Y_mean
            out0.append(y0.detach().cpu().numpy().ravel())
            out1.append(y1.detach().cpu().numpy().ravel())

        # the sample for the treated and control group
        out0 = np.stack(out0, axis=-1)
        out1 = np.stack(out1, axis=-1)
        return out0, out1

    def evaluate(self, X, Y0, Y1, n_samples=500, alpha=0.05, return_p_values=False):
        """
        Evaluates
            - Point prediction related:
                - RMSE for Y0, Y1, and ITE
            - Interval prediction related:
                - Coverage for Y0, Y1, and ITE for a given alpha
                - Effiecency for Y0, Y1, and ITE for a given alpha
            - Distribution prediction related:
                - CRPS for Y0, Y1, and ITE
                - LL for Y0, Y1, and ITE

        :X: The input covariates
        :Y0: The counterfactual outcome Y0
        :Y1: The counterfactual outcome Y1
        :alpha: The significance level for the confidence interval
        """
        y0_pred, y1_pred = self.sample_y0_y1(X, n_samples)
        ite_pred = y1_pred - y0_pred
        # Point prediction
        rmse_y0 = np.sqrt(np.mean((y0_pred.mean(axis=1) - Y0) ** 2))
        rmse_y1 = np.sqrt(np.mean((y1_pred.mean(axis=1) - Y1) ** 2))
        rmse_ite = np.sqrt(np.mean((ite_pred.mean(axis=1) - (Y1 - Y0)) ** 2))
        # Interval prediction
        y0_upper = np.percentile(y0_pred, 100 * (1 - alpha / 2), axis=1)
        y0_lower = np.percentile(y0_pred, 100 * alpha / 2, axis=1)
        y1_upper = np.percentile(y1_pred, 100 * (1 - alpha / 2), axis=1)
        y1_lower = np.percentile(y1_pred, 100 * alpha / 2, axis=1)
        ite_upper = np.percentile(ite_pred, 100 * (1 - alpha / 2), axis=1)
        ite_lower = np.percentile(ite_pred, 100 * alpha / 2, axis=1)
        coverage_y0 = np.mean((Y0 >= y0_lower) & (Y0 <= y0_upper))
        coverage_y1 = np.mean((Y1 >= y1_lower) & (Y1 <= y1_upper))
        coverage_ite = np.mean(((Y1 - Y0) >= ite_lower) & ((Y1 - Y0) <= ite_upper))
        efficiency_y0 = np.mean((y0_upper - y0_lower))
        efficiency_y1 = np.mean((y1_upper - y1_lower))
        efficiency_ite = np.mean((ite_upper - ite_lower))
        # Distribution prediction
        crps_y0 = crps(Y0, y0_pred, return_average=True)
        crps_y1 = crps(Y1, y1_pred, return_average=True)
        crps_ite = crps(Y1 - Y0, ite_pred, return_average=True)
        ll_y0 = loglikelihood(Y0, y0_pred, return_average=True)
        ll_y1 = loglikelihood(Y1, y1_pred, return_average=True)
        ll_ite = loglikelihood(Y1 - Y0, ite_pred, return_average=True)

        # Dispersion
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
