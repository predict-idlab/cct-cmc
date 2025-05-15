# Inspired by: https://github.com/toonvds/NOFLITE
# Based on: https://github.com/thuizhou/Collaborating-Networks
import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d
from torch import nn, optim
from tqdm import tqdm

from src.metrics import calculate_dispersion, crps, loglikelihood


class cn_g(nn.Module):
    def __init__(self, hidden_size=25):
        super().__init__()
        self.k1 = 100
        self.k2 = 80

        self.fc1 = nn.Linear(hidden_size * 2 + 2, self.k1)
        self.fc2 = nn.Linear(self.k1, self.k2)
        self.fc3 = nn.Linear(self.k2, 1)

    def forward(self, y, x):
        data = torch.cat([y, x], dim=1)
        h1 = self.fc1(data)
        h1 = F.elu(h1)
        h2 = self.fc2(h1)
        h2 = F.elu(h2)
        h3 = self.fc3(h2)
        g_logit = h3
        return g_logit


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.001)


# critic for latent representation
class critic(nn.Module):
    def __init__(self, hidden_size=25):
        super().__init__()
        self.k1 = 50
        self.fc1 = nn.Linear(hidden_size, self.k1)
        self.fc2 = nn.Linear(self.k1, 1)

    def forward(self, s):
        h1 = F.elu(self.fc1(s))
        critic_out = self.fc2(h1)
        return critic_out


# generator for latent representation
class gen(nn.Module):
    def __init__(self, input_size, hidden_size=25):
        super().__init__()

        self.k1 = 100

        self.fc1 = nn.Linear(input_size, self.k1)
        self.fc2 = nn.Linear(self.k1, hidden_size)

    def forward(self, x):
        h1 = F.elu(self.fc1(x))
        gen_out = self.fc2(h1)
        return gen_out


# predictor for propensity model
class prop_pred(nn.Module):
    def __init__(self, hidden_size=25):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 1)

    def forward(self, s):
        prop_out = self.fc1(s)
        return prop_out


class FCCN(nn.Module):
    def __init__(self, input_size, hidden_size=25, alpha=5e-4, beta=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.g0 = cn_g(hidden_size)
        self.g1 = cn_g(hidden_size)
        self.critic_ipm = critic(hidden_size)

        self.g0.apply(weights_init)
        self.g1.apply(weights_init)

        # domain invariant and domain specific
        self.gen_lat_i = gen(input_size)
        self.gen_lat_s = gen(input_size)

        self.prop_est = prop_pred()

        self.gloss = nn.BCEWithLogitsLoss()
        self.proploss = nn.BCEWithLogitsLoss()
        self.poss_vals = 3000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        logging.info(f"FCCN is using device: {self.device}")

    def train(self, xtrain, ytrain, trttrain, iters=20000, batch_size=128):
        xtrain = torch.from_numpy(xtrain).float().to(self.device)
        ytrain = torch.from_numpy(ytrain).float().view(-1, 1).to(self.device)
        trttrain = torch.from_numpy(trttrain.reshape(-1, 1)).float().to(self.device)

        gparams = list(self.g0.parameters()) + list(self.g1.parameters())
        optimizer_g = optim.Adam(gparams, lr=1e-4)
        optimizer_critic = optim.RMSprop(self.critic_ipm.parameters(), lr=5e-4)
        optimizer_gen = optim.RMSprop(
            list(self.gen_lat_i.parameters()) + list(self.gen_lat_s.parameters()), lr=1e-4
        )
        optimizer_prop = optim.Adam(self.prop_est.parameters(), lr=1e-4)

        my_lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer_critic, gamma=0.998
        )
        my_lr_scheduler_prop = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer_prop, gamma=0.998
        )
        my_lr_scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer_gen, gamma=0.998
        )

        g_loss = []
        critic_loss = []
        gen_loss = []
        prop_loss = []

        n_critic = 5
        clip_value = torch.tensor(0.01, device=self.device)
        ipm_weight = torch.tensor(self.alpha, device=self.device)
        prop_weight = torch.tensor(self.beta, device=self.device)

        # Get ranges for prediction:
        self.y_poss = torch.linspace(
            ytrain.min() - 1, ytrain.max() + 1, self.poss_vals, device=self.device
        ).reshape(-1, 1)
        self.y_possnp = torch.linspace(
            ytrain.min() - 1, ytrain.max() + 1, self.poss_vals, device=self.device
        )

        self.g0.train()
        self.g1.train()
        self.critic_ipm.train()
        self.gen_lat_i.train()
        self.gen_lat_s.train()

        train_data = torch.utils.data.TensorDataset(xtrain, ytrain, trttrain)
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(iters // len(data_loader))):
            for xs, ys, trts in data_loader:
                # train g-network and latent representation ###########################
                yhat = (
                    torch.rand_like(ys, device=self.device) * (ytrain.max() + 2 - ytrain.min())
                    + ytrain.min()
                    - 1
                )

                ntrt = torch.mean(trts)
                ncon = 1 - ntrt

                optimizer_g.zero_grad()
                optimizer_critic.zero_grad()
                optimizer_gen.zero_grad()
                optimizer_prop.zero_grad()

                with torch.no_grad():
                    ylt = ys < yhat
                    ylt = ylt.float()

                lats_i = self.gen_lat_i(xs)
                lats_s = self.gen_lat_s(xs)
                lats = torch.cat([lats_i, lats_s], dim=1)

                proplogit = self.prop_est(lats_s)

                props = torch.sigmoid(proplogit)

                lats1 = torch.cat([lats, props], dim=1)

                propl = self.proploss(proplogit, trts)

                qhat_logit0 = self.g0(yhat, lats1)
                qhat_logit1 = self.g1(yhat, lats1)
                qhat_logit = qhat_logit0 * (1 - trts) + qhat_logit1 * trts

                gl = self.gloss(qhat_logit, ylt)
                ipms = self.critic_ipm(lats_i)
                pos_ipm_loss = (
                    torch.mean(ipms * trts) / ntrt - torch.mean(ipms * (1.0 - trts)) / ncon
                )

                combined_loss = (
                    gl + torch.mul(pos_ipm_loss, ipm_weight) + torch.mul(propl, prop_weight)
                )
                combined_loss.backward(retain_graph=False)

                optimizer_gen.step()
                optimizer_prop.step()
                optimizer_g.step()

                my_lr_scheduler_prop.step()
                my_lr_scheduler_gen.step()

                g_loss.append(gl.cpu().item())
                prop_loss.append(propl.cpu().item())

                ##train critic #########################
                for j in range(n_critic):
                    optimizer_g.zero_grad()
                    optimizer_critic.zero_grad()
                    optimizer_gen.zero_grad()

                    i = torch.tensor(
                        np.random.choice(xtrain.shape[0], size=batch_size, replace=False),
                        device=self.device,
                    )
                    xs = xtrain[i, :]
                    trts = trttrain[i, :]

                    lats_i = self.gen_lat_i(xs)
                    ipms = self.critic_ipm(lats_i)
                    ntrt = torch.mean(trts)
                    ncon = 1 - ntrt
                    criticl = -(
                        torch.mean(ipms * trts) / ntrt - torch.mean(ipms * (1.0 - trts)) / ncon
                    )
                    criticl.backward(retain_graph=False)
                    optimizer_critic.step()

                    # Clip critic weights
                    for p in self.critic_ipm.parameters():
                        p.data.clamp_(-clip_value, clip_value)

                    my_lr_scheduler_critic.step()
                    critic_loss.append(criticl.cpu().item())

        self.g0.eval()
        self.g1.eval()
        self.critic_ipm.eval()
        self.gen_lat_i.eval()
        self.gen_lat_s.eval()

        return [g_loss, critic_loss, gen_loss, prop_loss]

    def sample_y0_y1(self, xtest, n_samples=500):
        xtest = torch.from_numpy(xtest).float().to(self.device)

        # Generate latent representations for all test instances
        stmp_i = self.gen_lat_i(xtest)
        stmp_s = self.gen_lat_s(xtest)
        proplogitte = self.prop_est(stmp_s)
        propste = torch.sigmoid(proplogitte)
        stest = torch.cat([stmp_i, stmp_s, propste], dim=1)  # shape: (n_test, hidden_size*2 + 1)

        n_test = xtest.size(0)
        y_poss = self.y_poss.to(self.device)  # shape: (poss_vals, 1)

        # Expand dimensions to create a grid of all test instances and possible y values
        # y_poss_expanded shape: (n_test, poss_vals, 1)
        y_poss_expanded = y_poss.unsqueeze(0).expand(n_test, -1, -1)
        # stest_expanded shape: (n_test, poss_vals, hidden_size*2 + 1)
        stest_expanded = stest.unsqueeze(1).expand(-1, self.poss_vals, -1)

        # Combine and reshape for batch processing
        combined_input0 = torch.cat([y_poss_expanded, stest_expanded], dim=2).view(
            -1, stest.size(1) + 1
        )
        combined_input1 = combined_input0  # same input structure for g1

        # Batch forward pass through g0 and g1
        with torch.no_grad():
            probs0 = torch.sigmoid(self.g0(combined_input0[:, :1], combined_input0[:, 1:]))
            probs0 = probs0.view(n_test, self.poss_vals)
            probs1 = torch.sigmoid(self.g1(combined_input1[:, :1], combined_input1[:, 1:]))
            probs1 = probs1.view(n_test, self.poss_vals)

        # Interpolate samples for all test instances
        y0_preds = np.zeros((n_test, n_samples))
        y1_preds = np.zeros((n_test, n_samples))
        y_poss_np = self.y_possnp.cpu().numpy()

        for i in range(n_test):
            # Process y0
            p0 = probs0[i].cpu().numpy().ravel()
            p0[0], p0[-1] = 0, 1  # Ensure endpoints
            y0_preds[i] = interp1d(p0, y_poss_np)(np.random.uniform(0.002, 0.998, n_samples))

            # Process y1
            p1 = probs1[i].cpu().numpy().ravel()
            p1[0], p1[-1] = 0, 1  # Ensure endpoints
            y1_preds[i] = interp1d(p1, y_poss_np)(np.random.uniform(0.002, 0.998, n_samples))

        return y0_preds, y1_preds

    def evaluate(self, X, Y0, Y1, alpha=0.05, n_samples=500, return_p_values=False):
        """
        Evaluates the FCCN model. The evaluation metrics include:
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
