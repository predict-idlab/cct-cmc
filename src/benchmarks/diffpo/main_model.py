import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.metrics import calculate_dispersion, crps, loglikelihood

from .diff_model import diff_CSDI
from .utils import train


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        # keep the __init__ the same
        super().__init__()
        self.device = device
        self.target_dim = config["train"]["batch_size"]

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]

        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim

        self.cond_dim = config["diffusion"]["cond_dim"]
        self.mapping_noise = nn.Linear(2, self.cond_dim)

        if self.is_unconditional == False:
            self.emb_total_dim += 1

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)

        for i in range(len(observed_mask)):
            sample_ratio = 0.5
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_side_info(self, cond_mask):
        B, K, L = cond_mask.shape

        side_info = cond_mask

        return side_info

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(self, observed_data, cond_mask, gt_mask, side_info, is_train, set_t=-1, ps=None):
        B, K, L = observed_data.shape
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data[:, :, 1:3])
        noisy_data = (current_alpha**0.5) * observed_data[:, :, 1:3] + (
            1.0 - current_alpha
        ) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        a = observed_data[:, :, 0].unsqueeze(2)
        x = observed_data[:, :, 5:]
        cond_obs = torch.cat([a, x], dim=2)

        noisy_target = self.mapping_noise(noisy_data)
        diff_input = cond_obs + noisy_target

        predicted = self.diffmodel(diff_input, cond_obs, t).to(self.device)

        target_mask = gt_mask - cond_mask
        target_mask = target_mask.squeeze(1)[:, 1:3]

        noise = noise.squeeze(1)
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()

        x_batch = observed_data[:, :, 5:]
        if len(x_batch.shape) > 2:
            x_batch = x_batch.squeeze()
        if len(x_batch.shape) < 2:
            x_batch = x_batch.unsqueeze(1)
        t_batch = observed_data[:, :, 0].squeeze()

        weights = (t_batch / ps[:, 0]) + ((1 - t_batch) / (1 - ps[:, 0]))
        weights = weights.reshape(-1, 1, 1)
        weights = torch.clamp(weights, min=0.1, max=0.9)

        loss = (weights * (residual**2)).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)
        else:
            a = observed_data[:, :, 0].unsqueeze(2)
            x = observed_data[:, :, 5:]
            cond_obs = torch.cat([a, x], dim=2)
            noisy_target = self.mapping_noise(noisy_data)
            total_input = cond_obs + noisy_target
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, 2).to(self.device)

        for i in range(n_samples):
            generated_target = observed_data[:, :, 1:3]

            current_sample = torch.randn_like(generated_target)

            for t in range(self.num_steps - 1, -1, -1):
                a = observed_data[:, :, 0].unsqueeze(2)
                x = observed_data[:, :, 5:]
                cond_obs = torch.cat([a, x], dim=2)

                noisy_target = self.mapping_noise(current_sample)
                diff_input = cond_obs + noisy_target

                predicted = self.diffmodel(diff_input, cond_obs, t).to(self.device)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5

                current_sample = current_sample.squeeze(1)

                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

                current_sample = current_sample.unsqueeze(1)

            current_sample = current_sample.squeeze(1)
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (observed_data, observed_mask, gt_mask, for_pattern_mask, _, ps) = self.process_data(batch)

        if is_train == 0:
            cond_mask = gt_mask.clone()
        else:
            cond_mask = gt_mask.clone()

            cond_mask[:, :, 1] = 0
            cond_mask[:, :, 2] = 0

        side_info = self.get_side_info(cond_mask)
        loss_func = (
            self.calc_loss(observed_data, cond_mask, gt_mask, side_info, is_train, set_t=-1, ps=ps)
            if is_train == 1
            else self.calc_loss_valid
        )

        return loss_func

    def evaluate(self, batch, n_samples):
        (observed_data, observed_mask, gt_mask, _, cut_length, ps) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            cond_mask[:, :, 0] = 0
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask


class DiffPO(CSDI_base):
    def __init__(self, config, device, target_dim=1):
        super(DiffPO, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"][:, np.newaxis, :]
        observed_data = observed_data.to(self.device).float()

        observed_mask = batch["observed_mask"][:, np.newaxis, :]
        observed_mask = observed_mask.to(self.device).float()

        gt_mask = batch["gt_mask"][:, np.newaxis, :]

        gt_mask = gt_mask.to(self.device).float()

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        ps = batch["ps"].to(self.device).float()

        return (
            observed_data,
            observed_mask,
            gt_mask,
            for_pattern_mask,
            cut_length,
            ps,
        )


class DiffPOITE:

    def __init__(self, config, target_dim=1):
        self.target_dim = target_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DiffPO(config, self.device, target_dim).to(self.device)
        logging.info(f"DiffPO is using device: {self.device}")

    def fit(self, X, Y0, Y1, W, ps, split_ratio=0.9):
        """
        Fits the DiffPO model.

        :X: The input covariates
        :Y: The corresponding outcomes
        :W: The treatment assignments
        :ps: The propensity scores
        """
        self.dataset = tabular_dataset(X, Y0, Y1, W, ps)
        # train_loader = torch.utils.data.DataLoader(
        #     self.dataset, batch_size=self.config["train"]["batch_size"], shuffle=True
        # )
        # Split the dataset into training and validation sets
        train_dataset, val_dataset = train_test_split(self.dataset, test_size=1 - split_ratio)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["train"]["batch_size"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config["train"]["batch_size"], shuffle=False
        )
        train(
            self.model,
            self.config["train"],
            train_loader,
            valid_loader=val_loader,
            valid_epoch_interval=self.config["train"]["valid_epoch_interval"],
        )
        print("Training complete.")

    def evaluate(self, X, Y0, Y1, W, ps, n_samples=100, alpha=0.05, return_p_values=False):
        """
        Evaluates  model. The evaluation metrics include:
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
        test_dataset = tabular_dataset(
            X,
            Y0,
            Y1,
            W,
            ps,
            X_scaler=self.dataset.X_scaler,
            Y_scaler=self.dataset.Y_scaler,
            train=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["train"]["batch_size"], shuffle=False
        )

        y0_preds = []
        y1_preds = []
        print("Evaluating the model...")
        for batch_no, test_batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            samples, observed_data, target_mask, observed_mask = self.model.evaluate(
                test_batch, n_samples=n_samples
            )
            y0_preds.append(samples[:, :, 0].cpu().numpy())
            y1_preds.append(samples[:, :, 1].cpu().numpy())

        y0_preds = np.concatenate(y0_preds, axis=0)
        # rescale the predictions
        y0_preds = self.dataset.Y_scaler.inverse_transform(y0_preds.reshape(-1, 1)).reshape(
            y0_preds.shape
        )
        y1_preds = np.concatenate(y1_preds, axis=0)
        y1_preds = self.dataset.Y_scaler.inverse_transform(y1_preds.reshape(-1, 1)).reshape(
            y1_preds.shape
        )
        ite_preds = y1_preds - y0_preds
        # Point prediction
        rmse_y0 = np.sqrt(np.mean((np.median(y0_preds, axis=1, keepdims=True) - Y0) ** 2))
        rmse_y1 = np.sqrt(np.mean((np.median(y1_preds, axis=1, keepdims=True) - Y1) ** 2))
        rmse_ite = np.sqrt(np.mean((np.median(ite_preds, axis=1, keepdims=True) - (Y1 - Y0)) ** 2))
        # Interval prediction
        y0_upper = np.percentile(y0_preds, 100 * (1 - alpha / 2), axis=1)
        y0_lower = np.percentile(y0_preds, 100 * alpha / 2, axis=1)
        y1_upper = np.percentile(y1_preds, 100 * (1 - alpha / 2), axis=1)
        y1_lower = np.percentile(y1_preds, 100 * alpha / 2, axis=1)
        ite_upper = np.percentile(ite_preds, 100 * (1 - alpha / 2), axis=1)
        ite_lower = np.percentile(ite_preds, 100 * alpha / 2, axis=1)
        coverage_y0 = np.mean((Y0 >= y0_lower) & (Y0 <= y0_upper))
        coverage_y1 = np.mean((Y1 >= y1_lower) & (Y1 <= y1_upper))
        coverage_ite = np.mean(((Y1 - Y0) >= ite_lower) & ((Y1 - Y0) <= ite_upper))
        efficiency_y0 = np.mean((y0_upper - y0_lower))
        efficiency_y1 = np.mean((y1_upper - y1_lower))
        efficiency_ite = np.mean((ite_upper - ite_lower))
        # Distribution prediction
        crps_y0 = crps(Y0, y0_preds, return_average=True)
        crps_y1 = crps(Y1, y1_preds, return_average=True)
        crps_ite = crps(Y1 - Y0, ite_preds, return_average=True)
        ll_y0 = loglikelihood(Y0, y0_preds, return_average=True)
        ll_y1 = loglikelihood(Y1, y1_preds, return_average=True)
        ll_ite = loglikelihood(Y1 - Y0, ite_preds, return_average=True)

        # Dispersion
        dispersion_y0, p_values_y0 = calculate_dispersion(Y0, y0_preds, return_p_values=True)
        dispersion_y1, p_values_y1 = calculate_dispersion(Y1, y1_preds, return_p_values=True)
        dispersion_ite, p_values_ite = calculate_dispersion(
            Y1 - Y0, ite_preds, return_p_values=True
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


class tabular_dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y0, Y1, W, ps, X_scaler=None, Y_scaler=None, train=True):
        self.ps = ps.reshape(-1, 1)
        Y0 = Y0.reshape(-1, 1)
        Y1 = Y1.reshape(-1, 1)
        if X_scaler is not None:
            self.X_scaler = X_scaler
            X_scaled = self.X_scaler.transform(X)
        else:
            self.X_scaler = StandardScaler()
            X_scaled = self.X_scaler.fit_transform(X)
        if Y_scaler is not None:
            self.Y_scaler = Y_scaler
            Y0_scaled = self.Y_scaler.transform(Y0.reshape(-1, 1))
            Y1_scaled = self.Y_scaler.transform(Y1.reshape(-1, 1))
        else:
            self.Y_scaler = StandardScaler()
            self.Y_scaler.fit(np.concatenate([Y0, Y1]).reshape(-1, 1))
            Y0_scaled = self.Y_scaler.transform(Y0.reshape(-1, 1))
            Y1_scaled = self.Y_scaler.transform(Y1.reshape(-1, 1))

        full_data = np.concatenate(
            [
                W.reshape((-1, 1)),
                Y0_scaled.reshape((-1, 1)),
                Y1_scaled.reshape((-1, 1)),
                Y0_scaled.reshape((-1, 1)),
                Y1_scaled.reshape((-1, 1)),
                X_scaled,
            ],
            axis=1,
        )

        mask = np.ones(full_data.shape)
        mask[:, 3] = 0
        mask[:, 4] = 0

        for i in range(len(mask)):
            w = full_data[i, 0]
            if w == 0:
                mask[i, 2] = 0.0  # mask y1
            else:
                mask[i, 1] = 0.0  # mask y0

        if train:
            self.gt_masks = mask
            self.observed_values = np.nan_to_num(full_data)
            self.observed_masks = (~np.isnan(full_data)).astype(int)
            self.gt_masks = self.gt_masks.astype(int)
        else:
            self.gt_masks = mask
            self.gt_masks[:, 1] = 0.0
            self.gt_masks[:, 2] = 0.0

            self.observed_values = np.nan_to_num(full_data)
            self.observed_masks = (~np.isnan(full_data)).astype(int)
            self.gt_masks = self.gt_masks.astype(int)

    def __len__(self):
        return len(self.observed_values)

    def __getitem__(self, idx):
        return {
            "observed_data": self.observed_values[idx],
            "observed_mask": self.observed_masks[idx],
            "gt_mask": self.gt_masks[idx],
            "ps": self.ps[idx],
        }
