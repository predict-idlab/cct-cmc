# Copied from: https://github.com/toonvds/NOFLITE

import logging

import normflows as nf
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.stats import norm, sem, truncnorm
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.metrics import calculate_dispersion, crps, loglikelihood

from .gaussianization import rbig_block
from .lossfunctions import calcMMD
from .sigmoid_flow import DSFMarginal


class Noflite(pl.LightningModule):
    def __init__(self, params):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the data.
            cond_features - A list with conditional features
            cond_treatment - Conditional value on whether treated or control
        """
        super().__init__()

        self.lr = params["lr"]
        # self.beta1 = params['beta1']
        # self.beta2 = params['beta2']
        self.lambda_l1 = params["lambda_l1"]
        self.lambda_l2 = params["lambda_l2"]
        self.batch_size = params["batch_size"]
        self.noise_reg_x = params["noise_reg_x"]
        self.noise_reg_y = params["noise_reg_y"]

        self.bin_outcome = params["bin_outcome"]
        self.input_size = params["input_size"]
        self.lambda_mmd = params["lambda_mmd"]
        self.hidden_neurons_encoder = params["hidden_neurons_encoder"]
        self.hidden_layers_balancer = params["hidden_layers_balancer"]
        self.hidden_layers_encoder = params["hidden_layers_encoder"]
        self.hidden_layers_prior = params["hidden_layers_prior"]
        self.flow_type = params["flow_type"]
        self.n_flows = params["n_flows"]
        self.metalearner = params["metalearner"]
        self.metalearner = params["metalearner"]

        assert (
            self.hidden_layers_balancer > 0
        ), f"Balancer layers minimum 1, got: {self.hidden_layers_balancer}"
        assert (
            self.hidden_layers_encoder >= 0
        ), f"Encoder shared layers minimum 0, got: {self.hidden_layers_encoder}"
        assert (
            self.hidden_layers_prior > 0
        ), f"Encoder separate layers minimum 1, got: {self.hidden_layers_prior}"

        # Flow parameters:
        self.cur_datapoints = params["cur_datapoints"]
        self.datapoint_num = params["datapoint_num"]
        self.resid_layers = params["resid_layers"]

        # Sigmoid flow parameters:
        self.hidden_neurons_trans = params["hidden_neurons_trans"]
        self.hidden_neurons_cond = params["hidden_neurons_cond"]
        self.hidden_layers_cond = params["hidden_layers_cond"]
        self.dense = params["dense"]
        self.max_left = -1  # Expanded automatically (if needed)
        self.max_right = 1
        self.max_iter = 100

        self.sweep = False  # Set True in main.py for sweep test_step

        # Encoder: transform (x, t) to prior p(z|x, t)
        # Transform x to balanced representation
        # dropout_rate = 0.0
        # balancer_list = [nn.Dropout(dropout_rate), nn.Linear(self.input_size, self.hidden_neurons_encoder), nn.ELU()]
        balancer_list = [
            nn.Linear(self.input_size, self.hidden_neurons_encoder),
            nn.ELU(),
        ]  # Input size of X (without t)
        for layer in range(self.hidden_layers_balancer - 1):
            # balancer_list.append(nn.Dropout(dropout_rate))
            balancer_list.append(
                nn.Linear(self.hidden_neurons_encoder, self.hidden_neurons_encoder)
            )
            balancer_list.append(nn.ELU())
        self.balancer = nn.Sequential(*balancer_list)

        # Treatment arms: S- or T-learner
        if self.metalearner == "S":
            # Combine balanced representation with treatment
            # Concatenation of balanced x with t
            if self.hidden_layers_encoder > 0:
                prior_encoder_list = [
                    nn.Linear(self.hidden_neurons_encoder + 1, self.hidden_neurons_encoder),
                    nn.ELU(),
                ]
                for layer in range(self.hidden_layers_encoder - 1):
                    prior_encoder_list.append(
                        nn.Linear(self.hidden_neurons_encoder, self.hidden_neurons_encoder)
                    )
                    prior_encoder_list.append(nn.ELU())
            else:
                prior_encoder_list = []
            self.prior_encoder = nn.Sequential(*prior_encoder_list)

            # Get mean
            cond_mean_list = []
            for layer in range(self.hidden_layers_prior - 1):
                if layer == 0 and self.hidden_layers_encoder == 0:  # Concatenate t here
                    cond_mean_list.append(
                        nn.Linear(self.hidden_neurons_encoder + 1, self.hidden_neurons_encoder)
                    )
                else:
                    cond_mean_list.append(
                        nn.Linear(self.hidden_neurons_encoder, self.hidden_neurons_encoder)
                    )
                cond_mean_list.append(nn.ELU())
            if self.hidden_layers_encoder == 0 and self.hidden_layers_prior == 1:
                cond_mean_list.append(nn.Linear(self.hidden_neurons_encoder + 1, 1))
            else:
                cond_mean_list.append(nn.Linear(self.hidden_neurons_encoder, 1))
            self.cond_mean = nn.Sequential(*cond_mean_list)

            # Get log std
            cond_std_list = []
            for layer in range(self.hidden_layers_prior - 1):
                if layer == 0 and self.hidden_layers_encoder == 0:  # Concatenate t here
                    cond_std_list.append(
                        nn.Linear(self.hidden_neurons_encoder + 1, self.hidden_neurons_encoder)
                    )
                else:
                    cond_std_list.append(
                        nn.Linear(self.hidden_neurons_encoder, self.hidden_neurons_encoder)
                    )
                cond_std_list.append(nn.ELU())
            if self.hidden_layers_encoder == 0 and self.hidden_layers_prior == 1:
                cond_std_list.append(nn.Linear(self.hidden_neurons_encoder + 1, 1))
            else:
                cond_std_list.append(nn.Linear(self.hidden_neurons_encoder, 1))
            self.cond_std = nn.Sequential(*cond_std_list)

            # Decoder: transform Gaussian prior to complex posterior
            if self.n_flows > 0:
                if self.flow_type == "SigmoidXT":
                    self.flows = DSFMarginal(
                        context_dim=self.hidden_neurons_encoder
                        + 1,  # Balanced representation of x + t
                        mlp_layers=self.hidden_layers_cond,
                        mlp_dim=self.hidden_neurons_cond,
                        flow_layers=self.n_flows,
                        flow_hid_dim=self.hidden_neurons_trans,
                        no_logit=False,
                        dense=self.dense,
                    )
                elif self.flow_type == "SigmoidT":
                    self.flows = DSFMarginal(
                        context_dim=1,
                        mlp_layers=self.hidden_layers_cond,
                        mlp_dim=self.hidden_neurons_cond,
                        flow_layers=self.n_flows,
                        flow_hid_dim=self.hidden_neurons_trans,
                        no_logit=False,
                        dense=self.dense,
                    )
                else:
                    flows = []
                    for i in range(self.n_flows):
                        if self.flow_type == "GF":
                            flows += [
                                rbig_block(
                                    layer=i,
                                    dimension=1,
                                    datapoint_num=self.datapoint_num,  # For bandwidth parameters
                                    householder_iter=0,
                                    semi_learning=False,
                                    multidim_kernel=True,
                                    usehouseholder=False,
                                    need_rotation=False,
                                )
                            ]
                        elif self.flow_type == "RQNSF-AR":
                            flows += [
                                nf.flows.AutoregressiveRationalQuadraticSpline(
                                    num_input_channels=1,
                                    num_blocks=self.resid_layers,
                                    num_hidden_channels=self.hidden_neurons_trans,
                                    num_bins=8,  # Original default = 8
                                    tail_bound=3,  # Original default = 3
                                    dropout_probability=0.0,
                                    init_identity=True,
                                )
                            ]
                        elif self.flow_type == "Residual":
                            net = nf.nets.LipschitzMLP(
                                [1]
                                + [self.hidden_neurons_trans] * self.resid_layers
                                + [1],  # In, hidden x layers, out
                                init_zeros=True,
                                lipschitz_const=0.98,
                            )
                            flows += [nf.flows.Residual(net)]

                    self.flows = nn.ModuleList(flows)

        elif self.metalearner == "T":
            # Combine balanced representation with treatment
            # Concatenation of balanced x with t
            if self.hidden_layers_encoder > 0:
                prior_encoder_list0 = [
                    nn.Linear(self.hidden_neurons_encoder + 1, self.hidden_neurons_encoder),
                    nn.ELU(),
                ]
                for layer in range(self.hidden_layers_encoder - 1):
                    prior_encoder_list0.append(
                        nn.Linear(self.hidden_neurons_encoder, self.hidden_neurons_encoder)
                    )
                    prior_encoder_list0.append(nn.ELU())
                prior_encoder_list1 = [
                    nn.Linear(self.hidden_neurons_encoder + 1, self.hidden_neurons_encoder),
                    nn.ELU(),
                ]
                for layer in range(self.hidden_layers_encoder - 1):
                    prior_encoder_list1.append(
                        nn.Linear(self.hidden_neurons_encoder, self.hidden_neurons_encoder)
                    )
                    prior_encoder_list1.append(nn.ELU())
            else:
                prior_encoder_list0 = []
                prior_encoder_list1 = []
            self.prior_encoder0 = nn.Sequential(*prior_encoder_list0)
            self.prior_encoder1 = nn.Sequential(*prior_encoder_list1)

            # Get mean per treatment
            cond_mean_list0 = []
            for layer in range(self.hidden_layers_prior - 1):
                if layer == 0 and self.hidden_layers_encoder == 0:  # Concatenate t here
                    cond_mean_list0.append(
                        nn.Linear(self.hidden_neurons_encoder + 1, self.hidden_neurons_encoder)
                    )
                else:
                    cond_mean_list0.append(
                        nn.Linear(self.hidden_neurons_encoder, self.hidden_neurons_encoder)
                    )
                cond_mean_list0.append(nn.ELU())
            if self.hidden_layers_encoder == 0 and self.hidden_layers_prior == 1:
                cond_mean_list0.append(nn.Linear(self.hidden_neurons_encoder + 1, 1))
            else:
                cond_mean_list0.append(nn.Linear(self.hidden_neurons_encoder, 1))
            self.cond_mean0 = nn.Sequential(*cond_mean_list0)
            cond_mean_list1 = []
            for layer in range(self.hidden_layers_prior - 1):
                if layer == 0 and self.hidden_layers_encoder == 0:  # Concatenate t here
                    cond_mean_list1.append(
                        nn.Linear(self.hidden_neurons_encoder + 1, self.hidden_neurons_encoder)
                    )
                else:
                    cond_mean_list1.append(
                        nn.Linear(self.hidden_neurons_encoder, self.hidden_neurons_encoder)
                    )
                cond_mean_list1.append(nn.ELU())
            if self.hidden_layers_encoder == 0 and self.hidden_layers_prior == 1:
                cond_mean_list1.append(nn.Linear(self.hidden_neurons_encoder + 1, 1))
            else:
                cond_mean_list1.append(nn.Linear(self.hidden_neurons_encoder, 1))
            self.cond_mean1 = nn.Sequential(*cond_mean_list1)

            # Get log std per treatment
            cond_std_list0 = []
            for layer in range(self.hidden_layers_prior - 1):
                if layer == 0 and self.hidden_layers_encoder == 0:  # Concatenate t here
                    cond_std_list0.append(
                        nn.Linear(self.hidden_neurons_encoder + 1, self.hidden_neurons_encoder)
                    )
                else:
                    cond_std_list0.append(
                        nn.Linear(self.hidden_neurons_encoder, self.hidden_neurons_encoder)
                    )
                cond_std_list0.append(nn.ELU())
            if self.hidden_layers_encoder == 0 and self.hidden_layers_prior == 1:
                cond_std_list0.append(nn.Linear(self.hidden_neurons_encoder + 1, 1))
            else:
                cond_std_list0.append(nn.Linear(self.hidden_neurons_encoder, 1))
            # cond_std_list0.append(nn.PReLU())
            self.cond_std0 = nn.Sequential(*cond_std_list0)
            cond_std_list1 = []
            for layer in range(self.hidden_layers_prior - 1):
                if layer == 0 and self.hidden_layers_encoder == 0:  # Concatenate t here
                    cond_std_list1.append(
                        nn.Linear(self.hidden_neurons_encoder + 1, self.hidden_neurons_encoder)
                    )
                else:
                    cond_std_list1.append(
                        nn.Linear(self.hidden_neurons_encoder, self.hidden_neurons_encoder)
                    )
                cond_std_list1.append(nn.ELU())
            if self.hidden_layers_encoder == 0 and self.hidden_layers_prior == 1:
                cond_std_list1.append(nn.Linear(self.hidden_neurons_encoder + 1, 1))
            else:
                cond_std_list1.append(nn.Linear(self.hidden_neurons_encoder, 1))
            # cond_std_list1.append(nn.PReLU())
            self.cond_std1 = nn.Sequential(*cond_std_list1)

            # Decoder/normalizing flow: transform Gaussian prior to complex posterior
            if self.n_flows > 0:
                if self.flow_type == "SigmoidX":  # No need to include T here
                    self.flows0 = DSFMarginal(
                        context_dim=self.hidden_neurons_encoder,
                        mlp_layers=self.hidden_layers_cond,
                        mlp_dim=self.hidden_neurons_cond,
                        flow_layers=self.n_flows,
                        flow_hid_dim=self.hidden_neurons_trans,
                        no_logit=False,
                        dense=self.dense,
                    )
                    self.flows1 = DSFMarginal(
                        context_dim=self.hidden_neurons_encoder,
                        mlp_layers=self.hidden_layers_cond,
                        mlp_dim=self.hidden_neurons_cond,
                        flow_layers=self.n_flows,
                        flow_hid_dim=self.hidden_neurons_trans,
                        no_logit=False,
                        dense=self.dense,
                    )
                elif self.flow_type == "SigmoidT":  # Obsolete
                    self.flows0 = DSFMarginal(
                        context_dim=1,
                        mlp_layers=self.hidden_layers_cond,
                        mlp_dim=self.hidden_neurons_cond,
                        flow_layers=self.n_flows,
                        flow_hid_dim=self.hidden_neurons_trans,
                        no_logit=False,
                        dense=self.dense,
                    )
                    self.flows1 = DSFMarginal(
                        context_dim=1,
                        mlp_layers=self.hidden_layers_cond,
                        mlp_dim=self.hidden_neurons_cond,
                        flow_layers=self.n_flows,
                        flow_hid_dim=self.hidden_neurons_trans,
                        no_logit=False,
                        dense=self.dense,
                    )

                elif self.flow_type == "Sigmoid":
                    self.flows0 = DSFMarginal(
                        context_dim=0,
                        mlp_layers=self.hidden_layers_cond,
                        mlp_dim=self.hidden_neurons_cond,
                        flow_layers=self.n_flows,
                        flow_hid_dim=self.hidden_neurons_trans,
                        no_logit=False,
                        dense=self.dense,
                    )
                    self.flows1 = DSFMarginal(
                        context_dim=0,
                        mlp_layers=self.hidden_layers_cond,
                        mlp_dim=self.hidden_neurons_cond,
                        flow_layers=self.n_flows,
                        flow_hid_dim=self.hidden_neurons_trans,
                        no_logit=False,
                        dense=self.dense,
                    )
                else:
                    flows0 = []
                    flows1 = []
                    for i in range(self.n_flows):
                        if self.flow_type == "GF":
                            flows0 += [
                                rbig_block(
                                    layer=i,
                                    dimension=1,
                                    datapoint_num=self.datapoint_num,  # For bandwidth parameters
                                    householder_iter=0,
                                    semi_learning=False,
                                    multidim_kernel=True,
                                    usehouseholder=False,
                                    need_rotation=False,
                                )
                            ]
                            flows1 += [
                                rbig_block(
                                    layer=i,
                                    dimension=1,
                                    datapoint_num=self.datapoint_num,  # For bandwidth parameters
                                    householder_iter=0,
                                    semi_learning=False,
                                    multidim_kernel=True,
                                    usehouseholder=False,
                                    need_rotation=False,
                                )
                            ]
                        elif self.flow_type == "RQNSF-AR":
                            flows0 += [
                                nf.flows.AutoregressiveRationalQuadraticSpline(
                                    num_input_channels=1,
                                    num_blocks=self.resid_layers,
                                    num_hidden_channels=self.hidden_neurons_trans,
                                    num_bins=1,  # Original default = 8
                                    tail_bound=10,  # Original default = 3
                                    dropout_probability=0,
                                    init_identity=True,
                                )
                            ]
                            flows1 += [
                                nf.flows.AutoregressiveRationalQuadraticSpline(
                                    num_input_channels=1,
                                    num_blocks=self.resid_layers,
                                    num_hidden_channels=self.hidden_neurons_trans,
                                    num_bins=1,  # Original default = 8
                                    tail_bound=10,  # Original default = 3
                                    dropout_probability=0,
                                    init_identity=True,
                                )
                            ]
                        elif self.flow_type == "Residual":
                            net0 = nf.nets.LipschitzMLP(
                                [1]
                                + [self.hidden_neurons_trans] * self.resid_layers
                                + [1],  # In, hidden x layers, out
                                init_zeros=True,
                                lipschitz_const=0.98,
                            )
                            flows0 += [nf.flows.Residual(net0, reduce_memory=True)]
                            net1 = nf.nets.LipschitzMLP(
                                [1] + [self.hidden_neurons_trans] * self.resid_layers + [1],
                                init_zeros=True,
                                lipschitz_const=0.98,
                            )
                            flows1 += [nf.flows.Residual(net1, reduce_memory=True)]

                    self.flows0 = nn.ModuleList(flows0)
                    self.flows1 = nn.ModuleList(flows1)

        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.n_flows == 0:
            if self.metalearner == "S":
                optimizer = optim.Adam(
                    [
                        {"params": self.balancer.parameters()},
                        {"params": self.prior_encoder.parameters()},
                        {"params": self.cond_mean.parameters()},
                        {"params": self.cond_std.parameters()},
                    ],
                    lr=self.lr,
                    weight_decay=self.lambda_l2,
                )
            elif self.metalearner == "T":
                optimizer = optim.Adam(
                    [
                        {"params": self.balancer.parameters()},
                        {"params": self.prior_encoder0.parameters()},
                        {"params": self.prior_encoder1.parameters()},
                        {"params": self.cond_mean0.parameters()},
                        {"params": self.cond_mean1.parameters()},
                        {"params": self.cond_std0.parameters()},
                        {"params": self.cond_std1.parameters()},
                    ],
                    lr=self.lr,
                    weight_decay=self.lambda_l2,
                )
        else:
            if self.metalearner == "S":
                optimizer = optim.Adam(
                    [
                        {"params": self.balancer.parameters()},
                        {"params": self.prior_encoder.parameters()},
                        {"params": self.cond_mean.parameters()},
                        {"params": self.cond_std.parameters()},
                        {"params": self.flows.parameters()},
                    ],
                    lr=self.lr,
                    weight_decay=self.lambda_l2,
                )
            elif self.metalearner == "T":
                optimizer = optim.Adam(
                    [
                        {"params": self.balancer.parameters()},
                        {"params": self.prior_encoder0.parameters()},
                        {"params": self.prior_encoder1.parameters()},
                        {"params": self.cond_mean0.parameters()},
                        {"params": self.cond_mean1.parameters()},
                        {"params": self.cond_std0.parameters()},
                        {"params": self.cond_std1.parameters()},
                        {"params": self.flows0.parameters()},
                        {"params": self.flows1.parameters()},
                    ],
                    lr=self.lr,
                    weight_decay=self.lambda_l2,
                )
        return optimizer
        # return optim.RMSprop(self.parameters(), lr=self.lr)

    def get_conditional_prior(self, x, t):
        # Get N(mu, sigma) based on x and t
        # Also return balanced x for conditional flow
        x_bal = self.balancer(x)
        xt = torch.cat((x_bal, t[:, np.newaxis]), -1)
        if self.metalearner == "S":
            xt_latent = self.prior_encoder(xt)
            mu = self.cond_mean(xt_latent)
            log_std = self.cond_std(xt_latent)
        elif self.metalearner == "T":
            # Initialize:
            xt_latent = torch.zeros((len(x), self.hidden_neurons_encoder), device=x.device)
            mu = torch.zeros((len(x), 1), device=x.device)
            log_std = torch.zeros((len(x), 1), device=x.device)
            # If shared encoder
            if not (self.hidden_layers_encoder == 0):
                # t = 0:
                xt_latent[t == 0] = self.prior_encoder0(xt[t == 0, :])
                mu[t == 0] = self.cond_mean0(xt_latent[t == 0, :])
                log_std[t == 0] = self.cond_std0(xt_latent[t == 0, :])
                # t = 1:
                xt_latent[t == 1] = self.prior_encoder1(xt[t == 1, :])
                mu[t == 1] = self.cond_mean1(xt_latent[t == 1, :])
                log_std[t == 1] = self.cond_std1(xt_latent[t == 1, :])
            # Else no shared encoder:
            else:
                # t = 0:
                mu[t == 0] = self.cond_mean0(xt[t == 0, :])
                log_std[t == 0] = self.cond_std0(xt[t == 0, :])
                # t = 1:
                mu[t == 1] = self.cond_mean1(xt[t == 1, :])
                log_std[t == 1] = self.cond_std1(xt[t == 1, :])
        return mu, log_std, x_bal

    def encode(self, x_bal, y, t):  # From y to latent z
        # Given a potential outcome, return the latent representation z and the log determinant Jacobian (ldj)
        # Initialize z:
        z = y.clone()
        # Encode y to latent z with normalizing flow
        if self.n_flows == 0:
            return y, torch.zeros_like(y)
        else:
            # If context/condition is used in the flow:
            if self.flow_type == "SigmoidXT":
                xt = torch.cat((x_bal, t[:, np.newaxis]), -1)
            # Flow:
            if self.metalearner == "S":
                if self.flow_type == "SigmoidXT":
                    z, ldj = self.flows.forward_logdet(context=xt[:, None, :], x=z)
                    # if self.flows.dense:
                    #     ldj = ldj.flatten(start_dim=1)
                elif self.flow_type == "SigmoidT":
                    z, ldj = self.flows.forward_logdet(context=t[:, None, None], x=z)
                elif self.flow_type == "GF":
                    # Initialize logdet
                    ldj = torch.zeros((len(y), 1), dtype=y.dtype, device=y.device)
                    cur_datapoints = self.cur_datapoints  # Used for initialization
                    if not self.bin_outcome:
                        ldj = ldj[:, 0]
                    for flow in self.flows:
                        if flow.__module__ == "quantization":
                            z, ldj = flow(z, ldj, reverse=False)
                            ldj = ldj[:, 0]
                        else:
                            z, ldj, cur_datapoints = flow(
                                [z, ldj, cur_datapoints], process_size=self.datapoint_num
                            )
                # elif self.flow_type == 'Residual':
                #     ldj = torch.zeros((1), dtype=y.dtype, device=y.device)
                #     for flow in self.flows:
                #         z, log_det = flow.forward(z)
                #         ldj += log_det[:, None]
                else:
                    # Initialize logdet
                    ldj = torch.zeros((len(y), 1), dtype=y.dtype, device=y.device)
                    for flow in self.flows:
                        z, log_det = flow.forward(z)
                        ldj += log_det[:, None]
                return z, ldj
            elif self.metalearner == "T":
                if self.dense or self.flow_type == "Residual":
                    ldj = torch.zeros((len(y), 1), dtype=y.dtype, device=y.device)
                else:
                    ldj = torch.zeros((len(y)), dtype=y.dtype, device=y.device)
                if self.flow_type == "SigmoidX":
                    if not t.mean() == 1:
                        z[t == 0], ldj[t == 0] = self.flows0.forward_logdet(
                            context=x_bal[t == 0][:, None, :], x=z[t == 0]
                        )
                    if not t.mean() == 0:
                        z[t == 1], ldj[t == 1] = self.flows1.forward_logdet(
                            context=x_bal[t == 1][:, None, :], x=z[t == 1]
                        )
                elif self.flow_type == "SigmoidT":
                    if not t.mean() == 1:
                        z[t == 0], ldj[t == 0] = self.flows0.forward_logdet(
                            context=t[t == 0][:, None, None], x=z[t == 0]
                        )
                    if not t.mean() == 0:
                        z[t == 1], ldj[t == 1] = self.flows1.forward_logdet(
                            context=t[t == 1][:, None, None], x=z[t == 1]
                        )
                elif self.flow_type == "Sigmoid":
                    if not t.mean() == 1:
                        z[t == 0], ldj[t == 0] = self.flows0.forward_logdet(
                            context=torch.Tensor([]), x=z[t == 0]
                        )
                    if not t.mean() == 0:
                        z[t == 1], ldj[t == 1] = self.flows1.forward_logdet(
                            context=torch.Tensor([]), x=z[t == 1]
                        )
                # Todo: Other flows not yet implemented for T-learner:
                elif self.flow_type == "GF":
                    cur_datapoints = self.cur_datapoints  # Used for initialization
                    for flow in self.flows0:
                        z[t == 0], ldj[t == 0], cur_datapoints = flow(
                            [z[t == 0], ldj[t == 0], cur_datapoints],
                            process_size=self.datapoint_num,
                        )
                    for flow in self.flows1:
                        z[t == 1], ldj[t == 1], cur_datapoints = flow(
                            [z[t == 1], ldj[t == 1], cur_datapoints],
                            process_size=self.datapoint_num,
                        )
                elif self.flow_type == "RQNSF-AR":
                    if not t.mean() == 1.0:
                        for flow in self.flows0:
                            z[t == 0], log_det = flow.forward(z[t == 0])
                            ldj[t == 0] += log_det
                    if not t.mean() == 0.0:
                        for flow in self.flows1:
                            z[t == 1], log_det = flow.forward(z[t == 1])
                            ldj[t == 1] += log_det
                else:
                    if not t.mean() == 1.0:
                        for flow in self.flows0:
                            z[t == 0], log_det = flow.forward(z[t == 0])
                            ldj[t == 0] += log_det[:, None]
                    if not t.mean() == 0.0:
                        for flow in self.flows1:
                            z[t == 1], log_det = flow.forward(z[t == 1])
                            ldj[t == 1] += log_det[:, None]
                # else:
                #     raise NotImplementedError('Other flows not yet implemented for T-learner')
                return z, ldj

    def _get_log_likelihood(self, y, t, mu, log_std, x_bal, return_z=False):
        z, ldj = self.encode(x_bal, y, t)
        # Get NLL:
        log_pz = self._get_log_prob_normal(z=z, mu=mu, log_std=log_std)
        # calculate the log_px via change of variables formula
        log_px = ldj + log_pz
        if return_z:
            return log_px, z
        else:
            return log_px

    def _get_log_prob_normal(self, z, mu, log_std):
        var = (torch.exp(log_std) ** 2) + 1e-8
        return -torch.pow(z - mu, 2) / (2 * var) - 0.5 * torch.log(2 * np.pi * var)

    # def _get_log_prob_beta(self, z, alpha, beta):
    #     lls = torch.zeros_like(z)
    #     for i in range(len(z)):
    #         if z[i, 0] < 0 or z[i, 0] > 1:
    #             lls[i, 0] = torch.inf
    #         else:
    #             dist = torch.distributions.Beta(alpha[i], beta[i])
    #             lls[i, 0] = dist.log_prob(z[i, 0])
    #     return lls

    @torch.no_grad()
    def decode(self, z, x, t):  # Inference, sampling
        xt = torch.cat((x, t[:, None]), -1)
        y_est = z.clone().to(self.device)
        if self.n_flows > 0:
            if self.metalearner == "S":
                if self.flow_type == "SigmoidXT":
                    y_est = self.flows.inverse(
                        context=xt[:, None, :],
                        u=z,
                        max_iter=self.max_iter,
                        precision=1e-4,
                        max_left=self.max_left,
                        max_right=self.max_right,
                    )
                    # To check:
                    # z = self.flows.forward_no_logdet(context=context[:, None, :], x=y_est)
                elif self.flow_type == "SigmoidT":
                    y_est = self.flows.inverse(
                        context=xt[:, -1][:, None, None],
                        u=z,
                        max_iter=self.max_iter,
                        precision=1e-4,
                        max_left=self.max_left,
                        max_right=self.max_right,
                    ).to(self.device)
                # elif self.flow_type == 'Sigmoid':
                #     y_est = self.flows.inverse(context=None, u=z, max_iter=self.max_iter, precision=1e-4,
                #                                max_left=self.max_left, max_right=self.max_right)
                elif self.flow_type == "GF":
                    # datapoints_array = []
                    # cur_datapoints = self.cur_datapoints
                    # process_size = self.datapoint_num
                    # datapoints_array.append(cur_datapoints)

                    # for i in range(1 // process_size):
                    #     for l, flow in enumerate(reversed(self.flows)):
                    #         y_est[i * process_size: (i + 1) * process_size, :] = flow.sampling(
                    #             y_est[i * process_size: (i + 1) * process_size, :])
                    for l, flow in enumerate(reversed(self.flows)):
                        y_est = flow.sampling(y_est, verbose=False)
                else:
                    for flow in reversed(self.flows):
                        y_est, _ = flow.inverse(y_est)
            elif self.metalearner == "T":
                # Decode per flow - 0 / 1
                if self.flow_type == "SigmoidX":
                    y_est[t == 0] = self.flows0.inverse(
                        context=x[t == 0][:, None, :],
                        u=y_est[t == 0],
                        max_iter=self.max_iter,
                        precision=1e-4,
                        max_left=self.max_left,
                        max_right=self.max_right,
                    ).to(self.device)
                    y_est[t == 1] = self.flows1.inverse(
                        context=x[t == 1][:, None, :],
                        u=y_est[t == 1],
                        max_iter=self.max_iter,
                        precision=1e-4,
                        max_left=self.max_left,
                        max_right=self.max_right,
                    ).to(self.device)
                    # To check:
                    # z_est = y_est
                    # z_est[t == 0] = self.flows0.forward_no_logdet(context=x[t == 0][:, None, :], x=y_est[t == 0])
                    # z_est[t == 1] = self.flows0.forward_no_logdet(context=x[t == 1][:, None, :], x=y_est[t == 1])
                    # print('Inversion error:', ((z_est - z)**2).mean())
                elif self.flow_type == "SigmoidT":
                    y_est[t == 0] = self.flows0.inverse(
                        context=t[t == 0][:, None, None],
                        u=y_est[t == 0],
                        max_iter=self.max_iter,
                        precision=1e-4,
                        max_left=self.max_left,
                        max_right=self.max_right,
                    ).to(self.device)
                    y_est[t == 1] = self.flows1.inverse(
                        context=t[t == 1][:, None, None],
                        u=y_est[t == 1],
                        max_iter=self.max_iter,
                        precision=1e-4,
                        max_left=self.max_left,
                        max_right=self.max_right,
                    ).to(self.device)
                elif self.flow_type == "Sigmoid":
                    y_est[t == 0] = self.flows0.inverse(
                        context=torch.Tensor([]),
                        u=y_est[t == 0],
                        max_iter=self.max_iter,
                        precision=1e-4,
                        max_left=self.max_left,
                        max_right=self.max_right,
                    ).to(self.device)
                    y_est[t == 1] = self.flows1.inverse(
                        context=torch.Tensor([]),
                        u=y_est[t == 1],
                        max_iter=self.max_iter,
                        precision=1e-4,
                        max_left=self.max_left,
                        max_right=self.max_right,
                    ).to(self.device)
                elif self.flow_type == "GF":
                    # datapoints_array = []
                    # cur_datapoints = self.cur_datapoints
                    # process_size = self.datapoint_num
                    # datapoints_array.append(cur_datapoints)

                    # for i in range(len(y_est[t == 0]) // process_size):
                    #     for l, flow in enumerate(reversed(self.flows0)):
                    #         y_est[t == 0][i * process_size: (i + 1) * process_size, :] = flow.sampling(
                    #             y_est[t == 0][i * process_size: (i + 1) * process_size, :])
                    # for i in range(len(y_est[t == 1]) // process_size):
                    #     for l, flow in enumerate(reversed(self.flows1)):
                    #         y_est[t == 1][i * process_size: (i + 1) * process_size, :] = flow.sampling(
                    #             y_est[t == 1][i * process_size: (i + 1) * process_size, :])
                    for l, flow in enumerate(reversed(self.flows0)):
                        y_est[t == 0] = flow.sampling(y_est[t == 0], verbose=False)
                    for l, flow in enumerate(reversed(self.flows1)):
                        y_est[t == 1] = flow.sampling(y_est[t == 1], verbose=False)
                # Other flows not yet implemented
                else:
                    if not t.mean() == 1.0:
                        for flow in reversed(self.flows0):
                            y_est[t == 0], _ = flow.inverse(y_est[t == 0]).to(self.device)
                    if not t.mean() == 0.0:
                        for flow in reversed(self.flows1):
                            y_est[t == 1], _ = flow.inverse(y_est[t == 1]).to(self.device)
        return y_est

    def training_step(self, train_batch, batch_idx):
        """
        Normalizing flows are trained by maximum likelihood => return loss
        :param train_batch: int=64
        :param batch_idx: amount of loops
        :return: training loss
        """
        # Make layers Lipschitz continuous
        if self.flow_type == "Residual" and self.n_flows > 0:
            if self.metalearner == "S":
                nf.utils.update_lipschitz(self.flows, 50)
            elif self.metalearner == "T":
                nf.utils.update_lipschitz(self.flows0, 50)
                nf.utils.update_lipschitz(self.flows1, 50)

        # Forward pass and calculate losses
        x, y, t = train_batch
        # Noise regularization to x and y:
        x = x + torch.normal(0, self.noise_reg_x, x.shape, device=x.device)
        y = y + torch.normal(0, self.noise_reg_y, y.shape, device=y.device)
        # Get the conditional prior based on balanced x:
        mu, log_std, x_bal = self.get_conditional_prior(x, t)
        # Get the NLL for the normalizing flow
        ll, z = self._get_log_likelihood(y, t, mu, log_std, x_bal, return_z=True)
        nll = -torch.mean(ll)
        self.log("NLL", nll)
        train_loss = nll
        # Add MSE loss:
        # alpha = 1. / (self.global_step + 1)  # Steep decline
        # alpha = 1. / np.sqrt(self.global_step + 1)  # Smooth decline
        # alpha = 1.  # Constant
        # Add supervised MSE loss on mu:
        # mse_mu = F.mse_loss(z, mu, reduction='mean')
        # self.log('MSE_mu', mse_mu)
        # train_loss += alpha * mse_mu
        # train_loss = mse_mu
        # Add supervised MSE loss on output (slow):
        # y_hat = self.decode(z=mu, x=x_bal, t=t)
        # mse_y = F.mse_loss(y, y_hat, reduction='mean')
        # self.log('MSE_y', mse_y)
        # train_loss += alpha * mse_y
        # Calculate MMD:
        loss_mmd = calcMMD(x_bal, t)
        self.log("balancing_train", loss_mmd)
        train_loss = train_loss + self.lambda_mmd * loss_mmd
        # Add l1 regularization:
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        train_loss = train_loss + self.lambda_l1 * l1_norm
        self.log("train_loss", train_loss)

        return train_loss

    def validation_step(self, val_batch, batch_idx):
        try:
            x, y, t = val_batch
        except ValueError:  # In case counterfactuals are included - do not use during validation!
            x, y, _, t = val_batch
        # Get the conditional prior (based on balanced x):
        mu, log_std, x_bal = self.get_conditional_prior(x, t)
        # Get the loglikelihood
        loss, z = self._get_log_likelihood(y, t, mu, log_std, x_bal, return_z=True)
        val_loss = -torch.mean(loss)
        self.log("NLL_val", val_loss)
        # self.log('MSE_z_val', torch.mean((z - y) ** 2))
        # Calculate MMD
        loss_mmd = calcMMD(x_bal, t)
        self.log("balancing_val", loss_mmd)
        val_loss = val_loss + self.lambda_mmd * loss_mmd
        self.log("val_loss", val_loss)
        # MSE observed outcome only
        y_pred = self.decode(mu, x_bal, t)
        self.log("MSE_val", torch.mean((y_pred - y) ** 2))

        # Get counterfactual outcome:
        mu_cf, log_std_cf, x_bal_cf = self.get_conditional_prior(x, 1 - t)
        # loss, z = self._get_log_likelihood(y, 1 - t, mu_cf, log_std_cf, x_bal_cf)
        y_pred_cf = self.decode(mu_cf, x_bal_cf, 1 - t)
        # Get PEHE proxy based on 1-nearest neighbour
        pehe_proxy = pehe_nn(yf_p=y_pred[:, 0], ycf_p=y_pred_cf[:, 0], y=y[:, 0], x=x, t=t)
        self.log("PEHE_nn", pehe_proxy)

        # TODO: Get NLL proxy based on 1-nearest neighbour?

        return val_loss

    def test_step(self, test_batch, batch_idx):
        x, yf, yc, t = test_batch
        # Get the conditional prior based on balanced x:
        mu_f, log_std_f, x_bal_f = self.get_conditional_prior(x, t)
        mu_c, log_std_c, x_bal_c = self.get_conditional_prior(x, 1 - t)
        # Get the NLL for the ITE
        loss = -self._get_log_likelihood(yf, t, mu_f, log_std_f, x_bal_f)
        loss += -self._get_log_likelihood(yc, 1 - t, mu_c, log_std_c, x_bal_c)
        loss = loss / 2  # To scale
        test_loss = torch.mean(loss)
        # Calculate MMD - always 0
        # loss_mmd = lossfunctions.calcMMD(x_bal_f, t)
        # loss_mmd += lossfunctions.calcMMD(x_bal_c, 1-t)
        # loss_mmd = loss_mmd / 2
        # self.log('balancing_test', loss_mmd)
        # test_loss = test_loss + self.lambda_mmd * loss_mmd
        self.log("test_loss", test_loss)
        # self.log('NLL_test', loss)
        if self.sweep:
            # PEHE and MSE
            outcomes_test = np.zeros_like(np.hstack((yf, yc)))
            idx0, idx1 = np.where(t == 0), np.where(t == 1)
            outcomes_test[idx0] = np.hstack((yf[idx0], yc[idx0]))
            outcomes_test[idx1] = np.hstack((yc[idx1], yf[idx1]))
            # Get predictions
            mu0, _, x_bal0 = self.get_conditional_prior(x=x, t=torch.zeros(len(x)), device=x.device)
            mu1, _, x_bal1 = self.get_conditional_prior(x=x, t=torch.ones(len(x), device=x.device))
            y0_pred = self.decode(mu0.float(), x_bal0, torch.zeros(len(x)), device=x.device)
            y1_pred = self.decode(mu1.float(), x_bal1, torch.ones(len(x)), device=x.device)
            outcomes_pred = np.stack((y0_pred[:, 0], y1_pred[:, 0]), axis=-1)
            pehe = np.mean(
                np.square(
                    (outcomes_test[:, 1] - outcomes_test[:, 0])
                    - (outcomes_pred[:, 1] - outcomes_pred[:, 0])
                )
            )
            self.log("PEHE", pehe)
            self.log("PO_MSE", np.mean(np.square(outcomes_test - outcomes_pred)))

        return test_loss


class NOFLITE:

    def __init__(self, params):
        self.params = params

    def fit(self, X, Y, W):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        logging.info(f"NOFLITE is using device: {self.device}")
        self.params["cur_datapoints"] = torch.Tensor(Y.reshape((-1, 1)))[
            torch.randint(low=0, high=len(Y), size=(self.params["datapoint_num"], 1)), 0
        ]
        self.model = Noflite(params=self.params)
        self.trainer = pl.Trainer(
            accelerator=self.device,
            max_steps=self.params["max_steps"],
            logger=False,
            gradient_clip_val=10.0,
            callbacks=None,
            enable_checkpointing=False,
        )
        train_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y).view(-1, 1), torch.Tensor(W))
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )
        self.trainer.fit(self.model, train_dataloader)

    def sample_y0_y1(self, X, n_samples=500):
        self.model.eval()
        self.model.to(self.device)

        # Yf = Y0 * (1 - W) + Y1 * W
        # Ycf = Y0 * W + Y1 * (1 - W)
        # self.trainer.test(self.model, dataloaders=DataLoader(TensorDataset(X, Yf.reshape(-1,1), Ycf.reshape(-1,1), W.reshape(-1,1)), batch_size=self.params['batch_size']))

        # Take samples:
        # First, set max iter for sampling sufficiently high (only for Sigmoid flows)
        self.model.max_iter = 2000
        # Evaluation -- Get conditional prior's parameter (per instance)
        mu0, logsig0, x_bal0 = self.model.get_conditional_prior(
            x=torch.from_numpy(X).float().to(device=self.device),
            t=torch.zeros(len(X), device=self.device),
        )
        mu1, logsig1, x_bal1 = self.model.get_conditional_prior(
            x=torch.from_numpy(X).float().to(self.device), t=torch.ones(len(X), device=self.device)
        )
        sig0 = torch.exp(logsig0)
        sig1 = torch.exp(logsig1)

        # Calculate Y_estimated by using inverse flows and µ(x,t)
        n_samples = self.params["n_samples"]
        y0_posteriors = np.zeros((len(X), n_samples))
        y1_posteriors = np.zeros((len(X), n_samples))
        trunc_prob = self.params["trunc_prob"]
        if trunc_prob < 0.5:
            trunc_prob = 1 - trunc_prob
        # Sample from truncated normal for more stable results
        trunc_perc = norm.ppf(trunc_prob)
        for i in tqdm(range(len(X))):
            # t=0
            y0_prior_samples = (
                torch.Tensor(truncnorm.rvs(a=-trunc_perc, b=trunc_perc, size=n_samples)).to(
                    self.device
                )
                * sig0[i, :]
                + mu0[i, :]
            )  # Transform to samples from conditional prior
            y0_posterior = self.model.decode(
                z=y0_prior_samples[:, None],
                x=x_bal0[i, :][None, :].repeat((n_samples, 1)),
                t=torch.zeros(n_samples, device=self.device),
            )
            y0_posteriors[i, :] = y0_posterior.cpu().detach().numpy()[:, 0]
            # t=1
            y1_prior_samples = (
                torch.Tensor(truncnorm.rvs(a=-trunc_perc, b=trunc_perc, size=n_samples)).to(
                    self.device
                )
                * sig1[i, :]
                + mu1[i, :]
            )
            y1_posterior = self.model.decode(
                z=y1_prior_samples[:, None],
                x=x_bal1[i, :][None, :].repeat((n_samples, 1)),
                t=torch.ones(n_samples, device=self.device),
            )
            y1_posteriors[i, :] = y1_posterior.cpu().detach().numpy()[:, 0]
        return y0_posteriors, y1_posteriors

    def evaluate(self, X, Y0, Y1, W, n_samples=500, alpha=0.05, return_p_values=False):
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
        y0_pred, y1_pred = self.sample_y0_y1(X, n_samples)
        # Calculate ITE
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
