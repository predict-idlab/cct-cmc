# Copied from: https://github.com/toonvds/NOFLITE

# Adapted from: https://github.com/ServiceNow/tactis/blob/main/tactis/model/flow.py
# and https://github.com/ServiceNow/tactis/blob/main/tactis/model/marginal.py
# Which was, in turn, adapted from: https://github.com/CW-Huang/torchkit/blob/master/torchkit/flows.py

import math
import torch
from torch import nn
from typing import Tuple

# To be compared to the float32 machine epsilon of 2**-23 ~= 1.2e-7
EPSILON = 1e-6


def log_sum_exp(A, dim=-1, keepdim=False):
    """
    Compute a sum in logarithm space: log(exp(a) + exp(b))
    Properly handle values which exponential cannot be represented in float32.
    """
    max_A = A.max(axis=dim, keepdim=True).values
    norm_A = A - max_A
    result = torch.log(torch.exp(norm_A).sum(axis=dim, keepdim=True)) + max_A
    if keepdim:
        return result
    else:
        return torch.squeeze(result, dim=dim)


def log_sigmoid(x):
    """
    Logarithm of the sigmoid function.
    Substract the epsilon to avoid 0 as a possible value for large x.
    """
    return -nn.functional.softplus(-x) - EPSILON


class SigmoidFlow(nn.Module):
    # Indices:
    # b: batch
    # s: sample
    # o: output dimension
    # h: hidden dimension
    # i: input dimension

    def __init__(self, hidden_dim: int, no_logit: bool = False):
        """
        A single layer of the Deep Sigmoid Flow network.
        Does not contains its parameters, they must be sent in the forward method.
        Parameters:
        -----------
        hidden_dim: uint
            The number of hidden units
        no_logit: bool, default to False
            If set to True, then the network will return a value in the (0, 1) interval.
            If kept to False, then the network will apply a logit function to this value to return
            a value in the (-inf, inf) interval.
        """
        super(SigmoidFlow, self).__init__()
        self.hidden_dim = hidden_dim
        self.no_logit = no_logit

        self.act_a = lambda x: nn.functional.softplus(x) + EPSILON
        self.act_b = lambda x: x
        self.act_w = lambda x: nn.functional.softmax(x, dim=-1)

    def forward(self, params, x, logdet):
        """
        Transform the given value according to the given parameters,
        computing the derivative of the transformation at the same time.
        params third dimension must be equal to 3 times the number of hidden units.
        """
        # Indices:
        # b: batch
        # v: variables
        # h: hidden dimension

        # params: b, v, 3*h
        # x: b, v
        # logdet: b
        # output x: b, v
        # output logdet: b
        assert params.shape[-1] == 3 * self.hidden_dim

        a = self.act_a(params[..., : self.hidden_dim])  # b, v, h
        b = self.act_b(params[..., self.hidden_dim : 2 * self.hidden_dim])  # b, v, h
        pre_w = params[..., 2 * self.hidden_dim :]  # b, v, h
        w = self.act_w(pre_w)  # b, v, h

        pre_sigm = a * x[..., None] + b  # b, v, h
        sigm = torch.sigmoid(pre_sigm)  # b, v, h
        x_pre = (w * sigm).sum(dim=-1)  # b, v

        logj = (
            nn.functional.log_softmax(pre_w, dim=-1) + log_sigmoid(pre_sigm) + log_sigmoid(-pre_sigm) + torch.log(a)
        )  # b, v, h

        logj = log_sum_exp(logj, dim=-1, keepdim=False)  # b, v

        if self.no_logit:
            # Only keep the batch dimension, summing all others in case this method is called with more dimensions
            logdet = logj.sum(dim=tuple(range(1, logj.dim()))) + logdet
            return x_pre, logdet

        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5  # b, v
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)  # b, v

        logdet_ = logj + math.log(1 - EPSILON) - (torch.log(x_pre_clipped) + torch.log(-x_pre_clipped + 1))  # b, v

        # Only keep the batch dimension, summing all others in case this method is called with more dimensions
        logdet = logdet_.sum(dim=tuple(range(1, logdet_.dim()))) + logdet

        return xnew, logdet

    def forward_no_logdet(self, params, x):
        """
        Transform the given value according to the given parameters,
        but does not compute the derivative of the transformation.
        params third dimension must be equal to 3 times the number of hidden units.
        """
        # Indices: (when used for inversion)
        # b: batch
        # s: samples
        # h: hidden dimension

        # params: b, s, 3*h
        # x: b, s
        # output x: b, s
        assert params.shape[-1] == 3 * self.hidden_dim

        a = self.act_a(params[..., : self.hidden_dim])  # b, s, h
        b = self.act_b(params[..., self.hidden_dim : 2 * self.hidden_dim])  # b, s, h
        pre_w = params[..., 2 * self.hidden_dim :]  # b, s, h
        w = self.act_w(pre_w)  # b, s, h

        pre_sigm = a * x[..., None] + b  # b, s, h
        sigm = torch.sigmoid(pre_sigm)  # b, s, h
        x_pre = (w * sigm).sum(dim=-1)  # b, s

        if self.no_logit:
            return x_pre

        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5  # b, s
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)  # b, s

        return xnew


# Adapted to match TACTiS code from: https://github.com/CW-Huang/torchkit/blob/master/torchkit/flows.py#L418
class DenseSigmoidFlow(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DenseSigmoidFlow, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.act_a = lambda x: nn.functional.softplus(x) + EPSILON
        self.act_b = lambda x: x
        self.act_w = lambda x: nn.functional.softmax(x, dim=-1)
        self.act_u = lambda x: nn.functional.softmax(x, dim=-1)

        self.u_ = torch.nn.parameter.Parameter(torch.Tensor(hidden_dim, in_dim))
        self.w_ = torch.nn.parameter.Parameter(torch.Tensor(out_dim, hidden_dim))

        self.reset_parameters()

        # Operations:
        self.log = lambda x: torch.log(x * 1e2) - math.log(1e2)
        self.logsigmoid = lambda x: -nn.Softplus()(-x) + EPSILON

    def reset_parameters(self):
        self.u_.data.uniform_(-0.001, 0.001)
        self.w_.data.uniform_(-0.001, 0.001)

    def oper(self, array, oper, axis=-1, keepdims=False):
        a_oper = oper(array)
        if keepdims:
            shape = []
            for j, s in enumerate(array.size()):
                shape.append(s)
            shape[axis] = -1
            a_oper = a_oper.view(*shape)
        return a_oper

    def log_sum_exp(self, A, axis=-1, sum_op=torch.sum):
        maximum = lambda x: x.max(axis)[0]
        A_max = self.oper(A, maximum, axis, True)
        summation = lambda x: sum_op(torch.exp(x - A_max), axis)
        B = torch.log(self.oper(A, summation, axis, True)) + A_max
        return B

    def forward(self, dsparams, x, logdet):
        if logdet.ndim == 1:
            logdet = logdet[:, None, None, None]
        if dsparams.ndim == 1:
            dsparams = dsparams.repeat(len(x), 1, 1)

        inv = math.log(math.exp(1 - EPSILON) - 1)
        ndim = self.hidden_dim

        # Get parameters:
        pre_u = self.u_[None, None, :, :] + dsparams[:, :, -self.in_dim:][:, :, None, :]
        pre_w = self.w_[None, None, :, :] + dsparams[:, :, 2 * ndim:3 * ndim][:, :, None, :]
        a = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim] + inv)
        b = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(pre_w)
        u = self.act_u(pre_u)

        pre_sigm = torch.sum(u * a[:, :, :, None] * x[:, None, None, :], -1) + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm[:, :, None, :], dim=-1)
        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5
        xnew = self.log(x_pre_clipped) - self.log(1 - x_pre_clipped)
        xnew = xnew[:, 0, :]

        logj = nn.functional.log_softmax(pre_w, dim=3) + \
               self.logsigmoid(pre_sigm[:, :, None, :]) + \
               self.logsigmoid(-pre_sigm[:, :, None, :]) + \
               self.log(a[:, :, None, :])
        # n, d, d2, dh

        logj = logj[:, :, :, :, None] + nn.functional.log_softmax(pre_u, dim=3)[:, :, None, :, :]
        # n, d, d2, dh, d1

        # logj = log_sum_exp(logj, dim=-1, keepdim=False)
        logj = self.log_sum_exp(logj, 3).sum(3)
        # n, d, d2, d1

        logdet_ = logj + math.log(1 - EPSILON) - (self.log(x_pre_clipped) + self.log(-x_pre_clipped + 1))[:, :, :, None]

        logdet = self.log_sum_exp(logdet_[:, :, :, :, None] + logdet[:, :, None, :, :], 3).sum(3)
        # logdet = logdet[:, 0, 0, 0]
        # logdet = logdet_.sum(dim=tuple(range(1, logdet_.dim()))) + logdet
        # n, d, d2, d1, d0 -> n, d, d2, d0

        return xnew, logdet

    def forward_no_logdet(self, dsparams, x):
        if dsparams.ndim == 1:
            dsparams = dsparams.repeat(len(x), 1, 1)

        inv = math.log(math.exp(1 - EPSILON) - 1)
        ndim = self.hidden_dim

        # Get parameters:
        pre_u = self.u_[None, None, :, :] + dsparams[:, :, -self.in_dim:][:, :, None, :]
        pre_w = self.w_[None, None, :, :] + dsparams[:, :, 2 * ndim:3 * ndim][:, :, None, :]
        a = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim] + inv)
        b = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(pre_w)
        u = self.act_u(pre_u)

        pre_sigm = torch.sum(u * a[:, :, :, None] * x[:, None, None, :], -1) + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm[:, :, None, :], dim=-1)
        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5
        xnew = self.log(x_pre_clipped) - self.log(1 - x_pre_clipped)
        xnew = xnew[:, 0, :]

        return xnew


class DeepSigmoidFlow(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int, no_logit: bool, dense: bool):
        """
        A Deep Sigmoid Flow network, made of multiple Sigmoid Flow layers.
        This model a flexible transformation from any real values into the (0, 1) interval.
        Does not contains its parameters, they must be sent in the forward method.
        Parameters:
        -----------
        n_layers: uint
            The number of sigmoid flow layers
        hidden_dim: uint
            The number of hidden units
        """
        super(DeepSigmoidFlow, self).__init__()

        self.dense = dense

        if dense:
            self.params_length = 4 * hidden_dim     # Slight overestimate (first/last layer only has 3 * hidden_dim + 1)

            if n_layers == 1:
                elayers = [DenseSigmoidFlow(in_dim=1, hidden_dim=hidden_dim, out_dim=1)]
            else:
                elayers = nn.ModuleList([DenseSigmoidFlow(in_dim=1, hidden_dim=hidden_dim, out_dim=hidden_dim) for _ in range(n_layers - 1)])
                elayers += [DenseSigmoidFlow(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=1)]
        else:
            self.params_length = 3 * hidden_dim

            elayers = nn.ModuleList([SigmoidFlow(hidden_dim) for _ in range(n_layers - 1)])
            elayers += [SigmoidFlow(hidden_dim, no_logit=no_logit)]
        self.layers = nn.Sequential(*elayers)

    @property
    def total_params_length(self):
        return len(self.layers) * self.params_length

    def forward(self, params, x):
        """
        Transform the given value according to the given parameters,
        computing the derivative of the transformation at the same time.
        params third dimension must be equal to total_params_length.
        """
        # params: batches, variables, params dim
        # x: batches, variables
        logdet = torch.zeros(
            x.shape[0],
        ).to(x.device)
        for i, layer in enumerate(self.layers):
            x, logdet = layer(params[..., i * self.params_length: (i + 1) * self.params_length], x, logdet)
        if self.dense:
            logdet = logdet.flatten(start_dim=1)
        return x, logdet

    def forward_no_logdet(self, params, x):
        """
        Transform the given value according to the given parameters,
        but does not compute the derivative of the transformation.
        params third dimension must be equal to total_params_length.
        """
        # params: batches, samples, params dim
        # x: batches, samples
        for i, layer in enumerate(self.layers):
            x = layer.forward_no_logdet(params[..., i * self.params_length: (i + 1) * self.params_length], x)
        return x


class DSFMarginal(nn.Module):
    """
    Compute the marginals using a Deep Sigmoid Flow conditioned using a MLP.
    The conditioning MLP uses the embedding from the encoder as its input.
    """

    def __init__(self, context_dim: int, mlp_layers: int, mlp_dim: int, flow_layers: int, flow_hid_dim: int,
                 no_logit: bool, dense: bool = False):
        """
        Parameters:
        -----------
        context_dim: int
            Size of the context (embedding created by the encoder) that will be sent to the conditioner.
        mlp_layers: int
            Number of layers for the conditioner MLP.
        mlp_dim: int
            Dimension of the hidden layers of the conditioner MLP.
        flow_layers: int
            Number of layers for the Dense Sigmoid Flow.
        flow_hid_dim: int
            Dimension of the hidden layers of the Dense Sigmoid Flow.
        """
        super().__init__()

        self.context_dim = context_dim
        self.mlp_layers = mlp_layers
        self.mlp_dim = mlp_dim
        self.flow_layers = flow_layers
        self.flow_hid_dim = flow_hid_dim

        self.no_logit = no_logit
        self.dense = dense

        self.marginal_flow = DeepSigmoidFlow(n_layers=self.flow_layers, hidden_dim=self.flow_hid_dim,
                                             no_logit=no_logit, dense=dense)

        elayers = [nn.Linear(self.context_dim, self.mlp_dim), nn.ReLU()]
        for _ in range(1, self.mlp_layers):
            elayers += [nn.Linear(self.mlp_dim, self.mlp_dim), nn.ReLU()]
        elayers += [nn.Linear(self.mlp_dim, self.marginal_flow.total_params_length)]
        self.marginal_conditioner = nn.Sequential(*elayers)

        # Identity initialization:
        # - Small weights
        for module in self.marginal_conditioner:
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(-0.001, 0.001)
                module.bias.data.uniform_(-0.001, 0.001)
        # - Output biases of last layer = 0
        self.marginal_conditioner[-1].bias.data.zero_()
        # - Add 1 to a outputs corresponding to a (as ELU(1) = 1), i.e., every third/fourth element
        if dense:
            nn.init.constant_(self.marginal_conditioner[-1].bias[::4].data, 1)
        else:
            nn.init.constant_(self.marginal_conditioner[-1].bias[::3].data, 1)

    def forward_logdet(self, context: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cumulative density function of a marginal conditioned using the given context, for the given value of x.
        Also returns the logarithm of the determinant of this transformation.
        Parameters:
        -----------
        context: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            The series and time steps dimensions are merged.
        x: Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            A tensor containing the value to be transformed using the CDF.
            The series and time steps dimensions are merged.
            If a third dimension is present, then the context is considered to be constant across this dimension.
        Returns:
        --------
        u: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The CDF at the given point, a value between 0 and 1.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of x.
        logdet: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The logarithm of the derivative of the transformation.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of x.
        """
        marginal_params = self.marginal_conditioner(context)
        # If x has both a variable and a sample dimension, then add a singleton dimension to marginal_params to have the correct shape
        if marginal_params.dim() == x.dim():
            marginal_params = marginal_params[:, :, None, :]

        return self.marginal_flow.forward(marginal_params, x)

    def forward_no_logdet(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative density function of a marginal conditioned using the given context, for the given value of x.
        Parameters:
        -----------
        context: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            The series and time steps dimensions are merged.
        x: Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            A tensor containing the value to be transformed using the CDF.
            The series and time steps dimensions are merged.
            If a third dimension is present, then the context is considered to be constant across this dimension.
        Returns:
        --------
        u: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The CDF at the given point, a value between 0 and 1.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of x.
        """
        marginal_params = self.marginal_conditioner(context)
        # If x has both a variable and a sample dimension, then add a singleton dimension to marginal_params to have the correct shape
        if marginal_params.dim() == x.dim():
            marginal_params = marginal_params[:, :, None, :]

        return self.marginal_flow.forward_no_logdet(marginal_params, x)

    def inverse(
            self,
            context: torch.Tensor,
            u: torch.Tensor,
            max_iter: int = 100,
            precision: float = 1e-4,
            max_left: float = -1000.0,
            max_right: float = 1000.0,
    ) -> torch.Tensor:
        """
        Compute the inverse cumulative density function of a marginal conditioned using the given context, for the given value of u.
        This method uses a dichotomic search.
        The gradient of this method cannot be computed, so it should only be used for sampling.
        Parameters:
        -----------
        context: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            The series and time steps dimensions are merged.
        u: Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            A tensor containing the value to be transformed using the inverse CDF.
            The series and time steps dimensions are merged.
            If a third dimension is present, then the context is considered to be constant across this dimension.
        max_iter: int, default = 100
            The maximum number of iterations for the dichotomic search.
            The precision of the result should improve by a factor of 2 at each iteration.
        precision: float, default = 1e-6
            If the difference between CDF(x) and u is less than this value for all variables, stop the search.
        max_value: float, default = 1000.0
            The absolute upper bound on the possible output.
        Returns:
        --------
        x: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The inverse CDF at the given value.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of u.
        """
        # Check if input is empty:
        if u.nelement() == 0:
            return torch.Tensor([])[:, None]

        marginal_params = self.marginal_conditioner(context)
        # If u has both a variable and a sample dimension,
        # then add a singleton dimension to marginal_params to have the correct shape
        if marginal_params.dim() == u.dim():
            marginal_params = marginal_params[:, :, None, :]

        # Make sure borders for search are in order
        assert max_left < 0
        assert max_right > 0

        left = max_left * torch.ones_like(u)
        right = max_right * torch.ones_like(u)

        # Use interpolation search:
        error_left = self.marginal_flow.forward_no_logdet(marginal_params, left) - u
        error_right = self.marginal_flow.forward_no_logdet(marginal_params, right) - u

        # Check if borders are far enough: error_left < 0 and error_right > 0
        # If not, double max distance
        while (error_left > 0).sum() > 0:
            if left.isinf().sum() > 0:
                left[left.isinf()] = -1e3
                break
            left[error_left > 0] = left[error_left > 0] * 1.1
            error_left = self.marginal_flow.forward_no_logdet(marginal_params, left) - u
        while (error_right < 0).sum() > 0:
            if right.isinf().sum() > 0:
                right[right.isinf()] = 1e3
                break
            right[error_right < 0] = right[error_right < 0] * 1.1
            error_right = self.marginal_flow.forward_no_logdet(marginal_params, right) - u

        # plt.plot([left, right], [error_left, error_right])
        # mids = []
        # mid_errors = []

        for _ in range(max_iter):

            # Binary search:
            # mid = (left + right) / 2
            # Interpolation search:
            mid = left + (right - left) * (0 - error_left) / ((error_right - error_left) + 1e-9)
            mid = torch.clamp(mid, min=left, max=right)
            # mids.append(mid)

            error_mid = self.marginal_flow.forward_no_logdet(marginal_params, mid) - u
            # mid_errors.append(error_mid)

            # Update left boundary in case error mid < 0; otherwise update right boundary
            left[error_mid < 0] = mid[error_mid < 0]
            right[error_mid > 0] = mid[error_mid > 0]

            # Similarly, update corresponding errors
            error_left[error_mid < 0] = error_mid[error_mid < 0]
            error_right[error_mid > 0] = error_mid[error_mid > 0]

            max_error = error_mid.abs().max().item()
            if max_error < precision:
                # print(_)
                break

        if _ == max_iter - 1:
            print('\n[MAX ITER] Interpolation search. Max error =', max_error)

        # plt.plot(mids, mid_errors, "-o", alpha=0.2)
        # for iter, (mid, mid_error) in enumerate(zip(mids, mid_errors)):
        #     plt.text(mid, mid_error, str(iter), color="red", fontsize=12)
        # plt.show()

        return mid