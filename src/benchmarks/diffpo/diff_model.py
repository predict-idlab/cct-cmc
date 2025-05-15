import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=180, hidden_dim=100, output_dim=180):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.config = config
        self.channels = config["channels"]
        self.cond_dim = config["cond_dim"]
        self.hidden_dim = config["hidden_dim"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.token_emb_dim = config["token_emb_dim"] if config["mixed"] else 1
        inputdim = 2 * self.token_emb_dim

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = nn.Linear(self.cond_dim, self.hidden_dim)
        self.output_projection2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.y0_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.y1_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.output_y0 = nn.Linear(self.hidden_dim, 1)
        self.output_y1 = nn.Linear(self.hidden_dim, 1)

        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    cond_dim=self.cond_dim,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        x = x.squeeze(1)
        B, cond_dim = x.shape
        x = F.relu(x)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = F.relu(x)

        y0 = self.y0_layer(x)
        y0 = F.relu(y0)
        y0 = self.output_y0(y0)
        y1 = self.y1_layer(x)
        y1 = F.relu(y1)
        y1 = self.output_y1(y1)

        x = torch.cat((y0, y1), 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, cond_dim, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, cond_dim)
        self.mid_projection = nn.Linear(cond_dim, cond_dim * 2)
        self.output_projection = nn.Linear(cond_dim, cond_dim * 2)
        self.output_projection_for_x = nn.Linear(cond_dim, 2)
        self.time_layer = MLP(input_dim=cond_dim, output_dim=cond_dim)
        self.feature_layer = MLP(input_dim=cond_dim, output_dim=cond_dim)

    def forward_time(self, y, base_shape):
        y = self.time_layer(y)
        return y

    def forward_feature(self, y, base_shape):
        y = self.feature_layer(y)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, cond_dim = x.shape
        base_shape = x.shape
        diffusion_emb = self.diffusion_projection(diffusion_emb)

        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)

        y = y
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip
