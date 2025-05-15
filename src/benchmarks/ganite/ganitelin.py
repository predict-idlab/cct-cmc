import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def xavier_init(fan_in, fan_out, constant=1.0):
    """Xavier initialization of network weights"""
    # Source: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return torch.FloatTensor(fan_in, fan_out).uniform_(low, high)


def batch_generator(data_x, data_t, data_y, batch_size):
    """Random batch generator"""
    num_samples = data_x.shape[0]
    indices = np.random.choice(num_samples, batch_size, replace=False)
    return data_x[indices], data_t[indices], data_y[indices]


class Generator(nn.Module):
    def __init__(self, input_dim, h_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim + 2, h_dim)  # X + T + Y
        self.fc2 = nn.Linear(h_dim, h_dim)

        # Branch for t=0
        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)

        # Branch for t=1
        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, t, y):
        inputs = torch.cat([x, t, y], dim=1)
        h1 = F.relu(self.fc1(inputs))
        h2 = F.relu(self.fc2(h1))

        # t=0 branch
        h31 = F.relu(self.fc31(h2))
        logit0 = self.fc32(h31)

        # t=1 branch
        h41 = F.relu(self.fc41(h2))
        logit1 = self.fc42(h41)

        return torch.cat([logit0, logit1], dim=1)


class Discriminator(nn.Module):
    def __init__(self, input_dim, h_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim + 2, h_dim)  # X + input0 + input1
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, t, y, hat_y):
        input0 = (1 - t) * y + t * hat_y[:, 0].unsqueeze(1)
        input1 = t * y + (1 - t) * hat_y[:, 1].unsqueeze(1)
        inputs = torch.cat([x, input0, input1], dim=1)

        h1 = F.relu(self.fc1(inputs))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)


class Inference(nn.Module):
    def __init__(self, input_dim, h_dim):
        super(Inference, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)

        # Branch for t=0
        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)

        # Branch for t=1
        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        # t=0 branch
        h31 = F.relu(self.fc31(h2))
        logit0 = self.fc32(h31)

        # t=1 branch
        h41 = F.relu(self.fc41(h2))
        logit1 = self.fc42(h41)

        return torch.cat([logit0, logit1], dim=1)


def ganitelin(train_x, train_t, train_y, test_x, parameters):
    # Parameters
    h_dim = parameters["h_dim"]
    batch_size = parameters["batch_size"]
    iterations = parameters["iteration"]
    alpha = parameters["alpha"]

    no, dim = train_x.shape

    # Convert data to tensors
    train_x = torch.FloatTensor(train_x)
    train_t = torch.FloatTensor(train_t)
    train_y = torch.FloatTensor(train_y)
    test_x = torch.FloatTensor(test_x)

    # Initialize models
    G = Generator(dim, h_dim)
    D = Discriminator(dim, h_dim)
    I = Inference(dim, h_dim)

    # Optimizers
    G_optim = torch.optim.Adam(G.parameters())
    D_optim = torch.optim.Adam(D.parameters())
    I_optim = torch.optim.Adam(I.parameters())

    print("Start training Generator and Discriminator")
    for it in range(iterations):
        # Train Discriminator twice
        for _ in range(2):
            X_mb, T_mb, Y_mb = batch_generator(
                train_x.numpy(), train_t.numpy(), train_y.numpy(), batch_size
            )
            X_mb = torch.FloatTensor(X_mb)
            T_mb = torch.FloatTensor(T_mb).unsqueeze(1)
            Y_mb = torch.FloatTensor(Y_mb).unsqueeze(1)

            D.zero_grad()
            Y_tilde = G(X_mb, T_mb, Y_mb)
            D_logit = D(X_mb, T_mb, Y_mb, Y_tilde)
            D_loss = F.binary_cross_entropy_with_logits(D_logit, T_mb)
            D_loss.backward()
            D_optim.step()

        # Train Generator
        X_mb, T_mb, Y_mb = batch_generator(
            train_x.numpy(), train_t.numpy(), train_y.numpy(), batch_size
        )
        X_mb = torch.FloatTensor(X_mb)
        T_mb = torch.FloatTensor(T_mb).unsqueeze(1)
        Y_mb = torch.FloatTensor(Y_mb).unsqueeze(1)

        G.zero_grad()
        Y_tilde = G(X_mb, T_mb, Y_mb)
        D_logit = D(X_mb, T_mb, Y_mb, Y_tilde)

        G_loss_GAN = -D_loss  # Negative of D's loss
        factual_pred = T_mb * Y_tilde[:, 1].unsqueeze(1) + (1 - T_mb) * Y_tilde[:, 0].unsqueeze(1)
        G_loss_Factual = F.mse_loss(factual_pred, Y_mb)
        G_loss = G_loss_Factual + alpha * G_loss_GAN

        G_loss.backward()
        G_optim.step()

        if it % 1000 == 0:
            print(
                f"Iter: {it}/{iterations}, D_loss: {D_loss.item():.4f}, G_loss: {G_loss.item():.4f}"
            )

    print("Start training Inference network")
    for it in range(iterations):
        X_mb, T_mb, Y_mb = batch_generator(
            train_x.numpy(), train_t.numpy(), train_y.numpy(), batch_size
        )
        X_mb = torch.FloatTensor(X_mb)
        T_mb = torch.FloatTensor(T_mb).unsqueeze(1)
        Y_mb = torch.FloatTensor(Y_mb).unsqueeze(1)

        I.zero_grad()
        with torch.no_grad():
            Y_tilde = G(X_mb, T_mb, Y_mb)

        Y_hat = I(X_mb)
        target1 = T_mb * Y_mb + (1 - T_mb) * Y_tilde[:, 1].unsqueeze(1)
        target2 = (1 - T_mb) * Y_mb + T_mb * Y_tilde[:, 0].unsqueeze(1)

        I_loss1 = F.mse_loss(Y_hat[:, 1].unsqueeze(1), target1)
        I_loss2 = F.mse_loss(Y_hat[:, 0].unsqueeze(1), target2)
        I_loss = I_loss1 + I_loss2

        I_loss.backward()
        I_optim.step()

        if it % 1000 == 0:
            print(f"Iter: {it}/{iterations}, I_loss: {I_loss.item():.4f}")

    # Predict on test data
    with torch.no_grad():
        test_y_hat = I(test_x).numpy()

    return test_y_hat
