# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import torch
import torch.nn as nn
from torch.distributions import Normal


class MLP_Encoder(nn.Module):
    def __init__(
        self,
        num_input_dim,
        num_output_dim,
        hidden_dims=[256, 128],
        activation="elu",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(MLP_Encoder, self).__init__()

        self.num_input_dim = num_input_dim
        self.num_output_dim = num_output_dim

        activation = get_activation(activation)

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(num_input_dim, hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                encoder_layers.append(nn.Linear(hidden_dims[l], num_output_dim))
            else:
                encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        print(f"Encoder MLP: {self.encoder}")

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def forward(self, input):
        return self.encoder(input)

    def encode(self, input):
        self.encoder_out = self.encoder(input)
        return self.encoder_out.detach()

    def get_encoder_out(self):
        return self.encoder_out

    def inference(self, input):
        with torch.no_grad():
            return self.encoder(input)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
