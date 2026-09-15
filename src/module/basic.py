
import torch
import torch.nn as nn


class ShallowMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_layers: int = 2,
        hidden_channels: int = 32,
        activation: nn.Module = nn.ReLU(),
        output_activation: nn.Module = nn.ReLU(),
    ):
        super(ShallowMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_layers = hidden_layers
        self.hidden_channels = hidden_channels
        self.activation = activation
        self.output_activation = output_activation
        
        self.layers = nn.ModuleList()
        if self.hidden_layers == -1:
            pass  # Apply activation only
        elif self.hidden_layers == 0:
            self.layers.append(nn.Linear(self.in_channels, self.out_channels))
        else:
            self.layers.append(nn.Linear(self.in_channels, self.hidden_channels))
            for _ in range(self.hidden_layers - 1):
                self.layers.append(nn.Linear(self.hidden_channels, self.hidden_channels))
            self.layers.append(nn.Linear(self.hidden_channels, self.out_channels))
        
    def forward(self, x):
        if self.hidden_layers == -1:
            return self.output_activation(x)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return self.output_activation(x)

