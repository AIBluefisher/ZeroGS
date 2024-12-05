from typing import Callable, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled)
            )
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(
                in_features, self.output_dim, bias=bias_enabled
            )
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):
        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=0,  # no hidden layers
            **kwargs,
        )


class NormalizedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = None,
        net_depth: int = 8,
        net_width: int = 256,
        skip_layer: List = [4],
        hidden_activation: Callable = nn.ReLU(),
        bias: float = 0.5,
        weight_norm: bool = True,
        geometric_init: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_activation = hidden_activation
        self.bias = bias
        dims = [input_dim] + [net_width for _ in range(net_depth)] + [output_dim]
        self.num_layers = len(dims)

        for i in range(0, self.num_layers - 1):
            if (self.skip_layer is not None) and (i + 1) in self.skip_layer:
                out_dim = dims[i + 1] - dims[0]
            else:
                out_dim = dims[i + 1]

            lin = nn.Linear(dims[i], out_dim)

            if geometric_init:
                if i == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight,
                        mean=np.sqrt(np.pi) / np.sqrt(dims[i]),
                        std=0.0001
                    )
                    torch.nn.init.constant_(lin.bias, -bias)
                elif i == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :3],
                        0.0,
                        np.sqrt(2) / np.sqrt(out_dim)
                    )
                elif (self.skip_layer is not None) and (i in self.skip_layer):
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight,
                        0.0,
                        np.sqrt(2) / np.sqrt(out_dim)
                    )
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight,
                        0.0,
                        np.sqrt(2) / np.sqrt(out_dim)
                    )

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(i), lin)

    def forward(self, x):
        inputs = x
        for i in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(i))

            if (self.skip_layer is not None) and (i in self.skip_layer):
                x = torch.cat([x, inputs], dim=-1) / np.sqrt(2)
            
            x = lin(x)

            if i < self.num_layers - 2:
                x = self.hidden_activation(x)
        
        return x
