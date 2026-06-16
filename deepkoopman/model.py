from __future__ import annotations

import torch
from torch import nn

from .config import DeepKoopmanConfig


def _act(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    return nn.ReLU()


class MLP(nn.Module):
    def __init__(self, widths: list[int], act_type: str = "relu", final_linear: bool = True):
        super().__init__()
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1], dtype=torch.float64))
            if i < len(widths) - 2 or not final_linear:
                layers.append(_act(act_type))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepKoopmanModule(nn.Module):
    def __init__(self, config: DeepKoopmanConfig, act_type: str | None = None):
        super().__init__()
        self.config = config
        act_type = act_type or config.act_type
        depth = (len(config.widths) - 4) // 2
        self.encoder = MLP(config.widths[: depth + 2], act_type=act_type)
        self.decoder = MLP(config.widths[depth + 2 :], act_type=act_type)

        self.omega_complex = nn.ModuleList(
            [MLP([1] + config.hidden_widths_omega + [2], act_type=act_type) for _ in range(config.num_complex_pairs)]
        )
        self.omega_real = nn.ModuleList(
            [MLP([1] + config.hidden_widths_omega + [1], act_type=act_type) for _ in range(config.num_real)]
        )
        torch.manual_seed(config.seed)
        self.reset_parameters()

    def _init_linear(self, layer: nn.Linear, distribution: str, scale: float) -> None:
        if distribution == "tn":
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=scale, a=-2 * scale, b=2 * scale)
        elif distribution == "dl":
            bound = 1.0 / (layer.in_features ** 0.5)
            nn.init.uniform_(layer.weight, -bound, bound)
        elif distribution == "xavier":
            bound = 4.0 * (6.0 / (layer.in_features + layer.out_features)) ** 0.5
            nn.init.uniform_(layer.weight, -bound, bound)
        elif distribution == "glorot_bengio":
            bound = (6.0 / (layer.in_features + layer.out_features)) ** 0.5
            nn.init.uniform_(layer.weight, -bound, bound)
        elif distribution == "he":
            nn.init.normal_(layer.weight, mean=0.0, std=(2.0 / layer.in_features) ** 0.5)
        else:
            raise ValueError(f"Unsupported initialization distribution: {distribution}")
        nn.init.zeros_(layer.bias)

    def reset_parameters(self) -> None:
        for module in list(self.encoder.modules()) + list(self.decoder.modules()):
            if isinstance(module, nn.Linear):
                self._init_linear(module, self.config.init_distribution, self.config.init_scale)
        for module in list(self.omega_complex.modules()) + list(self.omega_real.modules()):
            if isinstance(module, nn.Linear):
                self._init_linear(module, self.config.omega_init_distribution, self.config.omega_init_scale)

    def _omega_net_apply(self, y: torch.Tensor) -> list[torch.Tensor]:
        omegas: list[torch.Tensor] = []
        for j, net in enumerate(self.omega_complex):
            ind = 2 * j
            pair = y[:, ind : ind + 2]
            radius = (pair**2).sum(dim=1, keepdim=True)
            omegas.append(net(radius))
        for j, net in enumerate(self.omega_real):
            ind = 2 * self.config.num_complex_pairs + j
            omegas.append(net(y[:, ind : ind + 1]))
        return omegas

    def varying_multiply(self, y: torch.Tensor, omegas: list[torch.Tensor]) -> torch.Tensor:
        dt = self.config.delta_t
        parts = []
        for j in range(self.config.num_complex_pairs):
            ind = 2 * j
            pair = y[:, ind : ind + 2]
            om = omegas[j]
            scale = torch.exp(om[:, 1] * dt)
            c = torch.cos(om[:, 0] * dt)
            s = torch.sin(om[:, 0] * dt)
            x0 = scale * (c * pair[:, 0] - s * pair[:, 1])
            x1 = scale * (s * pair[:, 0] + c * pair[:, 1])
            parts.append(torch.stack([x0, x1], dim=1))

        for j in range(self.config.num_real):
            ind = 2 * self.config.num_complex_pairs + j
            lam = torch.exp(omegas[self.config.num_complex_pairs + j][:, 0] * dt)
            parts.append((y[:, ind] * lam).unsqueeze(1))

        return torch.cat(parts, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def predict_latent(self, g0: torch.Tensor, steps: int) -> list[torch.Tensor]:
        out = [g0]
        g = g0
        for _ in range(steps):
            om = self._omega_net_apply(g)
            g = self.varying_multiply(g, om)
            out.append(g)
        return out

    def predict(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        g0 = self.encode(x0)
        latents = self.predict_latent(g0, steps)
        decoded = [self.decoder(g) for g in latents]
        return torch.stack(decoded, dim=0)

    def forward(self, stacked: torch.Tensor, shifts: list[int], shifts_middle: list[int]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        g0 = self.encode(stacked[0])
        max_shift = max([0] + shifts)
        pred = self.predict(stacked[0], max_shift)
        y = [pred[0]] + [pred[s] for s in shifts]

        g_list = [g0]
        for s in shifts_middle:
            g_list.append(self.encode(stacked[s]))
        return y, g_list
