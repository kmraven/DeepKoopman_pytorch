from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import DeepKoopmanConfig


def _act(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    return nn.ReLU()


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


class MLP(nn.Module):
    def __init__(self, widths: list[int], act_type: str = "relu", final_linear: bool = True, dtype: torch.dtype = torch.float64):
        super().__init__()
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1], dtype=dtype))
            if i < len(widths) - 2 or not final_linear:
                layers.append(_act(act_type))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepKoopmanModule(nn.Module):
    def __init__(self, config: DeepKoopmanConfig, act_type: str | None = None):
        super().__init__()
        self.config = config
        model_cfg = config.model
        act_type = act_type or model_cfg.activation
        dtype = _torch_dtype(config.runtime.dtype)
        depth = (len(model_cfg.widths) - 4) // 2
        self.encoder = MLP(model_cfg.widths[: depth + 2], act_type=act_type, dtype=dtype)
        self.decoder = MLP(model_cfg.widths[depth + 2 :], act_type=act_type, dtype=dtype)

        self.omega_complex = nn.ModuleList(
            [MLP([1] + model_cfg.omega_hidden_widths + [2], act_type=act_type, dtype=dtype) for _ in range(model_cfg.num_complex_pairs)]
        )
        self.omega_real = nn.ModuleList(
            [MLP([1] + model_cfg.omega_hidden_widths + [1], act_type=act_type, dtype=dtype) for _ in range(model_cfg.num_real)]
        )
        torch.manual_seed(config.runtime.seed)
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
        init = self.config.model.initialization
        for module in list(self.encoder.modules()) + list(self.decoder.modules()):
            if isinstance(module, nn.Linear):
                self._init_linear(module, init.distribution, init.scale)
        for module in list(self.omega_complex.modules()) + list(self.omega_real.modules()):
            if isinstance(module, nn.Linear):
                self._init_linear(module, init.omega_distribution, init.omega_scale)

    def _omega_net_apply(self, y: torch.Tensor) -> list[torch.Tensor]:
        omegas: list[torch.Tensor] = []
        for j, net in enumerate(self.omega_complex):
            ind = 2 * j
            pair = y[:, ind : ind + 2]
            radius = (pair**2).sum(dim=1, keepdim=True)
            omegas.append(net(radius))
        for j, net in enumerate(self.omega_real):
            ind = 2 * self.config.model.num_complex_pairs + j
            omegas.append(net(y[:, ind : ind + 1]))
        return omegas

    def varying_multiply(self, y: torch.Tensor, omegas: list[torch.Tensor]) -> torch.Tensor:
        dt = self.config.data.delta_t
        parts = []
        for j in range(self.config.model.num_complex_pairs):
            ind = 2 * j
            pair = y[:, ind : ind + 2]
            om = omegas[j]
            scale = torch.exp(om[:, 1] * dt)
            c = torch.cos(om[:, 0] * dt)
            s = torch.sin(om[:, 0] * dt)
            x0 = scale * (c * pair[:, 0] - s * pair[:, 1])
            x1 = scale * (s * pair[:, 0] + c * pair[:, 1])
            parts.append(torch.stack([x0, x1], dim=1))

        for j in range(self.config.model.num_real):
            ind = 2 * self.config.model.num_complex_pairs + j
            lam = torch.exp(omegas[self.config.model.num_complex_pairs + j][:, 0] * dt)
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


class RatConvEncoder(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1, dtype=dtype),
            nn.GELU(),
            nn.LayerNorm((32, 8, 8), dtype=dtype),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, dtype=dtype),
            nn.GELU(),
            nn.LayerNorm((32, 8, 8), dtype=dtype),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, dtype=dtype),
            nn.GELU(),
            nn.LayerNorm((64, 8, 8), dtype=dtype),
            nn.Conv2d(64, 16, kernel_size=1, padding=0, dtype=dtype),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 16, 64, dtype=dtype),
            nn.GELU(),
            nn.Linear(64, 3, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            if x.shape[1] != 8 * 8 * 6:
                raise ValueError(f"Rat encoder expects flattened 8*8*6 features, got {tuple(x.shape)}")
            x = x.reshape(x.shape[0], 8, 8, 6).permute(0, 3, 1, 2)
        elif x.ndim == 4:
            if x.shape[1:] == (8, 8, 6):
                x = x.permute(0, 3, 1, 2)
            elif x.shape[1:] != (6, 8, 8):
                raise ValueError(f"Rat encoder expects NHWC (8,8,6) or NCHW (6,8,8), got {tuple(x.shape)}")
        else:
            raise ValueError(f"Rat encoder expects 2-D or 4-D input, got {tuple(x.shape)}")
        return self.net(x)


class RatConvDecoder(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64, dtype=dtype),
            nn.GELU(),
            nn.Linear(64, 8 * 8 * 16, dtype=dtype),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1, dtype=dtype),
            nn.GELU(),
            nn.LayerNorm((64, 8, 8), dtype=dtype),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, dtype=dtype),
            nn.GELU(),
            nn.LayerNorm((32, 8, 8), dtype=dtype),
            nn.Conv2d(32, 6, kernel_size=3, padding=1, dtype=dtype),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).reshape(z.shape[0], 16, 8, 8)
        x = self.conv(x)
        return x.permute(0, 2, 3, 1).reshape(z.shape[0], 8 * 8 * 6)


class RatEigenvalueNetwork(nn.Module):
    def __init__(self, condition_dim: int, dtype: torch.dtype):
        super().__init__()
        self.condition_dim = condition_dim
        self.net = nn.Sequential(
            nn.Linear(3 + condition_dim, 64, dtype=dtype),
            nn.GELU(),
            nn.LayerNorm(64, dtype=dtype),
            nn.Linear(64, 64, dtype=dtype),
            nn.GELU(),
            nn.LayerNorm(64, dtype=dtype),
            nn.Linear(64, 3, dtype=dtype),
        )

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim == 1 or (condition.ndim == 2 and condition.shape[-1] == 1):
            condition = condition.reshape(-1).long()
            condition = F.one_hot(condition, num_classes=self.condition_dim).to(dtype=z.dtype, device=z.device)
        else:
            condition = condition.to(dtype=z.dtype, device=z.device)
        if condition.shape[0] != z.shape[0]:
            raise ValueError(f"Condition batch {condition.shape[0]} does not match latent batch {z.shape[0]}")
        raw = self.net(torch.cat([z, condition], dim=1))
        lambda_r = 1.0 + 0.2 * torch.tanh(raw[:, 0])
        rho = torch.exp(0.1 * torch.tanh(raw[:, 1]))
        theta = torch.pi * torch.tanh(raw[:, 2])
        return torch.stack([lambda_r, rho, theta], dim=1)


class RatConditionalKoopmanModule(nn.Module):
    is_conditioned = True

    def __init__(self, config: DeepKoopmanConfig):
        super().__init__()
        self.config = config
        dtype = _torch_dtype(config.runtime.dtype)
        self.condition_dim = config.model.condition_dim
        self.encoder = RatConvEncoder(dtype)
        self.decoder = RatConvDecoder(dtype)
        self.eigenvalue_network = RatEigenvalueNetwork(self.condition_dim, dtype)
        torch.manual_seed(config.runtime.seed)
        self.reset_parameters()

    def _init_weight(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_parameters(self) -> None:
        self.apply(self._init_weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def eigen_params(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is None:
            condition = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        return self.eigenvalue_network(z, condition)

    def step_latent(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        params = self.eigen_params(z, condition)
        lambda_r = params[:, 0]
        rho = params[:, 1]
        theta = params[:, 2]
        z1 = lambda_r * z[:, 0]
        c = torch.cos(theta)
        s = torch.sin(theta)
        z2 = rho * (c * z[:, 1] - s * z[:, 2])
        z3 = rho * (s * z[:, 1] + c * z[:, 2])
        return torch.stack([z1, z2, z3], dim=1)

    def predict_latent(
        self,
        g0: torch.Tensor,
        steps: int,
        conditions: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        out = [g0]
        g = g0
        for h in range(steps):
            cond = None
            if conditions is not None:
                cond = conditions[h] if conditions.ndim >= 2 else conditions
            g = self.step_latent(g, cond)
            out.append(g)
        return out

    def predict(self, x0: torch.Tensor, steps: int, conditions: torch.Tensor | None = None) -> torch.Tensor:
        g0 = self.encode(x0)
        latents = self.predict_latent(g0, steps, conditions)
        decoded = [self.decoder(g) for g in latents]
        return torch.stack(decoded, dim=0)

    def forward(
        self,
        stacked: torch.Tensor,
        shifts: list[int],
        shifts_middle: list[int],
        conditions: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        g0 = self.encode(stacked[0])
        max_shift = max([0] + shifts)
        pred = self.predict(stacked[0], max_shift, conditions)
        y = [pred[0]] + [pred[s] for s in shifts]
        g_list = [g0] + [self.encode(stacked[s]) for s in shifts_middle]
        return y, g_list


def build_model(config: DeepKoopmanConfig) -> nn.Module:
    if config.model.architecture == "rat_conditional_conv":
        return RatConditionalKoopmanModule(config)
    if config.model.architecture != "mlp":
        raise ValueError(f"Unsupported model architecture: {config.model.architecture}")
    return DeepKoopmanModule(config)
