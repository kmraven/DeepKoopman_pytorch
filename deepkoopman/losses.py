from __future__ import annotations

import torch

from .config import DeepKoopmanConfig
from .model import DeepKoopmanModule


def _relative_den(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return torch.tensor(1.0, dtype=x.dtype, device=x.device)
    return (x**2).mean() + 1e-5


def compute_losses(model: DeepKoopmanModule, stacked: torch.Tensor, config: DeepKoopmanConfig) -> dict[str, torch.Tensor]:
    shifts = config.data.shifts
    middle_shifts = config.data.middle_shifts
    loss_cfg = config.loss
    y, g_list = model(stacked, shifts, middle_shifts)

    loss1 = loss_cfg.reconstruction_weight * ((y[0] - stacked[0]) ** 2).mean() / _relative_den(stacked[0], loss_cfg.relative)

    loss2 = torch.tensor(0.0, dtype=stacked.dtype, device=stacked.device)
    if shifts:
        for j, s in enumerate(shifts):
            l = ((y[j + 1] - stacked[s]) ** 2).mean() / _relative_den(stacked[s], loss_cfg.relative)
            loss2 = loss2 + loss_cfg.reconstruction_weight * l
        loss2 = loss2 / len(shifts)

    loss3 = torch.tensor(0.0, dtype=stacked.dtype, device=stacked.device)
    if middle_shifts:
        g = g_list[0]
        omegas = model._omega_net_apply(g)
        next_step = model.varying_multiply(g, omegas)
        count = 0
        for j in range(max(middle_shifts)):
            if (j + 1) in middle_shifts:
                target = g_list[count + 1]
                l = ((next_step - target) ** 2).mean() / _relative_den(target, loss_cfg.relative)
                loss3 = loss3 + loss_cfg.middle_shift_weight * l
                count += 1
            omegas = model._omega_net_apply(next_step)
            next_step = model.varying_multiply(next_step, omegas)
        loss3 = loss3 / len(middle_shifts)

    linf1 = (y[0] - stacked[0]).abs().max() / (_relative_den(stacked[0].abs().max(), loss_cfg.relative))
    linf2 = (y[1] - stacked[1]).abs().max() / (_relative_den(stacked[1].abs().max(), loss_cfg.relative)) if len(y) > 1 else 0.0
    loss_linf = loss_cfg.linf_weight * (linf1 + linf2)

    l1 = sum(p.abs().sum() for p in model.parameters()) * loss_cfg.l1_weight
    l2 = sum((p**2).sum() for n, p in model.named_parameters() if "bias" not in n) * loss_cfg.l2_weight

    total = loss1 + loss2 + loss3 + loss_linf + l1 + l2
    return {
        "loss": total,
        "loss1": loss1,
        "loss2": loss2,
        "loss3": loss3,
        "loss_linf": loss_linf,
        "loss_l1": l1,
        "loss_l2": l2,
    }
