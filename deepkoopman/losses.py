from __future__ import annotations

import torch

from .config import DeepKoopmanConfig
from .model import DeepKoopmanModule


def _relative_den(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return torch.tensor(1.0, dtype=x.dtype, device=x.device)
    return (x**2).mean() + 1e-5


def compute_losses(model: DeepKoopmanModule, stacked: torch.Tensor, config: DeepKoopmanConfig) -> dict[str, torch.Tensor]:
    y, g_list = model(stacked, config.shifts, config.shifts_middle)

    loss1 = config.recon_lam * ((y[0] - stacked[0]) ** 2).mean() / _relative_den(stacked[0], config.relative_loss)

    loss2 = torch.tensor(0.0, dtype=stacked.dtype, device=stacked.device)
    if config.shifts:
        for j, s in enumerate(config.shifts):
            l = ((y[j + 1] - stacked[s]) ** 2).mean() / _relative_den(stacked[s], config.relative_loss)
            loss2 = loss2 + config.recon_lam * l
        loss2 = loss2 / len(config.shifts)

    loss3 = torch.tensor(0.0, dtype=stacked.dtype, device=stacked.device)
    if config.shifts_middle:
        g = g_list[0]
        omegas = model._omega_net_apply(g)
        next_step = model.varying_multiply(g, omegas)
        count = 0
        for j in range(max(config.shifts_middle)):
            if (j + 1) in config.shifts_middle:
                target = g_list[count + 1]
                l = ((next_step - target) ** 2).mean() / _relative_den(target, config.relative_loss)
                loss3 = loss3 + config.mid_shift_lam * l
                count += 1
            omegas = model._omega_net_apply(next_step)
            next_step = model.varying_multiply(next_step, omegas)
        loss3 = loss3 / len(config.shifts_middle)

    linf1 = (y[0] - stacked[0]).abs().max() / (_relative_den(stacked[0].abs().max(), config.relative_loss))
    linf2 = (y[1] - stacked[1]).abs().max() / (_relative_den(stacked[1].abs().max(), config.relative_loss)) if len(y) > 1 else 0.0
    loss_linf = config.Linf_lam * (linf1 + linf2)

    l1 = sum(p.abs().sum() for p in model.parameters()) * config.l1_lam
    l2 = sum((p**2).sum() for n, p in model.named_parameters() if "bias" not in n) * config.l2_lam

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
