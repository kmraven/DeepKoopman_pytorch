from __future__ import annotations

import torch

from .config import DeepKoopmanConfig
from .model import DeepKoopmanModule


def _relative_den(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return torch.tensor(1.0, dtype=x.dtype, device=x.device)
    return (x**2).mean() + 1e-5


def _regularization_losses(model: torch.nn.Module, config: DeepKoopmanConfig) -> tuple[torch.Tensor, torch.Tensor]:
    params = list(model.parameters())
    if not params:
        zero = torch.tensor(0.0)
        return zero, zero
    ref = params[0]
    l1 = sum(p.abs().sum() for p in params) * config.loss.l1_weight
    l2 = sum((p**2).sum() for n, p in model.named_parameters() if "bias" not in n) * config.loss.l2_weight
    return l1.to(dtype=ref.dtype, device=ref.device), l2.to(dtype=ref.dtype, device=ref.device)


def _covariance_loss(z: torch.Tensor, weight: float) -> torch.Tensor:
    if weight == 0.0:
        return torch.tensor(0.0, dtype=z.dtype, device=z.device)
    if z.shape[0] < 2:
        return torch.tensor(0.0, dtype=z.dtype, device=z.device)
    centered = z - z.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / (z.shape[0] - 1)
    eye = torch.eye(z.shape[1], dtype=z.dtype, device=z.device)
    return weight * ((cov - eye) ** 2).mean()


def _compute_conditioned_losses(
    model: torch.nn.Module,
    stacked: torch.Tensor,
    conditions: torch.Tensor | None,
    config: DeepKoopmanConfig,
) -> dict[str, torch.Tensor]:
    shifts = config.data.shifts
    middle_shifts = config.data.middle_shifts
    loss_cfg = config.loss
    if conditions is None:
        conditions = torch.zeros(stacked.shape[:2], dtype=torch.long, device=stacked.device)
    if conditions.ndim == 3 and conditions.shape[-1] == 1:
        conditions = conditions[..., 0]

    z0 = model.encode(stacked[0])
    x0_recon = model.decoder(z0)
    loss1 = loss_cfg.reconstruction_weight * ((x0_recon - stacked[0]) ** 2).mean() / _relative_den(
        stacked[0], loss_cfg.relative
    )

    max_shift = max([0] + shifts + middle_shifts)
    shift_set = set(shifts)
    middle_set = set(middle_shifts)
    pred_total = torch.tensor(0.0, dtype=stacked.dtype, device=stacked.device)
    lin_total = torch.tensor(0.0, dtype=stacked.dtype, device=stacked.device)
    pred_count = 0
    lin_count = 0
    z_pred = z0
    for h in range(1, max_shift + 1):
        z_pred = model.step_latent(z_pred, conditions[h - 1])
        if h in shift_set:
            x_pred = model.decoder(z_pred)
            pred_total = pred_total + ((x_pred - stacked[h]) ** 2).mean() / _relative_den(stacked[h], loss_cfg.relative)
            pred_count += 1
        if h in middle_set:
            target_z = model.encode(stacked[h])
            lin_total = lin_total + ((z_pred - target_z) ** 2).mean() / _relative_den(target_z, loss_cfg.relative)
            lin_count += 1

    loss2 = loss_cfg.prediction_weight * pred_total / max(pred_count, 1)
    loss3 = loss_cfg.middle_shift_weight * lin_total / max(lin_count, 1)
    loss_cov = _covariance_loss(z0, loss_cfg.covariance_weight)
    loss_linf = torch.tensor(0.0, dtype=stacked.dtype, device=stacked.device)
    if loss_cfg.linf_weight:
        loss_linf = loss_cfg.linf_weight * (x0_recon - stacked[0]).abs().max()
    loss_l1, loss_l2 = _regularization_losses(model, config)
    total = loss1 + loss2 + loss3 + loss_cov + loss_linf + loss_l1 + loss_l2
    return {
        "loss": total,
        "loss1": loss1,
        "loss2": loss2,
        "loss3": loss3,
        "loss_cov": loss_cov,
        "loss_linf": loss_linf,
        "loss_l1": loss_l1,
        "loss_l2": loss_l2,
    }


def compute_losses(
    model: DeepKoopmanModule,
    stacked: torch.Tensor,
    config: DeepKoopmanConfig,
    conditions: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if getattr(model, "is_conditioned", False):
        return _compute_conditioned_losses(model, stacked, conditions, config)

    shifts = config.data.shifts
    middle_shifts = config.data.middle_shifts
    loss_cfg = config.loss
    y, g_list = model(stacked, shifts, middle_shifts)

    loss1 = loss_cfg.reconstruction_weight * ((y[0] - stacked[0]) ** 2).mean() / _relative_den(stacked[0], loss_cfg.relative)

    loss2 = torch.tensor(0.0, dtype=stacked.dtype, device=stacked.device)
    if shifts:
        for j, s in enumerate(shifts):
            l = ((y[j + 1] - stacked[s]) ** 2).mean() / _relative_den(stacked[s], loss_cfg.relative)
            loss2 = loss2 + loss_cfg.prediction_weight * l
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

    loss_cov = _covariance_loss(g_list[0], loss_cfg.covariance_weight)
    l1, l2 = _regularization_losses(model, config)

    total = loss1 + loss2 + loss3 + loss_cov + loss_linf + l1 + l2
    return {
        "loss": total,
        "loss1": loss1,
        "loss2": loss2,
        "loss3": loss3,
        "loss_cov": loss_cov,
        "loss_linf": loss_linf,
        "loss_l1": l1,
        "loss_l2": l2,
    }
