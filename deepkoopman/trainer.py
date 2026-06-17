from __future__ import annotations

from pathlib import Path

from .lightning import DeepKoopmanLightningModule


class DeepKoopmanTrainer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "DeepKoopmanTrainer has been retired. Use DeepKoopmanLightningModule, "
            "DeepKoopmanDataModule, and deepkoopman.lightning.build_trainer instead."
        )

    @classmethod
    def load(cls, path: str | Path) -> DeepKoopmanLightningModule:
        return DeepKoopmanLightningModule.load_checkpoint(path)
