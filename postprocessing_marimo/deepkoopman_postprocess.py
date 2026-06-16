import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np

    from deepkoopman.trainer import DeepKoopmanTrainer
    from deepkoopman.visualization import plot_losses

    return DeepKoopmanTrainer, Path, json, mo, np, plot_losses


@app.cell
def _(mo):
    run_dir = mo.ui.text(value="results/search", label="run dir")
    dataset = mo.ui.dropdown(options=["DiscreteSpectrumExample", "Pendulum", "FluidFlowOnAttractor", "FluidFlowBox"], value="DiscreteSpectrumExample", label="dataset")
    steps = mo.ui.number(start=1, stop=50, step=1, value=5, label="steps")
    return dataset, run_dir, steps


@app.cell
def _(DeepKoopmanTrainer, Path, dataset, np, run_dir, steps):
    rd = Path(run_dir.value)
    ckpts = sorted(rd.glob("*.pt"))
    if not ckpts:
        result = "No checkpoint found"
    else:
        trainer = DeepKoopmanTrainer.load(ckpts[0])
        val = np.loadtxt(Path("data") / f"{dataset.value}_val_x.csv", delimiter=",", dtype=np.float64)
        sample = val[:1]
        recon = trainer.reconstruct(sample)
        pred = trainer.predict(sample, int(steps.value))
        result = {
            "checkpoint": str(ckpts[0]),
            "recon_shape": recon.shape,
            "pred_shape": pred.shape,
            "recon": recon,
            "pred": pred,
        }
    return rd, result


@app.cell
def _(json, plot_losses, rd):
    hist = sorted(rd.glob("*.history.json"))
    if hist:
        with open(hist[0], "r", encoding="utf-8") as f:
            history = json.load(f)
        out = rd / "marimo_losses.png"
        plot_losses(history, out)
        loss_fig = str(out)
    else:
        loss_fig = "No history"
    return (loss_fig,)


@app.cell
def _(loss_fig, mo, result):
    mo.md(f"""
    loss figure: `{loss_fig}`\\n\\nresult: `{result}`
    """)
    return


if __name__ == "__main__":
    app.run()
