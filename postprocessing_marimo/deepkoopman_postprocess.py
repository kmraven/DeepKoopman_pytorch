import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np

    from deepkoopman.lightning import DeepKoopmanLightningModule
    from deepkoopman.visualization import load_history, plot_losses

    return DeepKoopmanLightningModule, Path, load_history, mo, np, plot_losses


@app.cell
def _(mo):
    run_dir = mo.ui.text(value="results/search", label="run dir")
    dataset = mo.ui.dropdown(options=["DiscreteSpectrumExample", "Pendulum", "FluidFlowOnAttractor", "FluidFlowBox"], value="DiscreteSpectrumExample", label="dataset")
    steps = mo.ui.number(start=1, stop=50, step=1, value=5, label="steps")
    return dataset, run_dir, steps


@app.cell
def _(DeepKoopmanLightningModule, Path, dataset, np, run_dir, steps):
    rd = Path(run_dir.value)
    ckpts = sorted(rd.glob("**/*.ckpt"))
    if not ckpts:
        result = "No checkpoint found"
    else:
        module = DeepKoopmanLightningModule.load_checkpoint(ckpts[0])
        val = np.loadtxt(Path("data") / f"{dataset.value}_val_x.csv", delimiter=",", dtype=np.float64)
        sample = val[:1]
        recon = module.reconstruct_array(sample)
        pred = module.predict_array(sample, int(steps.value))
        result = {
            "checkpoint": str(ckpts[0]),
            "recon_shape": recon.shape,
            "pred_shape": pred.shape,
            "recon": recon,
            "pred": pred,
        }
    return rd, result


@app.cell
def _(load_history, plot_losses, rd):
    hist = sorted(rd.glob("**/metrics.csv"))
    if hist:
        history = load_history(hist[0])
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
