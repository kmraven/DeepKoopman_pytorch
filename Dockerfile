FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    XDG_CACHE_HOME=/workspace/.cache

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY deepkoopman ./deepkoopman
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY postprocessing ./postprocessing
COPY postprocessing_marimo ./postprocessing_marimo

RUN mkdir -p /workspace/.cache/matplotlib \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        "numpy>=1.24" \
        "pyyaml>=6.0" \
        "matplotlib>=3.8" \
        "marimo>=0.9.10" \
        "pytest>=8.0" \
    && python -m pip install --no-deps -e .

CMD ["python", "-m", "pytest", "-q"]
