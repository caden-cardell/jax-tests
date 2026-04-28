# jax-tests

A hands-on walkthrough of JAX's core array sharding primitives: `Mesh`, `PartitionSpec`, `NamedSharding`, and `jax.device_put`. Each step is visualized in the terminal so you can see exactly how data is distributed across devices.

Works with real GPUs or falls back to simulated CPU devices automatically.

## Setup

```bash
conda create -n jax-tests python=3.11
conda activate jax-tests
pip install -r requirements.txt
```

## Run

```bash
python sharding_walkthrough.py
```
