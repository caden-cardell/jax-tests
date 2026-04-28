"""Demo: sharding two 2x2 matrices the wrong way for matmul.

When `A @ B` is sharded with `shard_map`, each device runs the matmul on its local
shard. If we shard B along its rows, the contraction axis is split across devices,
and each device's local shape no longer multiplies. This script intentionally
triggers the failure and prints the resulting error inside a rich panel."""

from functools import partial

from utils import USE_CPU_FALLBACK, visualize_with_values

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from rich.console import Console
from rich.panel import Panel


platform = "cpu" if USE_CPU_FALLBACK else "gpu"
devices = np.array(jax.devices(platform)[:2])
mesh = Mesh(devices, ("x",))

A = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)
B = jnp.arange(4, dtype=jnp.float32).reshape(2, 2) + 10

# Both A and B are sharded along axis 0 (rows). For A @ B, B's rows are the
# contraction axis — splitting them across devices means each device only sees
# half the data it needs to compute its dot product.
row_sharding = NamedSharding(mesh, P("x", None))
A_sharded = jax.device_put(A, row_sharding)
B_sharded = jax.device_put(B, row_sharding)

visualize_with_values(A_sharded, title="A — sharded on rows")
visualize_with_values(B_sharded, title="B — also sharded on rows (WRONG: rows are the contraction axis)")


# shard_map runs the function locally on each device with no automatic resharding.
# Local shapes are (1, 2) @ (1, 2) — the inner dims (2 vs 1) don't match.
@partial(shard_map, mesh=mesh, in_specs=(P("x", None), P("x", None)), out_specs=P("x", None))
def bad_matmul(a, b):
    return a @ b


console = Console()
try:
    result = bad_matmul(A_sharded, B_sharded)
    visualize_with_values(result, title="result (should never print)")
except Exception as e:
    console.print(
        Panel(
            f"[bold red]{type(e).__name__}[/]: {e}",
            title="[bold yellow]matmul failed as expected[/]",
            title_align="left",
            border_style="red",
            expand=False,
        )
    )
    console.print(
        "[dim]Why: each device holds A[1,2] and B[1,2]. The matmul needs A's columns (2) "
        "to align with B's rows (1 locally) — they don't.\n"
        "Fix: shard B on its columns instead — P(None, 'x') — so each device gets the full "
        "rows of B it needs, or use an all-gather / psum inside shard_map to combine "
        "partial results.[/]"
    )
