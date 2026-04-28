"""Demo: the working counterpart to matrix_multiply_fail.

Same shapes and values as the failing version — A (4, 2) @ B (2, 4) = C (4, 4) —
but this time we *replicate* B across both devices instead of sharding it.
Each device then has:
  • a unique 2-row slice of A
  • the full B
and computes its 2 rows of C independently. No collectives needed."""

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

A = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
B = jnp.arange(8, dtype=jnp.float32).reshape(2, 4) + 10

visualize_with_values(A, title="A (4x2) — before placement")
visualize_with_values(B, title="B (2x4) — before placement")

# A: rows split across devices. B: fully replicated (no mesh axis in spec).
A_sharded = jax.device_put(A, NamedSharding(mesh, P("x", None)))
B_replicated = jax.device_put(B, NamedSharding(mesh, P(None, None)))

visualize_with_values(A_sharded, title="A — sharded on rows: P('x', None)")
visualize_with_values(B_replicated, title="B — replicated: P(None, None)")


# Local shapes inside shard_map: A (2,2) @ B (2,4) -> (2,4). out_specs=P('x', None)
# stitches the two (2,4) blocks along axis 0 to give the global (4,4) result.
@partial(shard_map, mesh=mesh, in_specs=(P("x", None), P(None, None)), out_specs=P("x", None))
def matmul(a, b):
    return a @ b


C = matmul(A_sharded, B_replicated)
visualize_with_values(C, title="C = A @ B — sharded on rows: P('x', None)")

console = Console()
expected = np.asarray(A) @ np.asarray(B)
matches = np.allclose(np.asarray(C), expected)
console.print(
    Panel(
        f"[bold green]matches host-computed reference: {matches}[/]",
        title="[bold green]matmul succeeded[/]",
        title_align="left",
        border_style="green",
        expand=False,
    )
)
