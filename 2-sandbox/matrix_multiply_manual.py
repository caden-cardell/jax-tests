"""Demo: a manually-written matmul, jit-compiled via shard_map (which opts out of
GSPMD's automatic collective insertion). Same mesh for everything — the failure
mode here isn't 'incompatible devices'; it's 'each device doesn't have the data
it needs' — because shard_map runs the function locally per device.

`manual_matmul` decomposes A @ B into a per-output-cell scalar dot product:
  c[i, j] = jnp.dot(a[i, :], b[:, j])
That dot requires a's row and b's column to have the *same length*. Inside
shard_map:
  • If both A and B are sharded P('x', None), each device sees A_local (M/2, K)
    and B_local (K/2, N) — row length K vs col length K/2. Mismatch → raises.
  • If B is replicated (P(None, None)), each device has full B, so the lengths
    match and the local computation works."""

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


def manual_matmul(a, b):
    """A @ B implemented as nested vmaps of vector dot products.
    No reductions across devices — purely local per shard."""
    return jax.vmap(lambda row: jax.vmap(lambda col: jnp.dot(row, col))(b.T))(a)


A_sharded = jax.device_put(A, NamedSharding(mesh, P("x", None)))
B_sharded = jax.device_put(B, NamedSharding(mesh, P("x", None)))
B_replicated = jax.device_put(B, NamedSharding(mesh, P(None, None)))

visualize_with_values(A_sharded, title="A (4x2) — sharded on rows: P('x', None)")
visualize_with_values(B_sharded, title="B (2x4) — sharded on rows: P('x', None)  [contraction axis split]")
visualize_with_values(B_replicated, title="B (2x4) — replicated: P(None, None)")

console = Console()


# --- Case 1: both A and B sharded on rows -----------------------------------

try:
    result = manual_matmul(A_sharded, B_sharded)
    visualize_with_values(result, title="result (should never print)")
except Exception as e:
    console.print(
        Panel(
            f"[bold red]{type(e).__name__}[/]: {e}",
            title="[bold yellow]both sharded on rows — fails[/]",
            title_align="left",
            border_style="red",
            expand=False,
        )
    )


# --- Case 2: A sharded on rows, B replicated --------------------------------

C = manual_matmul(A_sharded, B_replicated)
visualize_with_values(C, title="C = manual_matmul(A, B) — B replicated")

expected = np.asarray(A) @ np.asarray(B)
console.print(
    Panel(
        f"[bold green]matches host-computed reference: {np.allclose(np.asarray(C), expected)}[/]",
        title="[bold green]B replicated — succeeds[/]",
        title_align="left",
        border_style="green",
        expand=False,
    )
)

console.print(
    "[dim]Note: same Mesh in both cases. The failure isn't a device-incompatibility "
    "issue — it's that shard_map runs the function locally on each device with no "
    "automatic collectives, so the contraction axis must be available in full on "
    "every device. Replicating B (or A) gives every device the full contraction axis; "
    "sharding the contraction axis breaks the local dot product.[/]"
)
