"""Demo: even sharding A and B in *different* directions doesn't work on a 1-D
mesh — at least one operand has to be replicated.

A is (4, 2), B is (2, 4), so C = A @ B is (4, 4).

Sharding choice (different directions, contraction axis K=2 fully replicated on both):
  A: P('x', None)  →  rows split; each device holds A[2, 2]
  B: P(None, 'x')  →  cols split; each device holds B[2, 2]

Each device's local matmul produces a (2, 2) block — but those blocks are
*different* slices of C (top-left and bottom-right quadrants). They neither
tile the full (4, 4) output along any single axis of a 1-D mesh, nor are they
replicas of one global value, so shard_map's check_rep catches it. The fix is
to *replicate* one of the operands (see matrix_multiply_pass.py)."""

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

A_sharded = jax.device_put(A, NamedSharding(mesh, P("x", None)))
# B_sharded = jax.device_put(B, NamedSharding(mesh, P(None, "x")))
B_sharded = jax.device_put(B, NamedSharding(mesh, P("x", None)))

visualize_with_values(A_sharded, title="A — sharded on rows: P('x', None)")
visualize_with_values(B_sharded, title="B — sharded on cols: P(None, 'x')")

console = Console()
try:
    result = A_sharded @ B_sharded
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
        "[dim]Why: sharding direction isn't the issue — A and B are sharded on "
        "different axes, and the contraction dim K=2 is fully replicated on both.\n"
        "The issue is that on a 1-D 2-device mesh, the (4, 4) output can't be tiled: "
        "device 0 computes C[:2, :2] (top-left) and device 1 computes C[2:, 2:] "
        "(bottom-right) — disjoint quadrants, not strips along any single axis.\n"
        "Fix: replicate one operand. With B replicated (P(None, None)), each device "
        "holds the full B and computes its rows of C — see matrix_multiply_pass.py. "
        "Or use a 2-D mesh of shape (2, 2) (4 devices) and out_specs=P('x', 'y').[/]"
    )
