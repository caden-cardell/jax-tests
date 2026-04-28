"""Demo: even sharding A and B in *different* directions doesn't tile the output
on a 1-D mesh — at least one operand has to be replicated (or you need a 2-D mesh
plus collectives).

A is (4, 2), B is (2, 4), so C = A @ B is (4, 4).

Sharding choice:
  A: P('x', None)  →  rows split; each device holds A[2,2]
  B: P(None, 'x')  →  cols split; each device holds B[2,2]

Each device's local matmul produces a (2, 2) block — but those blocks are
*different* slices of C (top-left and bottom-right quadrants). They don't tile
the full (4, 4) output along any single axis of a 1-D mesh, and they aren't
replicas of one global value either, so shard_map's check_rep catches the
inconsistency."""

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
B = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)

A_sharded = jax.device_put(A, NamedSharding(mesh, P("x", None)))
B_sharded = jax.device_put(B, NamedSharding(mesh, P(None, "x")))

visualize_with_values(A_sharded, title="A (4x2) — sharded on rows: P('x', None)")
visualize_with_values(B_sharded, title="B (2x4) — sharded on cols: P(None, 'x')")


# Local shapes inside shard_map: A (2,2) @ B (2,2) -> (2,2). That (2,2) is a
# different block of C on each device. We declare out_specs=P(None, None) to
# claim the result is replicated — which it isn't — so check_rep raises.
@partial(shard_map, mesh=mesh, in_specs=(P("x", None), P(None, "x")), out_specs=P(None, None))
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
        "[dim]Why: device 0 computes C[:2, :2] (top-left block), device 1 computes "
        "C[2:, 2:] (bottom-right block). On a 1-D mesh you can only tile output along "
        "one axis — neither device has rows-of-C nor cols-of-C, just disjoint quadrants.\n"
        "Fixes:\n"
        "  • Replicate B (P(None, None)) so every device has the full B and computes its "
        "rows of C — out_specs=P('x', None).\n"
        "  • Replicate A (P(None, None)) and shard B on cols — out_specs=P(None, 'x').\n"
        "  • Use a 2-D mesh of shape (2, 2) and out_specs=P('x', 'y') to tile the "
        "output as quadrants (needs 4 devices).[/]"
    )
