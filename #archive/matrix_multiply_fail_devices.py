"""Demo: a jitted matmul that fails with 'Received incompatible devices' because
the two inputs are committed to incompatible meshes — so the jit dispatcher
rejects the call before GSPMD ever gets to insert collectives.

Why this happens despite jit:
  GSPMD does operate inside jit, but only within a single compatible device
  assignment. When committed inputs disagree on which devices they live on
  (here: same physical devices, but in opposite order), JAX can't pick a
  consistent device set for the compiled program and raises at dispatch time —
  before tracing, before any collective insertion."""

from utils import USE_CPU_FALLBACK, visualize_with_values

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from rich.console import Console
from rich.panel import Panel


platform = "cpu" if USE_CPU_FALLBACK else "gpu"
devices = jax.devices(platform)[:2]

# Two meshes wrapping the same physical devices in opposite orders. Each is a
# valid 1-D mesh on its own, but they're not the *same* device assignment.
mesh_forward = Mesh(np.array(devices), ("x",))
mesh_reversed = Mesh(np.array(list(reversed(devices))), ("x",))

A = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
B = jnp.arange(8, dtype=jnp.float32).reshape(2, 4) + 10

visualize_with_values(A, title="A (4x2) — before placement")
visualize_with_values(B, title="B (2x4) — before placement")

A_sharded = jax.device_put(A, NamedSharding(mesh_forward, P("x", None)))
B_sharded = jax.device_put(B, NamedSharding(mesh_forward, P("x", None)))

visualize_with_values(A_sharded, title="A — sharded on mesh_forward (devices [0, 1])")
visualize_with_values(B_sharded, title="B — sharded on mesh_reversed (devices [1, 0])")


@jax.jit
def matmul(a, b):
    return a @ b


console = Console()
try:
    result = matmul(A_sharded, B_sharded)
    visualize_with_values(result, title="result (should never print)")
except Exception as e:
    console.print(
        Panel(
            f"[bold red]{type(e).__name__}[/]: {e}",
            title="[bold yellow]jitted matmul failed as expected[/]",
            title_align="left",
            border_style="red",
            expand=False,
        )
    )
    console.print(
        "[dim]Why: A's shards live on physical devices [0, 1], B's on [1, 0]. The jit "
        "dispatcher needs one consistent device assignment for the compiled program, "
        "and these two don't agree. The rejection happens before tracing — GSPMD never "
        "runs, so no collective is inserted, and you see 'Received incompatible devices'.\n"
        "Fix: place both arrays on the *same* Mesh object (or call jax.device_put again "
        "to re-pin one onto the other's mesh) before invoking the jitted function.[/]"
    )
