from utils import USE_CPU_FALLBACK, visualize_with_values

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

platform = "cpu" if USE_CPU_FALLBACK else "gpu"
devices = np.array(jax.devices(platform)[:2])
if len(devices) < 2:
    raise RuntimeError(f"Expected 2 {platform} devices, found {len(devices)}")

print(f"Using platform: {platform}")
print("Available devices:", devices)

# 1-D mesh with a single named axis 'x' across the 2 devices.
mesh = Mesh(devices, ("x",))
print("Created Mesh:", mesh)

sharding = NamedSharding(mesh, P("x"))
print("NamedSharding:", sharding)

# Shard a small array along axis 'x' so each device gets half the rows.
x = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)
visualize_with_values(x, title="x values before being placed on devices")

x_sharded = jax.device_put(x, sharding)
print("x_sharded.shape:", x_sharded.shape)
visualize_with_values(x_sharded, title="x_sharded values per device")
