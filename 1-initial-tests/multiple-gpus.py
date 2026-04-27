import os
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding

if len(jax.devices('gpu')) < 8:
    raise RuntimeError("At least 8 GPUs are required, but fewer were found.")

# Get available gpu devices
devices = np.array(jax.devices('gpu')[:8])
print("Available devices:", devices)

# Create a mesh with named axes 'x' and 'y'
mesh = Mesh(devices.reshape((2, 4)), ('x', 'y'))
print("Created Mesh:", mesh)

# Create two large arrays sharded across mesh
x = jax.random.normal(jax.random.PRNGKey(0), (64*1024, 64*1024), dtype=np.float32)
y = jax.random.normal(jax.random.PRNGKey(1), (64*1024, 64*1024), dtype=np.float32)

x = jax.device_put(x, NamedSharding(mesh, P('x', 'y')))
y = jax.device_put(y, NamedSharding(mesh, P('x', 'y')))

@jax.jit
def simple_op(x, y):
    return x @ y

result = simple_op(x, y)

result_gathered = jax.lax.with_sharding_constraint(
    result, NamedSharding(mesh, P(None, None))
)

print("result_gathered.shape:", result_gathered.shape)