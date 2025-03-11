import os
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding

if len(jax.devices('gpu')) < 1:
    raise RuntimeError("At least 1 GPU is required, but none were found.")

# Get available gpu devices
devices = np.array(jax.devices('gpu')[:1])
print("Available devices:", devices)

# Create a mesh with named axis 'x'
mesh = Mesh(devices.reshape((1, )), ('x'))
print("Created Mesh:", mesh)

# Create two large arrays sharded across mesh
x = jax.random.normal(jax.random.PRNGKey(0), (64*1024, 64*1024), dtype=np.float32)
y = jax.random.normal(jax.random.PRNGKey(1), (64*1024, 64*1024), dtype=np.float32)

x = jax.device_put(x, NamedSharding(mesh, P('x')))
y = jax.device_put(y, NamedSharding(mesh, P('x')))

@jax.jit
def simple_op(x, y):
    return x @ y

result = simple_op(x, y)

result_gathered = jax.lax.with_sharding_constraint(
    result, NamedSharding(mesh, P(None, None))
)

print("result_gathered.shape:", result_gathered.shape)