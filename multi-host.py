import multiprocessing
import sys
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
import numpy as np

def worker_process(process_id, num_processes):
    jax.distributed.initialize(
        coordinator_address="localhost:1234",
        num_processes=num_processes,
        process_id=process_id,
    )

    devices = np.array(jax.devices())
    mesh = Mesh(devices.reshape(2, 2), ('x', 'y'))

    a = jnp.arange(2 * 2.).reshape(2, 2)
    b = jnp.arange(2 * 2.).reshape(2, 2) * -1

    a = jax.device_put(a, NamedSharding(mesh, P('x', 'y')))
    b = jax.device_put(b, NamedSharding(mesh, P('x', 'y')))

    @jax.jit
    def matmul_reference(a, b):
        c = jnp.dot(a, b)
        return c
    
    c_ref = matmul_reference(a, b)

    # Inspect local shard on each process:
    print(f"Process {process_id}: local a shard: {a.addressable_shards[0].data}, local b shard: {b.addressable_shards[0].data}, local c_ref shard: {c_ref.addressable_shards[0].data}")

    # Explicitly gather
    c_gathered = jax.lax.with_sharding_constraint(c_ref, NamedSharding(mesh, P(None, None)))
    # print(f"Process {process_id}: local c_gathered shard: {c_gathered.addressable_shards[0].data}")

    # if host then print so we only get one message
    if process_id == 0:
        c_host = np.array(c_gathered)
        print(c_host)

    jax.distributed.shutdown()

    print(f"Process {process_id}: Shutdown completed")
    sys.exit(0)

def launch_processes(num_processes):
    processes = []

    for pid in range(num_processes):
        p = multiprocessing.Process(target=worker_process, args=(pid, num_processes))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    launch_processes(4)