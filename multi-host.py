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
    mesh = Mesh(devices.reshape(num_processes), ('x',))

    a = jnp.arange(8 * 16.).reshape(8, 16)
    b = jnp.arange(16 * 4.).reshape(16, 4)

    a = jax.device_put(a, NamedSharding(mesh, P('x', None)))
    b = jax.device_put(b, NamedSharding(mesh, P('x', None)))

    @jax.jit
    def matmul_reference(a, b):
        c = jnp.dot(a, b)
        c_gathered = jax.lax.with_sharding_constraint(c, NamedSharding(mesh, P(None, None)))
        return c_gathered
    
    c_ref = matmul_reference(a, b)

    # Explicitly transfer to host/device 0 after computation:
    if process_id == 0:
        c_host = np.array(c_ref)
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