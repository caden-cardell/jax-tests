import multiprocessing
import sys
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
import numpy as np

verbose = False

def distributed_worker(process_id, total_processes):
    """
    Initializes JAX distributed environment, performs matrix multiplication
    on sharded arrays, and prints local shard data for debugging.

    Args:
        process_id (int): The unique ID of the current process.
        total_processes (int): Total number of distributed processes.
    """
    jax.distributed.initialize(
        coordinator_address="localhost:1234",
        num_processes=total_processes,
        process_id=process_id,
    )

    devices = np.array(jax.devices())
    mesh = Mesh(devices.reshape(2, 2), ('x', 'y'))

    matrix_a = jnp.array([1, 2, 3, 5], dtype=jnp.float32).reshape(2, 2)
    matrix_b = jnp.array([11, 13, 17, 19], dtype=jnp.float32).reshape(2, 2)

    matrix_a = jax.device_put(matrix_a, NamedSharding(mesh, P('x', 'y')))
    matrix_b = jax.device_put(matrix_b, NamedSharding(mesh, P('x', 'y')))

    @jax.jit
    def multiply_matrices(a, b):
        return jnp.dot(a, b)

    result_matrix = multiply_matrices(matrix_a, matrix_b)

    # Print local shard data for debugging purposes.
    print(
        f"Process {process_id}: "
        f"Local shard of matrix_a: {matrix_a.addressable_shards[0].data}, "
        f"Local shard of matrix_b: {matrix_b.addressable_shards[0].data}, "
        f"Local shard of result_matrix: {result_matrix.addressable_shards[0].data}"
    )

    # Gather result explicitly
    result_gathered = jax.lax.with_sharding_constraint(
        result_matrix, NamedSharding(mesh, P(None, None))
    )

    # Only the host process prints the fully gathered result
    if process_id == 0:
        print(
            f"Process {process_id}: "
            f"Local shard of result_gathered: \n{result_gathered.addressable_shards[0].data}"
        )
    elif verbose:
        print(
            f"Process {process_id}: "
            f"Local shard of result_gathered: \n{result_gathered.addressable_shards[0].data}"
        )

    jax.distributed.shutdown()

    print(f"Process {process_id}: Shutdown completed")
    sys.exit(0)


def start_distributed_workers():
    """
    Launches multiple distributed worker processes.
    """
    total_processes = 4
    processes = []

    for pid in range(total_processes):
        process = multiprocessing.Process(
            target=distributed_worker, args=(pid, total_processes)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    start_distributed_workers()
