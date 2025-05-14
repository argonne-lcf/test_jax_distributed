from mpi4py import MPI

rank = int(MPI.COMM_WORLD.Get_rank())
world_size = int(MPI.COMM_WORLD.Get_size())
COMM_WORLD = MPI.COMM_WORLD

import jax
import jaxlib

import socket
master_addr = socket.gethostname()
sock = 2345

new_comm = COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)

jax.distributed.initialize(coordinator_address=f'{master_addr}:{sock}', num_processes=12, process_id=rank, local_device_ids=int(new_comm.Get_rank()))#cluster_detection_method="mpi4py")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from functools import partial

from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils


jax.config.update('jax_threefry_partitionable', True)


n_devices = jax.device_count()
devices = mesh_utils.create_device_mesh((n_devices,))
mesh = Mesh(devices, axis_names=('i',))

a = jnp.arange(8 * 24.).reshape(8, 24)
b = jnp.arange(24 * 4.).reshape(24, 4)


@partial(jax.experimental.shard_map.shard_map, mesh=mesh, in_specs=(P(None, 'i'), P('i', None)),
         out_specs=P(None))
def matmul_basic(a_block, b_block):
    # a_block: f32[2, 8]
    # b_block: f32[8, 4]
    c_partialsum = jnp.dot(a_block, b_block)
    c_block = jax.lax.psum(c_partialsum, 'i')
    # c_block: f32[2, 4]
    return c_block


c = matmul_basic(a, b)   # c: f32[8, 4]
print(c)
