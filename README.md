# jax-tests

## Setup
Setup for locally run tests.
```
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Local Multi-host Test
This test shows that shards can be spread across multiple hosts. This should enable expanding past a single host and its limit on a maximum number of GPUs.
```
python3 local-multi-host.py
```

## Multiple GPUs Test
When run on a `gpu_8x_a100_80gb_sxm4` GPU instance on Lambda Labs a single GPU runs out of resources.
```
python single_gpu.py
```
```
Available devices: [cuda(id=0)]
Created Mesh: Mesh('x': 1)
2025-03-11 02:52:50.155782: W external/xla/xla/service/hlo_rematerialization.cc:3005] Can't reduce memory use below 41.19GiB (44223270085 bytes) by rematerialization; only reduced to 48.00GiB (51539607572 bytes), down from 48.00GiB (51539607572 bytes) originally
2025-03-11 02:52:50.483350: W external/xla/xla/tsl/framework/bfc_allocator.cc:291] Allocator (GPU_0_bfc) ran out of memory trying to allocate 16.02GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ubuntu/single_gpu.py", line 30, in <module>
    result = simple_op(x, y)
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 17196646400 bytes.
```
When using all 8 GPUs on a `gpu_8x_a100_80gb_sxm4` instance the calculation succeeds.
```
python multiple_gpus.py
```
```
Available devices: [cuda(id=0) cuda(id=1) cuda(id=2) cuda(id=3) cuda(id=4) cuda(id=5)
 cuda(id=6) cuda(id=7)]
Created Mesh: Mesh('x': 2, 'y': 4)
2025-03-11 02:53:01.106732: W external/xla/xla/service/hlo_rematerialization.cc:3005] Can't reduce memory use below 41.19GiB (44223270085 bytes) by rematerialization; only reduced to 48.00GiB (51539607572 bytes), down from 48.00GiB (51539607572 bytes) originally
2025-03-11 02:53:14.062404: W external/xla/xla/service/hlo_rematerialization.cc:3005] Can't reduce memory use below 39.29GiB (42183160627 bytes) by rematerialization; only reduced to 48.00GiB (51539607568 bytes), down from 48.00GiB (51539607568 bytes) originally
result_gathered.shape: (65536, 65536)
```
