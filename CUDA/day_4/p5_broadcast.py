import numba
import numpy as np
from numba import cuda

@cuda.jit
def call(out, a, b, size):
    # Get thread indices
    local_i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    local_j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if local_i < size and local_j < size:
        out[local_i, local_j] = a[local_i, 0] + b[0, local_j]


# Define matrix size
SIZE = 2

# Allocate memory
out = np.zeros((SIZE, SIZE), dtype=np.float32)
a = np.arange(SIZE, dtype=np.float32).reshape(SIZE, 1)
b = np.arange(SIZE, dtype=np.float32).reshape(1, SIZE)

# Allocate GPU memory
d_out = cuda.to_device(out)
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)

# Define block and grid sizes
threads_per_block = (16, 16)
blocks_per_grid_x = (SIZE + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (SIZE + threads_per_block[1] - 1) // threads_per_block[1]

# Launch kernel
call[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](d_out, d_a, d_b, SIZE)

# Copy result back to CPU
out = d_out.copy_to_host()

# Print results
print("Output Matrix:")
print(out)
