#include "config.h"

#define swap(a, b)                                                             \
  {                                                                            \
    __local int *tmp = a;                                                      \
    a = b;                                                                     \
    b = tmp;                                                                   \
  }

__kernel void scan_256(__global data_t *input, int n, __global data_t *chunks) {
  const int global_id = get_global_id(0);
  const int local_id = get_local_id(0);
  const int group_id = get_group_id(0);

  const int needed_groups = n / 256 + (n % 256 == 0 ? 0 : 1);

  if (group_id >= needed_groups)
    return;

  __local data_t buf1[256];
  __local data_t buf2[256];

  if (global_id < n) {
    buf1[local_id] = input[global_id];
    buf2[local_id] = input[global_id];
  } else {
    buf1[local_id] = 0;
    buf2[local_id] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  __local data_t *a = buf1;
  __local data_t *b = buf2;

  for (int s = 1; s < 256; s <<= 1) {
    if (local_id > (s - 1)) {
      b[local_id] = a[local_id] + a[local_id - s];
    } else {
      b[local_id] = a[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    swap(a, b)
  }

  // save last item of each chunk
  if (local_id == 255 && chunks != NULL) {
    if (group_id == needed_groups - 1) {
      chunks[0] = 0;
    } else {
      chunks[group_id + 1] = a[local_id];
    }
  }

  if (global_id < n) {
    input[global_id] = a[local_id];
  }
}

__kernel void add_chunk_sum(__global data_t *input, int n,
                            __global data_t *chunk_sum) {
  const int global_id = get_global_id(0);
  const int group_id = get_group_id(0);

  if (global_id >= n)
    return;

  input[global_id] += chunk_sum[group_id];
}