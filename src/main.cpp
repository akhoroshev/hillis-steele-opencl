#include <cmath>
#include <iostream>
#include <stack>

#include "config.h"
#include "util.h"

std::vector<data_t> inclusive_scan(const std::vector<data_t> &input,
                                   cl::Device &device) {
  cl::Context context(device);
  cl::Program program(context, load_program("scan.cl"));
  cl::CommandQueue queue(context, device);
  program.build({device});

  cl::Kernel kernel_scan_256(program, "scan_256");
  cl::Kernel kernel_add_chunk_sum(program, "add_chunk_sum");

  const size_t block_size = 256;

  std::stack<std::pair<cl::Buffer, size_t>> to_calc;

  to_calc.emplace(cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                             input.size() * sizeof(data_t),
                             const_cast<data_t *>(input.data())),
                  input.size());

  while (true) {
    const cl::Buffer &top_buffer = to_calc.top().first;
    const size_t top_buffer_size = to_calc.top().second;

    cl::Buffer chunk_buffer;
    size_t chunk_count = ceil((double)top_buffer_size / block_size);

    if (top_buffer_size > block_size) {
      chunk_buffer =
          cl::Buffer(context, CL_MEM_READ_WRITE, chunk_count * sizeof(data_t));
    }

    kernel_scan_256.setArg(0, top_buffer);
    kernel_scan_256.setArg(1, static_cast<int>(top_buffer_size));
    kernel_scan_256.setArg(2, chunk_buffer);

    size_t grid_size = ceil((double)top_buffer_size / block_size) * block_size;
    queue.enqueueNDRangeKernel(kernel_scan_256, cl::NullRange,
                               cl::NDRange(grid_size), cl::NDRange(block_size));

    if (top_buffer_size > block_size) {
      to_calc.emplace(std::move(chunk_buffer), chunk_count);
    } else {
      break;
    }
  }

  while (to_calc.size() > 1) {
    auto [chunks, _] = std::move(to_calc.top());
    to_calc.pop();
    const auto &[result, n] = to_calc.top();

    kernel_add_chunk_sum.setArg(0, result);
    kernel_add_chunk_sum.setArg(1, static_cast<int>(n));
    kernel_add_chunk_sum.setArg(2, chunks);

    size_t grid_size = ceil((double)n / block_size) * block_size;
    queue.enqueueNDRangeKernel(kernel_add_chunk_sum, cl::NullRange,
                               cl::NDRange(grid_size), cl::NDRange(block_size));
  }

  std::vector<data_t> result(to_calc.top().second);
  queue.enqueueReadBuffer(to_calc.top().first, CL_TRUE, 0,
                          result.size() * sizeof(data_t), result.data());
  queue.finish();
  return result;
}

int main(int argc, char **argv) {

  auto devices = get_devices();
  if (devices.empty()) {
    std::cerr << "OpenCL devices not found" << std::endl;
    return 1;
  }

  size_t n;
  std::cin >> n;
  std::vector<data_t> a;
  a.resize(n);
  std::cin >> a;

  try {
    auto res = inclusive_scan(a, devices[0]);
    std::cout << res;
  } catch (const cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    return 1;
  }

  return 0;
}
