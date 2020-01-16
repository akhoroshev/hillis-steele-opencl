#pragma once

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <fstream>
#include <vector>

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &array) {
  for (size_t i = 0; i < array.size(); i++) {
    os << array[i];
    if (i < array.size() - 1) {
      os << ' ';
    }
  }
  return os;
}

template <typename T>
std::istream &operator>>(std::istream &is, std::vector<T> &array) {
  for (size_t i = 0; i < array.size(); i++) {
    T elem;
    is >> elem;
    array[i] = elem;
  }
  return is;
}

inline auto get_devices() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  std::vector<cl::Device> devices;
  for (const auto &platform : platforms) {
    std::vector<cl::Device> devs;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devs);
    devices.insert(devices.begin(), devs.begin(), devs.end());
  }

  return devices;
}

inline std::string load_program(const char *filename) {
  std::ifstream file(filename);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}