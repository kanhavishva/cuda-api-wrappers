/**
 * @file nvrtc/types.hpp
 *
 * @brief Type definitions used in CUDA real-time compilation work wrappers.
 */
#pragma once
#ifndef SRC_CUDA_NVRTC_TYPES_HPP_
#define SRC_CUDA_NVRTC_TYPES_HPP_

#include <cuda/api/types.hpp>

#include <vector>

#if __cplusplus >= 201703L
// #include <filesystem>
#include <string_view>
namespace cuda {
using string_view = std::string_view;
// namespace filesystem = std::filesystem;
}
#else
#include <cuda/nvrtc/detail/string_view.hpp>
namespace cuda {
using string_view = bpstd::string_view;
}
#endif

namespace cuda {

// C++ standard library doesn't provide a static vector,
// and we won't introduce our own here. So...
template <typename T>
using static_vector = std::vector<T>;

} // namespace cuda

#endif /* SRC_CUDA_NVRTC_TYPES_HPP_ */
