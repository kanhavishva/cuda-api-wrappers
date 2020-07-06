/**
 * @file kernel.hpp
 *
 * @brief Contains a base wrapper class for CUDA kernels - both statically and
 * dynamically compiled; and some related functionality.
 *
 * @note This file does _not_ define any kernels itself.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_KERNEL_HPP_
#define CUDA_API_WRAPPERS_KERNEL_HPP_

#include <cuda/api/types.hpp>
#include <cuda/api/error.hpp>

#include <cuda_runtime_api.h>
#include <cuda.h>

///@cond
class device_t;
///@nocond


namespace cuda {

namespace kernel {

/**
 * @brief a wrapper around `cudaFuncAttributes`, offering
 * a few convenience member functions.
 */
struct attributes_t : cudaFuncAttributes {
	attributes_t() = default;
	attributes_t(const  attributes_t&) = default;
	attributes_t(attributes_t&&) = default;
	attributes_t(cudaFuncAttributes& other) : cudaFuncAttributes(other) { }

	cuda::device::compute_capability_t ptx_version() const noexcept {
		return device::compute_capability_t::from_combined_number(ptxVersion);
	}

	cuda::device::compute_capability_t binary_compilation_target_architecture() const noexcept {
		return device::compute_capability_t::from_combined_number(binaryVersion);
	}
};

} // namespace kernel

/**
 * A non-owning wrapper for CUDA kernels - whether they be `__global__` functions compiled
 * apriori, or the result of dynamic NVRTC compilation, or obtained in some other future
 * way.
 *
 * @note The association of a `kernel_t` with an individual device or context is somewhat
 * tenuous. That is, the same function could be used with any other compatible device;
 * However, many/most of the features, attributes and settings are context-specific
 * (or device-specific?)
 */
class kernel_t {

public: // statics

	static const char* attribute_name(int attribute_index)
	{
		static const char* names[] = {
			"Maximum number of threads per block",
			"Statically-allocated shared memory size in bytes",
			"Required constant memory size in bytes",
			"Required local memory size in bytes",
			"Number of registers used by each thread",
			"PTX virtual architecture version into which the kernel code was compiled",
			"Binary architecture version for which the function was compiled",
			"Indication whether the function was compiled with cache mode CA",
			"Maximum allowed size of dynamically-allocated shared memory use size bytes",
			"Preferred shared memory carve-out to actual shared memory"
		}; // Note: The same set of attribute names is used both the readable and the settable ones
		return names[attribute_index];
	}

public: // getters
	const context_t context() const noexcept;
	/**
	 * Whether the kernel requires cooperation between different thread blocks.
	 *
	 * @note While this value is determined by the contents of the kernel, this dependency
	 * is not reflected either in the CUDA Driver API nor the Runtime API.
	 */
	bool thread_block_cooperation() const noexcept { return thread_block_cooperation_; }

public:
	const device_t device() const noexcept;

protected:
	context::handle_t context_id() const noexcept { return context_handle_; }

public: // non-mutators

	virtual int get_attribute(int attribute) const = 0;
	virtual kernel::attributes_t attributes() const = 0;

/*
	// The following are commented out because there are no CUDA API calls for them!
	// You may uncomment them if you'd rather get an exception...

	multiprocessor_cache_preference_t                cache_preference() const;
	multiprocessor_shared_memory_bank_size_option_t  shared_memory_bank_size() const;
*/

public: // mutators

	virtual void set_attribute(cudaFuncAttribute attribute, int value) = 0;

	/**
	 * @brief Change the hardware resource carve-out between L1 cache and shared memory
	 * for launches of the kernel to allow for at least the specified amount of
	 * shared memory.
	 *
	 * On several nVIDIA GPU micro-architectures, the L1 cache and the shared memory in each
	 * symmetric multiprocessor (=physical core) use the same hardware resources. The
	 * carve-out between the two uses has a device-wide value (which can be changed), but can
	 * also be set on the individual device-function level, by specifying the amount of shared
	 * memory the kernel may require.
	 */
	virtual void opt_in_to_extra_dynamic_memory(cuda::memory::shared::size_t amount_required_by_kernel) {
		set_attribute(cudaFuncAttributeMaxDynamicSharedMemorySize, amount_required_by_kernel);
	}

	/**
	 * @brief Indicate the desired carve-out between shared memory and L1 cache when launching
	 * this kernel - with coarse granularity.
	 *
	 * On several nVIDIA GPU micro-architectures, the L1 cache and the shared memory in each
	 * symmetric multiprocessor (=physical core) use the same hardware resources. The
	 * carve-out between the two uses has a device-wide value (which can be changed), but the
	 * driver can set another value for a specific function. This function doesn't make a demand
	 * from the CUDA runtime (as in @p opt_in_to_extra_dynamic_memory), but rather indicates
	 * what is the fraction of L1 to shared memory it would like the kernel scheduler to carve
	 * out.
	 *
	 * @param preference one of: as much shared memory as possible, as much
	 * L1 as possible, or no preference (i.e. using the device default).
	 *
	 * @note similar to @ref set_preferred_shared_mem_fraction() - but with coarser granularity.
	 */
	virtual void set_cache_preference(multiprocessor_cache_preference_t preference) = 0;

	/**
	 * @brief Sets a device function's preference of shared memory bank size preference
	 * (for the current device probably)
	 *
	 * @param config bank size setting to make
	 */
	virtual void set_shared_memory_bank_size(multiprocessor_shared_memory_bank_size_option_t config) = 0;

protected: // ctors & dtor
	kernel_t(context::handle_t context_id, bool thread_block_cooperation = false)
	: context_handle_(context_id), thread_block_cooperation_(thread_block_cooperation) { }

public: // ctors & dtor
	virtual ~kernel_t() = default;

protected: // data members
	const context::handle_t context_handle_;
	const bool thread_block_cooperation_;
};

} // namespace cuda

#endif // CUDA_API_WRAPPERS_KERNEL_HPP_
