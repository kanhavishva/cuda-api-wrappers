/**
 * @file apriori_compiled_kernel.hpp
 *
 * @brief An implementation of a subclass of @ref `kernel_t`
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_APRIORI_COMPILED_KERNEL_HPP_
#define CUDA_API_WRAPPERS_APRIORI_COMPILED_KERNEL_HPP_

#include <cuda/api/kernel.hpp>
#include <cuda/api/current_context.hpp>
#include <type_traits>

///@cond
class device_t;
///@nocond


namespace cuda {

/**
 * @brief A subclass of the @ref `kernel_t` interface for kernels being
 * functions marked as __global__ in source files and compiled apriori.
 */
class apriori_compiled_kernel_t final : public kernel_t {
public: // getters
	const void *ptr() const noexcept { return ptr_; }

public: // type_conversions
	operator const void *() noexcept { return ptr_; }

public: // non-mutators
	int get_attribute(int attribute) const override;

	kernel::attributes_t attributes() const override;

	/**
	 * @brief Calculates the number of grid blocks which may be "active" on a given GPU
	 * multiprocessor simultaneously (i.e. with warps from any of these block
	 * being schedulable concurrently)
	 *
	 * @param num_threads_per_block
	 * @param dynamic_shared_memory_per_block
	 * @param disable_caching_override On some GPUs, the choice of whether to
	 * cache memory reads affects occupancy. But what if this caching results in 0
	 * potential occupancy for a kernel? There are two options, controlled by this flag.
	 * When it is set to false - the calculator will assume caching is off for the
	 * purposes of its work; when set to true, it will return 0 for such device functions.
	 * See also the "Unified L1/Texture Cache" section of the
	 * <a href="http://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html">Maxwell
	 * tuning guide</a>.
	 */
	grid::dimension_t maximum_active_blocks_per_multiprocessor(
		grid::block_dimension_t num_threads_per_block,
		memory::shared::size_t dynamic_shared_memory_per_block,
		bool disable_caching_override = false);

public: // mutators

	void set_attribute(cudaFuncAttribute attribute, int value) override;

	/**
	 *
	 * @param dynamic_shared_memory_size The amount of dynamic shared memory each grid block will
	 * need.
	 * @param block_size_limit do not return a block size above this value; the default, 0,
	 * means no limit on the returned block size.
	 * @param disable_caching_override On platforms where global caching affects occupancy,
	 * and when enabling caching would result in zero occupancy, the occupancy calculator will
	 * calculate the occupancy as if caching is disabled. Setting this to true makes the
	 * occupancy calculator return 0 in such cases. More information can be found about this
	 * feature in the "Unified L1/Texture Cache" section of the
	 * <a href="https://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html">Maxwell tuning guide</a>.
	 *
	 * @return A pair, with the second element being the maximum achievable block size
	 * (1-dimensional), and the first element being the minimum number of such blocks necessary
	 * for keeping the GPU "busy" (again, in a 1-dimensional grid).
	 */
	std::pair<grid::dimension_t, grid::block_dimension_t>
	min_grid_params_for_max_occupancy(
		memory::shared::size_t dynamic_shared_memory_size = no_dynamic_shared_memory,
		grid::block_dimension_t block_size_limit = 0,
		bool disable_caching_override = false);

	template<typename UnaryFunction>
	std::pair<grid::dimension_t, grid::block_dimension_t>
	min_grid_params_for_max_occupancy(
		UnaryFunction block_size_to_dynamic_shared_mem_size,
		grid::block_dimension_t block_size_limit = 0,
		bool disable_caching_override = false);


	void set_cache_preference(multiprocessor_cache_preference_t preference) override;

	void set_shared_memory_bank_size(multiprocessor_shared_memory_bank_size_option_t config) override;


	/**
	 * @brief Indicate the desired carve-out between shared memory and L1 cache when launching
	 * this kernel - with fine granularity.
	 *
	 * On several nVIDIA GPU micro-architectures, the L1 cache and the shared memory in each
	 * symmetric multiprocessor (=physical core) use the same hardware resources. The
	 * carve-out between the two uses has a device-wide value (which can be changed), but the
	 * driver can set another value for a specific function. This function doesn't make a demand
	 * from the CUDA runtime (as in @p opt_in_to_extra_dynamic_memory), but rather indicates
	 * what is the fraction of L1 to shared memory it would like the kernel scheduler to carve
	 * out.
	 *
	 * @param shared_mem_percentage The percentage - from 0 to 100 - of the combined L1/shared
	 * memory space the user wishes to assign to shared memory.
	 *
	 * @note similar to @ref set_cache_preference() - but with finer granularity.
	 */
	void set_preferred_shared_mem_fraction(unsigned shared_mem_percentage);


protected: // ctors & dtor
	apriori_compiled_kernel_t(context::handle_t context_handle, const void *f, bool thread_block_cooperation = false)
		: kernel_t(context_handle, thread_block_cooperation), ptr_(f) {
		// TODO: Consider checking whether this actually is a device function, at all and in this context
#ifndef NDEBUG
		assert(f != nullptr && "Attempt to construct a kernel object for a nullptr kernel function pointer");
#endif
	}

public: // ctors & dtor
	template<typename DeviceFunction>
	apriori_compiled_kernel_t(const device_t &device, DeviceFunction f, bool thread_block_cooperation = false);

	template<typename DeviceFunction>
	apriori_compiled_kernel_t(const context_t &context, DeviceFunction f, bool thread_block_cooperation = false);

protected: // data members
	const void *const ptr_;
};

inline kernel::attributes_t apriori_compiled_kernel_t::attributes() const {
	context::current::detail::scoped_override_t set_context_for_this_context(context_handle_);
	kernel::attributes_t function_attributes;
	auto status = cudaFuncGetAttributes(&function_attributes, ptr_);
	throw_if_error(status, "Failed obtaining attributes for a CUDA device function");
	return function_attributes;
}

inline void apriori_compiled_kernel_t::set_cache_preference(multiprocessor_cache_preference_t preference) {
	context::current::detail::scoped_override_t set_context_for_this_context(context_handle_);
	auto result = cudaFuncSetCacheConfig(ptr_, (cudaFuncCache) preference);
	throw_if_error(result,
		"Setting the multiprocessor L1/Shared Memory cache distribution preference for a "
		"CUDA device function");
}

inline void apriori_compiled_kernel_t::set_shared_memory_bank_size(
	multiprocessor_shared_memory_bank_size_option_t config) {
	context::current::detail::scoped_override_t set_context_for_this_context(context_handle_);
	auto result = cudaFuncSetSharedMemConfig(ptr_, static_cast<cudaSharedMemConfig>(config));
	throw_if_error(result, "Failed setting shared memory bank size");
}

inline grid::dimension_t apriori_compiled_kernel_t::maximum_active_blocks_per_multiprocessor(
	grid::block_dimension_t num_threads_per_block,
	memory::shared::size_t dynamic_shared_memory_per_block,
	bool disable_caching_override) {
	context::current::detail::scoped_override_t set_context_for_this_context(context_handle_);
	int result;
	unsigned int flags = disable_caching_override ?
						 cudaOccupancyDisableCachingOverride : cudaOccupancyDefault;
	auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
		&result, ptr_, num_threads_per_block,
		dynamic_shared_memory_per_block, flags);
	throw_if_error(status, "Failed calculating the maximum occupancy "
						   "of device function blocks per multiprocessor");
	return result;
}


template<typename DeviceFunction>
apriori_compiled_kernel_t::apriori_compiled_kernel_t(const device_t &device, DeviceFunction f,
													 bool thread_block_cooperation)
	: apriori_compiled_kernel_t(device.primary_context(), reinterpret_cast<const void *>(f),
	thread_block_cooperation) {}

template<typename DeviceFunction>
apriori_compiled_kernel_t::apriori_compiled_kernel_t(const context_t &context, DeviceFunction f,
													 bool thread_block_cooperation)
	: apriori_compiled_kernel_t(context.handle(), reinterpret_cast<const void *>(f), thread_block_cooperation) {}

inline int apriori_compiled_kernel_t::get_attribute(int attribute) const {
	auto attrs = attributes();
	switch (attribute) {
		case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
			return attrs.sharedSizeBytes;
		case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
			return attrs.constSizeBytes;
		case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
			return attrs.localSizeBytes;
		case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
			return attrs.maxThreadsPerBlock;
		case CU_FUNC_ATTRIBUTE_NUM_REGS:
			return attrs.numRegs;
		case CU_FUNC_ATTRIBUTE_PTX_VERSION:
			return attrs.ptxVersion;
		case CU_FUNC_ATTRIBUTE_BINARY_VERSION:
			return attrs.binaryVersion;
		case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA:
			return attrs.cacheModeCA;
		case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
			return attrs.maxDynamicSharedSizeBytes;
		case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
			return attrs.preferredShmemCarveout;
		default:
			throw std::invalid_argument("Invalid CUDA function attribute requested");
	}
}

inline void apriori_compiled_kernel_t::set_attribute(cudaFuncAttribute attribute, int value) {
#if CUDART_VERSION >= 9000
	context::current::detail::scoped_override_t set_context_for_this_context(context_handle_);
	auto result = cudaFuncSetAttribute(ptr_, attribute, value);
	throw_if_error(result,
		"Setting CUDA device function attribute " + std::to_string(attribute) + " to value " + std::to_string(value));
#else
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#endif
}

namespace kernel {

namespace detail {

// The helper code here is intended for re-imbuing kernel-related classes with the types
// of the kernel parameters. This is necessarily since kernel wrappers may be type-erased
// (which makes it much easier to work with them and avoids a bunch of code duplication).
//
// Note: The type-unerased kernel must be a non-const function pointer. Why? Not sure.
// even though function pointers can't get written through, for some reason they are
// expected not to be const.


template<typename... KernelParameters>
struct raw_kernel_typegen {
	// You should be careful to only instantiate this class with nice simple types we can pass to CUDA kernels.
//	static_assert(
//		all_true<
//		    std::is_same<
//		    	KernelParameters,
//		    	::cuda::detail::kernel_parameter_decay_t<KernelParameters>>::value...
//		    >::value,
//		"All kernel parameter types must be decay-invariant" );
	using type = void(*)(cuda::detail::kernel_parameter_decay_t<KernelParameters>...);
};

} // namespace detail

template<typename... KernelParameters>
typename detail::raw_kernel_typegen<KernelParameters...>::type
unwrap(apriori_compiled_kernel_t kernel)
{
	using raw_kernel_t = typename detail::raw_kernel_typegen<KernelParameters ...>::type;
	return reinterpret_cast<raw_kernel_t>(const_cast<void *>(kernel.ptr()));
}

} // namespace kernel

namespace detail {

template<typename... KernelParameters>
struct enqueue_launch_helper<apriori_compiled_kernel_t, KernelParameters...> {
	void operator()(
		apriori_compiled_kernel_t  wrapped_kernel,
		const stream_t &           stream,
		launch_configuration_t     launch_configuration,
		KernelParameters &&...     parameters);
};

} // namespace detail

} // namespace cuda

#endif // CUDA_API_WRAPPERS_APRIORI_COMPILED_KERNEL_HPP_
