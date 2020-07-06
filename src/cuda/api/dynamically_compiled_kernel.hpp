/**
 * @file dyanmically_compiled_kernel.hpp
 *
 * @brief An implementation of a subclass of @ref `kernel_t`
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DYNAMICALLY_COMPILED_KERNEL_HPP_
#define CUDA_API_WRAPPERS_DYNAMICALLY_COMPILED_KERNEL_HPP_

#include <cuda/api/kernel.hpp>
#include <cuda/api/current_context.hpp>
#include <cuda/api/module.hpp>
#include <array>

namespace cuda {

/**
 * @brief A subclass of the @ref `kernel_t` interface for kernels
 * which are compiled dynamically using NVRTC (or at least - with
 * dynamically-loaded binary objects.
 */
class dynamically_compiled_kernel_t final : public kernel_t {

public: // getters
	CUfunction handle() const noexcept { return handle_; }

public: // non-mutators
	int get_attribute(int attribute) const override;
	kernel::attributes_t attributes() const override;

public: // mutators

	void set_attribute(cudaFuncAttribute attribute, int value) override;
	void set_cache_preference(multiprocessor_cache_preference_t preference) override;
	void set_shared_memory_bank_size(multiprocessor_shared_memory_bank_size_option_t config) override;

protected: // ctors & dtor
	dynamically_compiled_kernel_t(context::handle_t context_id, CUfunction f, bool thread_block_cooperation = false)
	: kernel_t(context_id, thread_block_cooperation), handle_(f) { }

	friend  dynamically_compiled_kernel_t wrap(context::handle_t, CUfunction, bool);

public: // ctors & dtor
	dynamically_compiled_kernel_t(const dynamically_compiled_kernel_t& other) = default;
	dynamically_compiled_kernel_t(dynamically_compiled_kernel_t&& other) = default;

protected: // data members
	const CUfunction handle_;
};

inline dynamically_compiled_kernel_t wrap(
	context::handle_t  context_id,
	CUfunction         f,
	bool               thread_block_cooperation = false)
{
	return dynamically_compiled_kernel_t(context_id, f, thread_block_cooperation);
}

inline int dynamically_compiled_kernel_t::get_attribute(int attribute) const
{
	// TODO: Shouldn't I set the context?
	int attribute_value;
	auto result = cuFuncGetAttribute(&attribute_value, static_cast<CUfunction_attribute>(attribute), handle_);
	throw_if_error(result, std::string("Failed obtaining attribute ") + kernel_t::attribute_name(attribute) );
	return attribute_value;
}

inline kernel::attributes_t dynamically_compiled_kernel_t::attributes() const
{
	// TODO: Perhaps
	auto attrs = cudaFuncAttributes{
		static_cast<size_t>(get_attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)),
		static_cast<size_t>(get_attribute(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES)),
		static_cast<size_t>(get_attribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)),
		get_attribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
		get_attribute(CU_FUNC_ATTRIBUTE_NUM_REGS),
		get_attribute(CU_FUNC_ATTRIBUTE_PTX_VERSION),
		get_attribute(CU_FUNC_ATTRIBUTE_BINARY_VERSION),
		get_attribute(CU_FUNC_ATTRIBUTE_CACHE_MODE_CA),
		get_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES),
		get_attribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT)
	};
	return {attrs};
}

inline void dynamically_compiled_kernel_t::set_attribute(cudaFuncAttribute attribute, int value)
{
#if CUDART_VERSION >= 9000
	context::current::detail::scoped_override_t set_context_for_this_context(context_handle_);
	auto result = cuFuncSetAttribute(handle_, static_cast<CUfunction_attribute>(attribute), value);
	throw_if_error(result,
		"Setting CUDA device function attribute " + std::string(attribute_name(attribute)) +
		" to value " + std::to_string(value));
#else
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#endif
}

inline void dynamically_compiled_kernel_t::set_cache_preference(multiprocessor_cache_preference_t  preference)
{
	context::current::detail::scoped_override_t set_context_for_this_context(context_handle_);
	auto result = cuFuncSetCacheConfig(handle_, (CUfunc_cache) preference);
	throw_if_error(result,
		"Setting the multiprocessor L1/Shared Memory cache distribution preference for a "
		"CUDA device function");
}

inline void dynamically_compiled_kernel_t::set_shared_memory_bank_size(
	multiprocessor_shared_memory_bank_size_option_t  config)
{
	// TODO: Need to set a context, not a device
	context::current::detail::scoped_override_t set_context_for_this_context(context_handle_);
	auto result = cuFuncSetSharedMemConfig(handle_, static_cast<CUsharedconfig>(config) );
	throw_if_error(result, "Failed setting the shared memory bank size");
}

namespace detail {

template<typename... KernelParameters>
std::array<void*, sizeof...(KernelParameters)>
marshal_dynamic_kernel_arguments(KernelParameters&&... parameters)
{
	return std::array<void*, sizeof...(KernelParameters)>
	{
//		(std::is_pointer<kernel_parameter_decay_t<KernelParameters>>::value ?
//			reinterpret_cast<void*>(parameters) :
//			&parameters
//		)
		&parameters
		...
	};

}

template<typename... KernelParameters>
struct enqueue_launch_helper<dynamically_compiled_kernel_t, KernelParameters...> {

void operator()(
	dynamically_compiled_kernel_t  wrapped_kernel,
	const stream_t &               stream,
	launch_configuration_t         lc,
	KernelParameters &&...         parameters)
{
	auto marshalled_arguments { marshal_dynamic_kernel_arguments(std::forward<KernelParameters>(parameters)...) };
	auto function_handle = wrapped_kernel.handle();
	CUresult status;
	if (wrapped_kernel.thread_block_cooperation())
		status = cuLaunchCooperativeKernel(
			function_handle,
			lc.grid_dimensions.x,  lc.grid_dimensions.y,  lc.grid_dimensions.z,
			lc.block_dimensions.x, lc.block_dimensions.y, lc.block_dimensions.z,
			lc.dynamic_shared_memory_size,
			stream.id(),
			marshalled_arguments.data()
		);
	else {
		constexpr const auto no_arguments_in_alternative_format = nullptr; // TODO: Consider passing arguments in this format
		status = cuLaunchKernel(
			function_handle,
			lc.grid_dimensions.x,  lc.grid_dimensions.y,  lc.grid_dimensions.z,
			lc.block_dimensions.x, lc.block_dimensions.y, lc.block_dimensions.z,
			lc.dynamic_shared_memory_size,
			stream.id(),
			marshalled_arguments.data(),
			no_arguments_in_alternative_format
		);
	}
	throw_if_error(status,
		std::string("Failed launching the ")
		+ (wrapped_kernel.thread_block_cooperation() ? "cooperative " : "")
		+ " kernel at " + detail::ptr_as_hex(function_handle));
}

};

} // namespace detail

} // namespace cuda

#endif // CUDA_API_WRAPPERS_DYNAMICALLY_COMPILED_KERNEL_HPP_
