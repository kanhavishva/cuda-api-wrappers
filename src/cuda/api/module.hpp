/**
 * @file module.hpp
 *
 * @brief Wrappers for working with modules of JIT-compiled CUDA code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_MODULE_HPP_
#define CUDA_API_WRAPPERS_MODULE_HPP_

#include <cuda/api/context.hpp>
#include <cuda/api/dynamically_compiled_kernel.hpp>
#include <cuda/api/memory.hpp>
#include <cuda/api/array.hpp>
#include <cuda/api/link_options.hpp>
#include <cuda.h>
#include <array>

#if __cplusplus >= 201703L
#include <filesystem>
#endif

namespace cuda {

///@cond
class device_t;
class context_t;
class module_t;
///@endcond

namespace module {

using handle_t = CUmodule;

namespace detail {

inline module_t wrap(
	context::handle_t             context_handle,
	handle_t                      handle,
	link::options_t  options,
	bool                          take_ownership = false) noexcept;

} // namespace detail

/**
 * Load a module from an appropriate compiled or semi-compiled file, allocating all
 * relevant resources for it.
 *
 * @param path of a cubin, PTX, or fatbin file constituting the module to be loaded.
 * @return the loaded module
 *
 * @note this covers cuModuleLoadFatBinary() even though that's not directly used
 */
inline module_t load_from_file(const char* path, link::options_t options = {});
inline module_t load_from_file(const std::string& path, link::options_t options = {});
#if __cplusplus >= 201703L
inline module_t load_from_file(const std::filesystem::path& path, link::options_t options = {});
#endif

inline module_t create(const context_t& context, memory::region_t image, const link::options_t options);
inline module_t create(const context_t& context, memory::region_t image);

} // namespace module

/**
 * Wrapper class for a CUDA code module
 *
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to the module is a const-respecting operation on this class.
 */
class module_t {

public: // getters

	context::handle_t context_handle() const { return context_handle_; }
	context_t context() const;
	device_t device() const;

	// These API calls are not really the way you want to work.
	dynamically_compiled_kernel_t get_kernel(const char* name, bool thread_block_cooperation = false) const {
		context::current::detail::scoped_override_t set_context_for_this_scope(context_handle_);
		CUfunction fn;
		auto result = cuModuleGetFunction(&fn, handle_, name);
		throw_if_error(result, std::string("Failed obtaining function ") + name
			+ " from the module with id " + detail::ptr_as_hex(handle_)
			+ " in context " + detail::ptr_as_hex(context_handle_));
		return wrap(context_handle_, fn, thread_block_cooperation );
	}
	cuda::memory::region_t get_global_object(const char* name) const;

	// TODO: Implement a surface reference and texture reference class rather than these raw pointers.

	CUsurfref* get_surface(const char* name) const;
	CUtexref* get_texture_reference(const char* name) const;

protected: // constructors

	module_t(context::handle_t context, module::handle_t handle, link::options_t options, bool take_ownership) noexcept
	: context_handle_(context), handle_(handle), options_(options), owning_(take_ownership) { }

public: // friendship

	friend module_t module::detail::wrap(context::handle_t context, module::handle_t handle, link::options_t options, bool take_ownership) noexcept;


public: // constructors and destructor

	module_t(const module_t&) = delete;

	module_t(module_t&& other) noexcept :
		module_t(other.context_handle_, other.handle_, other.options_, other.owning_)
	{
		other.owning_ = false;
	};

	~module_t()
	{
		if (owning_) {
			context::current::detail::scoped_override_t set_context_for_this_scope(context_handle_);
			auto status = cuModuleUnload(handle_);
			throw_if_error(status,
				std::string("Failed unloading module " + detail::ptr_as_hex(handle_) +
				" from context " + detail::ptr_as_hex(context_handle_)));
		}
	}

public: // operators

	module_t& operator=(const module_t& other) = delete;
	module_t& operator=(module_t&& other) = delete;

protected: // data members
	const context::handle_t  context_handle_;
	const module::handle_t   handle_;
	link::options_t           options_;
	bool                     owning_;
		// this field is mutable only for enabling move construction; other
		// than in that case it must not be altered
};

namespace module {

using handle_t = CUmodule;

inline module_t load_from_file(const char* path, link::options_t options)
{
	handle_t new_module_handle;
	auto status = cuModuleLoad(&new_module_handle, path);
	throw_if_error(status, std::string("Failed loading a module from file ") + path);
	bool do_take_ownership { true };
	return detail::wrap(context::current::detail::get_handle(), new_module_handle, options, do_take_ownership);
}

inline module_t load_from_file(const std::string& path, link::options_t options)
{
	return load_from_file(path.c_str(), options);
}

#if __cplusplus >= 201703L
inline module_t load_from_file(const std::filesystem::path& path)
{
	return load_from_file(path.c_str());
}
#endif
inline module_t create(const context_t& context, memory::region_t image, link::options_t options)
{
	context::current::scoped_override_t set_context_for_this_scope(context);
	handle_t new_module_handle;
	auto marshalled_options = options.marshal();
	auto status = cuModuleLoadDataEx(
		&new_module_handle,
		image.data(),
		marshalled_options.count(),
		const_cast<link::option_t*>(marshalled_options.options()),
		const_cast<void**>(marshalled_options.values())
	);
	throw_if_error(status, std::string("Failed loading a module from memory location ")
		+ cuda::detail::ptr_as_hex(image.data())
		+ "within context " + cuda::detail::ptr_as_hex(context.handle())
		+ " on device " + std::to_string(context.device_id()));
	bool do_take_ownership { true };
	return detail::wrap(
		context.handle(),
		new_module_handle,
		options,
		do_take_ownership);
}

inline module_t create(context_t& context, memory::region_t image)
{
	context::current::scoped_override_t set_context_for_this_scope(context);
	handle_t new_module_handle;
	auto status = cuModuleLoadData(&new_module_handle, image.data());
	throw_if_error(status, std::string(
		"Failed loading a module from memory location ") + cuda::detail::ptr_as_hex(image.data()) +
		"within context " + cuda::detail::ptr_as_hex(context.handle()) + " on device " + std::to_string(context.device_id()));
	bool do_take_ownership { true };
	// TODO: Make sure the default-constructed options correspond to what cuModuleLoadData uses as defaults
	return detail::wrap(context.handle(), new_module_handle, link::options_t{}, do_take_ownership);
}

namespace detail {

inline module_t wrap(
	context::handle_t             context_handle,
	handle_t                      module_handle,
	link::options_t  options,
	bool                          take_ownership) noexcept
{
	return module_t{context_handle, module_handle, options, take_ownership};
}

} // namespace detail

} // namespace module

} // namespace cuda

#endif // CUDA_API_WRAPPERS_MODULE_HPP_
