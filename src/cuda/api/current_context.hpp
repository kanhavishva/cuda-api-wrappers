/**
 * @file current_context.hpp
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_CURRENT_CONTEXT_HPP_
#define CUDA_API_WRAPPERS_CURRENT_CONTEXT_HPP_

#include <cuda/api/types.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/versions.hpp>

#include <cuda.h>

namespace cuda {

///@cond
class device_t;
class context_t;
///@endcond

namespace context {

namespace detail {
using flags_t = unsigned;
} // namespace detail

namespace current {

namespace detail {

constexpr const CUcontext no_current_context { 0 };

/**
 * Returns a raw handle for the current CUDA context
 *
 * @return the raw handle from the CUDA driver - if one exists; no_current_context
 * if no context is current/active.
 */
inline handle_t get_handle()
{
	handle_t handle;
	auto status = cuCtxGetCurrent(&handle);
	throw_if_error(status, "Failed obtaining the current CUDA context");
	return handle;
}

// Note: not calling this get_ since flags are read-only anyway
inline context::detail::flags_t flags()
{
	context::detail::flags_t result;
	auto status = cuCtxGetFlags(&result);
	throw_if_error(status, "Failed obtaining CUDA context flags");
	return result;
}

inline device::id_t get_device_id()
{
	device::id_t device_id;
	auto result = cuCtxGetDevice(&device_id);
	throw_if_error(result, "Failed obtaining the current context's device");
	return device_id;
}

} // namespace detail

inline bool exists();
inline context_t get();
inline void set(const context_t& context);

namespace detail {

/**
 * Push a context handle onto the top of the context stack - if it is not already on the
 * top of the stack
 *
 * @param context_handle A context handle to push
 *
 * @note behavior undefined if you try to push @ref no_current_context
 */
inline void push(handle_t context_handle)
{
	status_t status;
	status = static_cast<status_t>(cuCtxPushCurrent(context_handle));
	throw_if_error(status,
			"Failed pushing the context " + cuda::detail::ptr_as_hex(context_handle) +
			" to the to of the context stack");
}

/**
 * Push a context handle onto the top of the context stack - if it is not already on the
 * top of the stack
 *
 * @param context_handle A context handle to push
 *
 * @return true if a push actually occurred
 *
 * @note behavior undefined if you try to push @ref no_current_context
 * @note The CUDA context stack is not a proper stack, in that it doesn't allow multiple
 * consecutive copes of the same context on the stack; hence there is no `push()` method.
 */
inline bool push_if_not_on_top(handle_t context_handle)
{
	if (detail::get_handle() == context_handle) { return false; }
	push(context_handle); return true;
}

inline context::handle_t pop()
{
	handle_t popped_context_handle;
	auto status = cuCtxPopCurrent(&popped_context_handle);
	throw_if_error(status, "Failed popping the current CUDA context");
	return popped_context_handle;
}

inline void set(handle_t context_handle)
{
	// TODO: Would this help?
	// if (detail::get_handle() == context_handle) { return; }
	auto status = static_cast<status_t>(cuCtxSetCurrent(context_handle));
	throw_if_error(status,
	    "Failed setting the current context to " + cuda::detail::ptr_as_hex(context_handle));
}

/**
 * @note See the out-of-`detail::` version of this class.
 *
 */
class scoped_override_t {
protected:
public:
	explicit scoped_override_t(handle_t context_handle) { push(context_handle); }
	~scoped_override_t() { pop(); }

//	explicit scoped_override_t(handle_t context_handle) :
//		did_push(push_if_not_on_top(context_handle)) { }
//	~scoped_override_t() { if (did_push) { pop(); } }
//
//protected:
//	bool did_push;
};

class scoped_current_device_fallback_t;

} // namespace detail

/**
 * A RAII-based mechanism for setting the current context for what remains of the
 * current (C++ language) scope, and changing it back to its previous value when
 * exiting the scope - restoring the context stack to its previous state.
 *
 */
class scoped_override_t : private detail::scoped_override_t {
protected:
	using parent = detail::scoped_override_t;
public:
	explicit scoped_override_t(const context_t& device);
    explicit scoped_override_t(context_t&& device);
	~scoped_override_t() = default;
};

/**
 * This macro will set the current device for the remainder of the scope in which it is
 * invoked, and will change it back to the previous value when exiting the scope. Use
 * it as an opaque command, which does not explicitly expose the variable defined under
 * the hood to effect this behavior.
 */
#define CUDA_CONTEXT_FOR_THIS_SCOPE(_cuda_context_ctor_argument) \
	::cuda::context::current::scoped_override_t scoped_device_override( ::cuda::context_t(_cuda_context_ctor_argument) )


inline bool push_if_not_on_top(const context_t& context);
inline void push(const context_t& context);

} // namespace current


} // namespace context

} // namespace cuda

#endif // CUDA_API_WRAPPERS_CURRENT_CONTEXT_HPP_
