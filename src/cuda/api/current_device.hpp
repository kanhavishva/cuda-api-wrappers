/**
 * @file current_device.hpp
 *
 * @brief Wrappers for getting and setting CUDA's choice of
 * which device is 'current'
 *
 * CUDA has one device set as 'current'; and much of the Runtime API
 * implicitly refers to that device only. This file contains wrappers
 * for getting and setting it - as standalone functions - and
 * a RAII class which can be used for setting it for the duration of
 * a scope, popping back the old setting as the scope is exited.
 *
 * @note that code for getting the current device as a CUDA device
 * proxy class is found in @ref device.hpp
 *
 * @note the scoped device setter is used extensively throughout
 * this CUDA API wrapper library.
 *
 */
#ifndef CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_
#define CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_

#include <cuda/api/constants.hpp>
#include <cuda/api/miscellany.hpp>
#include <cuda/api/current_context.hpp>
#include <cuda/api/primary_context.hpp>

#include <cuda_runtime_api.h>

namespace cuda {

///@cond
class device_t;
///@endcond

namespace device {

namespace current {

namespace detail {

/**
 * Obtains the numeric id of the device set as current for the CUDA Runtime API
 */
inline id_t get_id()
{
	id_t  device;
	status_t result = cudaGetDevice(&device);
	throw_if_error(result, "Failure obtaining current device index");
	return device;
}

/**
 * Set a device as the current one for the CUDA Runtime API (so that API calls
 * not specifying a device apply to it.)
 *
 * @note this replaces the current CUDA context (rather than pushing a context
 * onto the stack), so use with care.
 *
 * @param[in] device Numeric ID of the device to make current
 */
inline void set(id_t device)
{
	status_t result = cudaSetDevice(device);
	throw_if_error(result, "Failure setting current device to " + std::to_string(device));
}

/**
 * Set the first possible of several devices to be the current one for the CUDA Runtime API.
 *
 * @param[in] device_ids Numeric IDs of the devices to try and make current, in order
 * @param[in] num_devices The number of device IDs pointed to by @device_ids
 *
 * @note this replaces the current CUDA context (rather than pushing a context
 * onto the stack), so use with care.
 */
inline void set(const id_t* device_ids, size_t num_devices)
{
	if (num_devices > static_cast<size_t>(cuda::device::count())) {
		throw cuda::runtime_error(status::invalid_device, "More devices listed than exist on the system");
	}
	auto result = cudaSetValidDevices(const_cast<int*>(device_ids), num_devices);
	throw_if_error(result, "Failure setting the current device to any of the list of "
		+ std::to_string(num_devices) + " devices specified");
}

/**
 * @note See the out-of-`detail::` version of this class.
 *
 * @note Perhaps it would be better to keep a copy of the current context ID in a member of this class?
 *
 * @note we have no guarantee that the context stack is not altered during
 * the lifetime of this object; but - we assume it wasn't, and it's up to the users
 * of this class to assure that's the case or face the consequences
 *
 * @note We don't want to use the context setter class as the implementation,
 * since that would involve creating a primary context for the device, which
 * we still want to avoid.
 */

class scoped_override_t {
public:
	scoped_override_t(id_t new_device_id) {
		auto top_of_context_stack = context::current::detail::get_handle();
		if (top_of_context_stack != context::current::detail::no_current_context) {
			context::current::detail::push(top_of_context_stack); // Yes, we're pushing a copy of the same context
		}
		device::current::detail::set(new_device_id); // ... which now gets overwritten at the top of the stack
	}
	~scoped_override_t() {
		context::current::detail::pop();
	}
};


} // namespace detail

/**
 * Reset the CUDA Runtime API's current device to its default value - the default device
 */
inline void set_to_default() { return detail::set(device::default_device_id); }

void set(device_t device);

/**
 * A RAII-like mechanism for setting the CUDA Runtime API's current device for
 * what remains of the current scope, and changing it back to its previous value
 * when exiting the scope.
 *
 * @note The description says "RAII-like" because the reality is more complex. The
 * runtime API sets a device by overwriting the current
 */
class scoped_override_t : private detail::scoped_override_t {
protected:
	using parent = detail::scoped_override_t;
public:
	scoped_override_t(const device_t& device);
	scoped_override_t(device_t&& device);
	~scoped_override_t() = default;
};

/**
 * This macro will set the current device for the remainder of the scope in which it is
 * invoked, and will change it back to the previous value when exiting the scope. Use
 * it as an opaque command, which does not explicitly expose the variable defined under
 * the hood to effect this behavior.
 */
#define CUDA_DEVICE_FOR_THIS_SCOPE(_cuda_device_ctor_argument) \
	::cuda::device::current::scoped_override_t scoped_device_override( ::cuda::device_t(_cuda_device_ctor_argument) )


} // namespace current
} // namespace device
} // namespace cuda

#endif // CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_
