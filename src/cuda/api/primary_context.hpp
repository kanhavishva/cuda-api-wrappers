/**
 * @file primary_context.hpp
 */

#ifndef SRC_CUDA_DRIVER_API_PRIMARY_CONTEXT_HPP_
#define SRC_CUDA_DRIVER_API_PRIMARY_CONTEXT_HPP_

#include <cuda/api/context.hpp>
#include <cuda/api/current_context.hpp>

namespace cuda {

namespace device {

///@cond
class primary_context_t;
///@endcond

namespace primary_context {

using handle_t = context::handle_t;

// TODO: Make sure this actually means "current", not "active"
inline bool is_current()
{
	auto device_id = context::current::detail::get_device_id();
	unsigned flags;
	int active;
	auto status = cuDevicePrimaryCtxGetState(device_id, &flags, &active);
	throw_if_error(status, "Failed obtaining the state of the primary context for device " + std::to_string(device_id));
	return active;
}

namespace detail {

inline void decrease_refcount(device::id_t device_id)
{
	auto status = cuDevicePrimaryCtxRelease(device_id);
	throw_if_error(status, "Failed releasing the reference to the primary context for device " + std::to_string(device_id));
}

inline handle_t obtain_and_increase_refcount(device::id_t device_id)
{
	handle_t primary_context_id;
	auto status = cuDevicePrimaryCtxRetain(&primary_context_id, device_id);
	throw_if_error(status, "Failed obtaining (and possibly creating, and adding a reference count to) the primary context for device " + std::to_string(device_id));
	return primary_context_id;
}

// Really don't like this function!
inline handle_t get_id(device::id_t device_id)
{
	auto primary_context_id = obtain_and_increase_refcount(device_id);
	decrease_refcount(device_id);
	return primary_context_id;
}

} // namespace detail

/**
 * @brief Destroy and clean up all resources associated with the specified device's primary context
 *
 * @param device The device whose primary context is to be destroyed
 */
void destroy(const device_t& device);

namespace detail {

inline primary_context_t wrap(
	device::id_t      device_id,
	context::handle_t handle) noexcept;

} // namespace detail

} // namespace primary_context

class primary_context_t : public context_t {

protected: // constructors

	primary_context_t(
			device::id_t       device_id,
			context::handle_t  handle) noexcept
	: context_t(device_id, handle, false) { }

public: // friendship

	friend primary_context_t device::primary_context::detail::wrap(device::id_t, context::handle_t) noexcept;

public: // constructors and destructor

	primary_context_t(const primary_context_t& other) : primary_context_t(other.device_id_, other.handle_)
	{
		// We don't need the handle again, but we do need to increasing the
		// reference count - since we've created a new, independent, reference;
		// and our destructor will decrease the count later on.
#ifndef NDEBUG
		auto extra_handle = primary_context::detail::obtain_and_increase_refcount(device_id_);
		assert(extra_handle == handle_ and "Primary context handle inconsistency");
#else
		primary_context::detail::obtain_and_increase_refcount(device_id_);
#endif
	}
	primary_context_t(primary_context_t&& other) noexcept = default;
	// : primary_context_t(other.device_id_, other.handle_) { }

	~primary_context_t()
	{
		device::primary_context::detail::decrease_refcount(device_id_);
	}

public: // operators

	context_t& operator=(const primary_context_t& other) = delete;
	context_t& operator=(primary_context_t&& other) = delete;
};

namespace primary_context {

primary_context_t get(const device_t& device);

namespace detail {

// Note that destroying the wrapped instance decreases the refcount,
// meaning that the handle must have been obtained with an "unmatched"
// refcount increase
inline device::primary_context_t wrap(
	id_t     device_id,
	handle_t handle) noexcept
{
	return device::primary_context_t(device_id, handle);
}

} // namespace detail


} // namespace primary_context_t
} // namespace device

} // namespace cuda


/*

The primary context is mostly just a context, and the context class' methods can work for the primary. Still, an is_primary()
method would be useful - should it be backed by a field? It could... since at construction of a context wrapper, we know this.

Also, the construction and destruction are different - sometimes. For a non-owning reference to the primary context, we don't
do anything and just assume it exists, exactly like with any context. For an owning instance of the wrapper class - a regular
context has Create and Delete API calls, while the primary has Retain and Release : The primary context is like a shared pointer.

Finally - do we want to let the primary context reset itself? That's something which regular contexts can't do. We might
preclude this, and allow it through the device_t class, instead - it already has device_t::reset(), which should be the exact
same thing. We don't really care if that destroys contexts we're holding on to, because: 1. It won't cause segmentation
violations - we're not dereferencing freed pointers and 2. it's the user's problem, not ours.




 */


#endif /* SRC_CUDA_DRIVER_API_PRIMARY_CONTEXT_HPP_ */
