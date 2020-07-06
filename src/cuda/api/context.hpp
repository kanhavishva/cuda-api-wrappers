/**
 * @file context.hpp
 *
 * @brief Contains a proxy class for CUDA execution contexts.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_CONTEXT_HPP_
#define CUDA_API_WRAPPERS_CONTEXT_HPP_

#include <cuda/api/types.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/versions.hpp>
#include <cuda/api/current_context.hpp>

#include <cuda.h>

namespace cuda {

///@cond
class device_t;
class context_t;
class stream_t;
///@endcond

namespace context {

using limit_value_t = size_t;

/**
 * A range of priorities supported by a CUDA context; ranges from the
 * higher numeric value to the lower.
 */
struct stream_priority_range_t {
	stream::priority_t least; /// Higher numeric value, lower priority
	stream::priority_t greatest; /// Lower numeric value, higher priority

	/**
	 * When true, stream prioritization is not supported, i.e. all streams have
	 * "the same" priority - the default one.
	 */
	constexpr bool is_trivial() const {
		return least == stream::default_priority and greatest == stream::default_priority;
	}
};

namespace detail {

inline limit_value_t get_limit(CUlimit limit_id)
{
	size_t limit_value;
	auto status = cuCtxGetLimit(&limit_value, limit_id);
	throw_if_error(status,
			"Failed obtaining CUDA context limit value");
	return limit_value;
}

inline void set_limit(CUlimit limit_id, limit_value_t new_value)
{
	auto status = cuCtxSetLimit(limit_id, new_value);
	throw_if_error(status, "Failed obtaining CUDA context limit value");
}

constexpr flags_t inline make_flags(
	host_thread_synch_scheduling_policy_t  synch_scheduling_policy,
	bool                                   keep_larger_local_mem_after_resize,
	bool                                   allow_pinned_mapped_memory_allocation)
{
	return(
		  synch_scheduling_policy // this enum value is also a valid bitmask
		| (keep_larger_local_mem_after_resize    ? cudaDeviceLmemResizeToMax : 0)
		| (allow_pinned_mapped_memory_allocation ? cudaDeviceMapHost         : 0));
}

inline device::id_t get_device_id(handle_t context_handle)
{
	auto needed_push = current::detail::push_if_not_on_top(context_handle);
	auto device_id = current::detail::get_device_id();
	if (needed_push) {
		current::detail::pop();
	}
	return device_id;
}



/**
 * @brief Wrap an existing CUDA context in a @ref context_t instance
 *
 * @param device_id ID of the device for which the context is defined
 * @param context_id
 * @param take_ownership When set to `false`, the CUDA context
 * will not be destroyed along with its proxy. When set to `true`,
 * the proxy class will destroy the context when itself being destructed.
 * @return The constructed `cuda::context_t`.
 */
context_t wrap(
	device::id_t       device_id,
	context::handle_t  context_id,
	bool               take_ownership = false) noexcept;

context_t from_handle(
	context::handle_t  context_handle,
	bool               take_ownership = false);

} // namespace detail

} // namespace context

/**
 * @brief Wrapper class for a CUDA context
 *
 * Use this class - built around a context id - to perform all
 * context-related operations the CUDA Driver (or, in fact, Runtime) API is capable of.
 *
 * @note By default this class has RAII semantics, i.e. it creates a
 * context on construction and destroys it on destruction, and isn't merely
 * an ephemeral wrapper one could apply and discard; but this second kind of
 * semantics is also supported, through the @ref context_t::owning_ field.
 *
 * @note A context is a specific to a device; see, therefore, also @ref device_t .
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to properties of the context is a const-respecting operation on this class.
 */
class context_t {
public: // types
	using scoped_setter_type = context::current::detail::scoped_override_t;
	using shared_memory_bank_size_t = cudaSharedMemConfig;
	using limit_value_t = context::limit_value_t;

	static_assert(
		std::is_same< std::underlying_type<CUsharedconfig>::type, std::underlying_type<cudaSharedMemConfig>::type >::value,
		"Unexpected difference between enumerators used for the same purpose by the CUDA runtime and the CUDA driver");

public: // data member non-mutator getters

	/**
	 * The CUDA context ID this object is wrapping
	 */
	context::handle_t handle() const noexcept { return handle_; }

	/**
	 * The device with which this context is associated
	 */
	device::id_t device_id() const noexcept { return device_id_; }
	device_t device() const;

	/**
	 * Is this wrapper responsible for having the wrapped CUDA context destroyed on destruction?
	 */
	bool is_owning() const noexcept { return owning_;  }

public: // other non-mutator methods

	/**
	 * Determines the balance between L1 space and shared memory space set
	 * for kernels executing within this context.
	 */
	multiprocessor_cache_preference_t cache_preference() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		CUfunc_cache raw_preference;

		auto status = cuCtxGetCacheConfig(&raw_preference);
		throw_if_error(
			status,
			"Obtaining the multiprocessor L1/Shared Memory cache distribution preference for context " + detail::ptr_as_hex(handle_));
		return static_cast<multiprocessor_cache_preference_t>(raw_preference);
	}

	/**
	 * @return the stack size in bytes of each GPU thread
	 *
	 * @todo Is this really a feature of the context? Not of the device?
	 */
	size_t stack_size() const
	{
		return context::detail::get_limit(CU_LIMIT_STACK_SIZE);
	}

	/**
	 * @return the size of the FIFO (first-in, first-out) buffer used by the printf() function available to device kernels
	 *
	 * @todo Is this really a feature of the context? Not of the device?
	 */
	limit_value_t printf_buffer_size() const
	{
		return context::detail::get_limit(CU_LIMIT_PRINTF_FIFO_SIZE);
	}

	/**
	 * @return the size in bytes of the heap used by the malloc() and free() device system calls.
	 *
	 * @todo Is this really a feature of the context? Not of the device?
	 */
	limit_value_t memory_allocation_heap_size() const
	{
		return context::detail::get_limit(CU_LIMIT_MALLOC_HEAP_SIZE);
	}

	/**
	 * @return the maximum grid depth at which a thread can issue the device
	 * runtime call `cudaDeviceSynchronize()` / `cuda::device::synchronize()`
     * to wait on child grid launches to complete.
	 *
	 * @todo Is this really a feature of the context? Not of the device?
	 */
	limit_value_t maximum_depth_of_child_grid_synch_calls() const
	{
		return context::detail::get_limit(CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH);
	}

	/**
	 * @return maximum number of outstanding device runtime launches that can be made from this context.
	 *
	 * @todo Is this really a feature of the context? Not of the device?
	 */
	limit_value_t maximum_outstanding_kernel_launches() const
	{
		return context::detail::get_limit(CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT);
	}


	/**
	 * @return maximum granularity of fetching from the L2 cache
	 *
	 * @note A value between 0 and 128; it is apparently a "hint" somehow.
	 *
	 * @todo Is this really a feature of the context? Not of the device?
	 */
	limit_value_t l2_fetch_granularity() const
	{
		return context::detail::get_limit(CU_LIMIT_MAX_L2_FETCH_GRANULARITY);
	}

	/**
	 * @brief Returns the shared memory bank size, as described in
	 * <a href="https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/">this Parallel-for-all blog entry</a>
	 *
	 * @return the shared memory bank size in bytes
	 */
	shared_memory_bank_size_t shared_memory_bank_size() const
	{
		scoped_setter_type set_device_for_this_scope(handle_);
		CUsharedconfig bank_size;

		auto status = cuCtxGetSharedMemConfig(&bank_size);
		throw_if_error(status,
			"Obtaining the multiprocessor shared memory bank size for context " + detail::ptr_as_hex(handle_));

		return static_cast<shared_memory_bank_size_t>(bank_size);
	}

	bool is_current() const
	{
		// Note: assuming context handles are unique over multiple devices
		return handle_ == context::current::detail::get_handle();
	}

	bool is_primary() const;

	/**
	 *
	 * @todo isn't this a feature of devices?
	 */
	context::stream_priority_range_t stream_priority_range() const
	{
		scoped_setter_type set_device_for_this_scope(handle_);
		context::stream_priority_range_t result;
		auto status = cuCtxGetStreamPriorityRange(&result.least, &result.greatest);
		throw_if_error(
				status,
				"Obtaining the priority range for streams within context " + detail::ptr_as_hex(handle_));
		return result;
	}

protected:
	context::detail::flags_t flags() const
	{
		scoped_setter_type set_device_for_this_scope(handle_);
		return context::current::detail::flags();
	}

public:
	host_thread_synch_scheduling_policy_t synch_scheduling_policy() const
	{
		return host_thread_synch_scheduling_policy_t(flags() & CU_CTX_SCHED_MASK);
	}

	bool keeping_larger_local_mem_after_resize() const
	{
		return flags() & CU_CTX_LMEM_RESIZE_TO_MAX;
	}

	/**
	 * Can we allocated mapped pinned memory on this device?
	 */
	bool can_map_host_memory() const
	{
		return flags() & CU_CTX_MAP_HOST;
	}

public: // Methods which don't mutate the context, but affect the device itself

	void reset_persisting_l2_cache() const
	{
#if (CUDART_VERSION >= 11000)
		CUresult status = cuCtxResetPersistingL2Cache();
		throw_if_error(status, "Failed resetting/clearing the persisting L2 cache memory");
#endif
		throw cuda::runtime_error(
			cuda::status::insufficient_driver,
			"Resetting/clearing the persisting L2 cache memory is not supported when compiling CUDA versions lower than 11.0");
	}

public: // other mutator methods

	/**
	 * @brief Sets the shared memory bank size, described in
	 * <a href="https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/">this Parallel-for-all blog entry</a>
	 *
	 * @param new_bank_size the shared memory bank size to set, in bytes
	 */
	void set_shared_memory_bank_size(shared_memory_bank_size_t new_bank_size) const
	{
		scoped_setter_type set_device_for_this_scope(handle_);
		auto status = cuCtxSetSharedMemConfig(static_cast<CUsharedconfig>(new_bank_size));
		throw_if_error(
			status,
			"Setting the multiprocessor shared memory bank size for context " + detail::ptr_as_hex(handle_));
	}


	/**
	 * Controls the balance between L1 space and shared memory space for
	 * kernels executing within this context.
	 *
	 * @param preference the preferred balance between L1 and shared memory
	 */
	void set_cache_preference(multiprocessor_cache_preference_t preference) const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		auto status = cuCtxSetCacheConfig(static_cast<CUfunc_cache>(preference));
		throw_if_error(status,
			"Setting the multiprocessor L1/Shared Memory cache distribution preference for context " + detail::ptr_as_hex(handle_));
	}

	limit_value_t get_limit(CUlimit limit_id) const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail::get_limit(limit_id);
	}
	limit_value_t get_limit(cudaLimit limit_id) const
	{
		return get_limit(CUlimit(limit_id));
	}

	void set_limit(CUlimit limit_id, limit_value_t new_value) const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail::set_limit(limit_id, new_value);
	}
	void set_limit(cudaLimit limit_id, limit_value_t new_value) const
	{
		set_limit(CUlimit(limit_id), new_value);
	}

	void stack_size(limit_value_t new_value) const
	{
		return set_limit(CU_LIMIT_STACK_SIZE, new_value);
	}

	void printf_buffer_size(limit_value_t new_value) const
	{
		return set_limit(CU_LIMIT_PRINTF_FIFO_SIZE, new_value);
	}

	void memory_allocation_heap_size(limit_value_t new_value) const
	{
		return set_limit(CU_LIMIT_MALLOC_HEAP_SIZE, new_value);
	}

	void set_maximum_depth_of_child_grid_synch_calls(limit_value_t new_value) const
	{
		return set_limit(CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH, new_value);
	}

	void set_maximum_outstanding_kernel_launches(limit_value_t new_value) const
	{
		return set_limit(CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, new_value);
	}

	/**
	 * Have the calling thread wait - either busy-waiting or blocking - and
	 * return only after all pending actions within this context have concluded.
	 */
	void synchronize() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		auto status = cuCtxSynchronize();
		cuda::throw_if_error(status,
			"Failed synchronizing on event " + detail::ptr_as_hex(handle_));
	}

	stream_t default_stream() const noexcept;

protected: // constructors

	context_t(
			device::id_t   device_id,
			context::handle_t  context_id,
			bool           take_ownership) noexcept
	: device_id_(device_id), handle_(context_id), owning_(take_ownership) { }

public: // friendship

	friend context_t context::detail::wrap(
			device::id_t device_id,
			context::handle_t context_id,
			bool take_ownership) noexcept;

public: // constructors and destructor

	context_t(const context_t& other) :
		context_t(other.device_id_, other.handle_, false) { };

	context_t(context_t&& other) noexcept :
		context_t(other.device_id_, other.handle_, other.owning_)
	{
		other.owning_ = false;
	};

	~context_t() { destroy(); }

protected:
	void destroy()
	{
		if (owning_) { cuCtxDestroy(handle_); }
	}

public: // operators

	context_t& operator=(const context_t& other)
	{
		destroy();
		device_id_ = other.device_id_;
		handle_ = other.handle_;
		owning_ = false;
		return *this;
	}

	// Deleted since the id_t and handle_t are constant
	context_t& operator=(context_t&& other)
	{
		std::swap(device_id_, other.device_id_);
		std::swap(handle_, other.handle_);
		std::swap(owning_, other.owning_);
		return *this;
	}

protected: // data members
	device::id_t       device_id_;
	context::handle_t  handle_;
	bool               owning_;
		// this field is mutable only for enabling move construction; other
		// than in that case it must not be altered
};

inline bool operator==(const context_t& lhs, const context_t& rhs)
{
	return lhs.handle() == rhs.handle();
}

inline bool operator!=(const context_t& lhs, const context_t& rhs)
{
	return lhs.handle() != rhs.handle();
}

namespace context {

namespace detail {

/**
 * Obtain a wrapper for an already-existing CUDA context
 *
 * @note This is a named constructor idiom instead of direct access to the ctor of the same
 * signature, to emphase what this construction means - a new context is _not_
 * created.
 *
 * @param device_id Device with which the context is associated
 * @param context_id id of the context to wrap with a proxy
 * @param take_ownership when true, the wrapper will have the CUDA driver destroy
 * the cuntext when the wrapper itself destruct; otherwise, it is assumed
 * that the context is "owned" elsewhere in the code, and that location or entity
 * is responsible for destroying it when relevant (possibly after this wrapper
 * ceases to exist)
 * @return a context wrapper associated with the specified context
 */
inline context_t wrap(
	device::id_t       device_id,
	handle_t           context_id,
	bool               take_ownership) noexcept
{
	return context_t(device_id, context_id, take_ownership);
}

inline context_t from_handle(
	context::handle_t  context_handle,
	bool               take_ownership)
{
	device::id_t device_id = get_device_id(context_handle);
	return wrap(device_id, context_handle, take_ownership);
}

inline context_t create_and_push(
	device::id_t                           device_id,
	host_thread_synch_scheduling_policy_t  synch_scheduling_policy,
	bool                                   keep_larger_local_mem_after_resize,
	bool                                   allow_pinned_mapped_memory_allocation)
{
	auto flags = context::detail::make_flags(
			synch_scheduling_policy,
			keep_larger_local_mem_after_resize,
			allow_pinned_mapped_memory_allocation);
	context::handle_t new_context_id;
	auto status = cuCtxCreate(&new_context_id, flags, device_id);
	cuda::throw_if_error(
		status, "failed creating a CUDA context associated with device " + std::to_string(device_id));
	bool take_ownership = true;
	return context::detail::wrap(device_id, new_context_id, take_ownership);
}

} // namespace detail

/**
 * @brief creates a new execution stream on a device.
 *
 * @param device              The device on which to create the new stream
 * @param uses_blocking_sync  When synchronizing on this new event, shall a thread busy-wait for it, or block?
 * @param records_timing      Can this event be used to record time values (e.g. duration between events)?
 * @param interprocess        Can multiple processes work with the constructed event?
 * @return The constructed event proxy
 *
 * @note Creating an event
 */
inline context_t create(
	const device_t&                        device,
	host_thread_synch_scheduling_policy_t  synch_scheduling_policy = heuristic,
	bool                                   keep_larger_local_mem_after_resize = true,
	bool                                   allow_pinned_mapped_memory_allocation = false);

inline context_t create_and_push(
	const device_t&                        device,
	host_thread_synch_scheduling_policy_t  synch_scheduling_policy = heuristic,
	bool                                   keep_larger_local_mem_after_resize = true,
	bool                                   allow_pinned_mapped_memory_allocation = false);


namespace current {

/**
 * Determine whether any CUDA context is current, or whether the context stack is empty
 */
inline bool exists()
{
	return (detail::get_handle() != detail::no_current_context);
}

/**
 * Obtain the current CUDA context, if one exists.
 *
 * @throws std::runtime_error in case there is no current context
 */
inline context_t get()
{
	auto handle = detail::get_handle();
	if (handle == detail::no_current_context) {
		throw std::runtime_error("Attempt to obtain the current CUDA context when no context is current.");
	}
	return context::detail::from_handle(handle);
}

inline void set(const context_t& context)
{
	return detail::set(context.handle());
}

inline bool push_if_not_on_top(const context_t& context)
{
	return context::current::detail::push_if_not_on_top(context.handle());
}

inline void push(const context_t& context)
{
	return context::current::detail::push(context.handle());
}

inline context_t pop()
{
	constexpr const bool do_not_take_ownership { false };
	// Unfortunately, since we don't store the device IDs of contexts
	// on the stack, this incurs an extra API call beyond just the popping...
	auto handle = context::current::detail::pop();
	auto device_id = context::detail::get_device_id(handle);
	return context::detail::wrap(device_id, handle, do_not_take_ownership);
}

namespace detail {

/**
 * If now current context exists, push the current device's primary context onto the stack
 */
inline handle_t push_default_if_missing();

/**
 * Ensures that a current context exists by pushing the current device's primary context
 * if necessary, and returns the current context
 *
 * @throws std::runtime_error in case there is no current context
 */
inline context_t get_with_fallback_push()
{
	push_default_if_missing();
	return current::get();
}

} // namespace detail

} // namespace current

} // namespace context

} // namespace cuda

#endif // CUDA_API_WRAPPERS_CONTEXT_HPP_
