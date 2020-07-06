/**
 * @file multi_wrapper_impls.hpp
 *
 * @brief Implementations of methods or functions requiring the definitions of
 * multiple CUDA entity proxy classes. In some cases these are declared in the
 * individual proxy class files, with the other classes forward-declared.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_HPP_
#define MULTI_WRAPPER_IMPLS_HPP_

#include <cuda/api/array.hpp>
#include <cuda/api/device.hpp>
#include <cuda/api/event.hpp>
#include <cuda/api/kernel_launch.hpp>
#include <cuda/api/pointer.hpp>
#include <cuda/api/stream.hpp>
#include <cuda/api/unique_ptr.hpp>
#include <cuda/api/primary_context.hpp>
#include <cuda/api/apriori_compiled_kernel.hpp>
#include <cuda/api/module.hpp>
#include <cuda/api/virtual_memory.hpp>
#include <cuda/api/current_context.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

#include <type_traits>
#include <vector>
#include <algorithm>

namespace cuda {

namespace array {

namespace detail {

template<typename T>
inline cudaArray* allocate(device_t& device, array::dimensions_t<3> dimensions)
{
	device::current::detail::scoped_override_t set_device_for_this_scope(device.id());
	return allocate_on_current_device<T>(dimensions);
}

template<typename T>
inline cudaArray* allocate(device_t& device, array::dimensions_t<2> dimensions)
{
	device::current::detail::scoped_override_t set_device_for_this_scope(device.id());
	return allocate_on_current_device<T>(dimensions);
}

} // namespace detail

} // namespace array

namespace event {

inline event_t create(
	const context_t&  context,
	bool       uses_blocking_sync,
	bool       records_timing,
	bool       interprocess)
{
	// Yes, we need the ID explicitly even on the current device,
	// because event_t's don't have an implicit device ID.
	return event::detail::create(context.handle(), uses_blocking_sync, records_timing, interprocess);
}

inline event_t create(
	device_t&  device,
	bool       uses_blocking_sync,
	bool       records_timing,
	bool       interprocess)
{
	return create(device.primary_context(), uses_blocking_sync, records_timing, interprocess);
}

namespace ipc {

inline handle_t export_(event_t& event)
{
	return detail::export_(event.id());
}

inline event_t import(const context_t& context, const handle_t& event_ipc_handle)
{
	bool do_not_take_ownership { false };
	return event::detail::wrap(context.handle(), detail::import(event_ipc_handle), do_not_take_ownership);
}


inline event_t import(const device_t& device, const handle_t& event_ipc_handle)
{
	return import(device.primary_context(), event_ipc_handle);
}

} // namespace ipc

} // namespace event


// device_t methods

inline device::primary_context_t device_t::primary_context() const
{
	return device::primary_context::get(*this);
}

inline stream_t device_t::default_stream() const noexcept
{
	return stream::detail::wrap(id(), primary_context().handle(), stream::default_stream_id);
}

inline stream_t
device_t::create_stream(
	bool                will_synchronize_with_default_stream,
	stream::priority_t  priority)
{
	auto pc = primary_context();
	return stream::detail::create(id(), pc.handle(), will_synchronize_with_default_stream, priority);
}

inline bool context_t::is_primary() const
{
	return handle_ == device::primary_context::detail::get_id(device_id_);
}

inline device_t context_t::device() const
{
	return device::detail::wrap(device_id_);
}


inline stream_t context_t::default_stream() const noexcept
{
	return stream::detail::wrap(device_id_, handle_, stream::default_stream_id);
}

namespace device {
namespace primary_context {

inline void destroy(const device_t& device)
{
	auto status = cuDevicePrimaryCtxReset(device.id());
	throw_if_error(status, "Failed destroying/resetting the primary context of device " + std::to_string(device.id()));
}

inline primary_context_t get(const device_t& device)
{
	return detail::wrap(device.id(), detail::obtain_and_increase_refcount(device.id()));
}

} // namespace primary_context

namespace current {

inline scoped_override_t::scoped_override_t(const device_t& device) : parent(device.id()) { }
inline scoped_override_t::scoped_override_t(device_t&& device) : parent(device.id()) { }

} // namespace current
} // namespace device


namespace detail {

} // namespace detail

template <typename KernelFunction, typename ... KernelParameters>
void device_t::launch(
	bool thread_block_cooperativity,
	KernelFunction kernel_function, launch_configuration_t launch_configuration,
	KernelParameters ... parameters)
{
	return default_stream().enqueue.kernel_launch(
		thread_block_cooperativity, kernel_function, launch_configuration, parameters...);
}

inline event_t device_t::create_event(
	bool          uses_blocking_sync,
	bool          records_timing,
	bool          interprocess)
{
	// The current implementation of event::create is not super-smart,
	// but it's probably not worth it trying to improve just this function
	return event::create(*this, uses_blocking_sync, records_timing, interprocess);
}

// event_t methods


inline device::id_t event_t::device_id() const noexcept
{
	return context::detail::get_device_id(context_handle_);
}

inline device_t event_t::device() const noexcept
{
	return cuda::device::get(device_id());
}

inline context_t event_t::context() const noexcept
{
	constexpr const bool dont_take_ownership { false };
	return context::detail::wrap(device_id(), context_handle_, dont_take_ownership);
}



inline void event_t::record(const stream_t& stream)
{
	// Note:
	// TODO: Perhaps check the context match here, rather than have the Runtime API call fail?
	event::detail::enqueue(stream.id(), id_);
}

inline void event_t::fire(const stream_t& stream)
{
	record(stream);
	stream.synchronize();
}

// stream_t methods

inline device_t stream_t::device() const noexcept
{
	return cuda::device::detail::wrap(device_id_);
}

inline context_t stream_t::context() const noexcept
{
	constexpr const bool dont_take_ownership { false };
	return context::detail::wrap(device_id_, context_handle_, dont_take_ownership);
}

inline void stream_t::enqueue_t::wait(const event_t& event_)
{
	auto device_id = associated_stream.device_id_;
	device::current::detail::scoped_override_t set_device_for_this_context(device_id);

	// Required by the CUDA runtime API; the flags value is currently unused
	constexpr const unsigned int flags = 0;

	auto status = cudaStreamWaitEvent(associated_stream.id_, event_.id(), flags);
	throw_if_error(status,
		std::string("Failed scheduling a wait for event ") + cuda::detail::ptr_as_hex(event_.id())
		+ " on stream " + cuda::detail::ptr_as_hex(associated_stream.id_)
		+ " in context " + cuda::detail::ptr_as_hex(associated_stream.context_handle_)
		+ " on CUDA device " + std::to_string(device_id));

}

inline event_t& stream_t::enqueue_t::event(event_t& existing_event)
{
	auto device_id = associated_stream.device_id_;
	auto context_handle = associated_stream.context_handle_;
	auto stream_context_handle_ = associated_stream.context_handle_;
	if (existing_event.context_handle() != stream_context_handle_) {
		throw std::invalid_argument("Attempt to enqueue a CUDA event associated with context "
			+ detail::ptr_as_hex(existing_event.context_handle()) 
            + " on device " + std::to_string(existing_event.device_id())
			+ " to be triggered by a stream in CUDA context "
			+ detail::ptr_as_hex(associated_stream.context_handle_) 
	        + " on CUDA device " + std::to_string(device_id ) );
	}
	context::current::detail::scoped_override_t set_context_for_this_scope(context_handle);
	stream::detail::record_event_in_current_context(device_id, context_handle, associated_stream.id_, existing_event.id());
	return existing_event;
}

inline event_t stream_t::enqueue_t::event(
    bool          uses_blocking_sync,
    bool          records_timing,
    bool          interprocess)
{
	auto context_handle_ = associated_stream.context_handle_;
	context::current::detail::scoped_override_t set_context_for_this_scope(context_handle_);

	event_t ev { event::detail::create_in_current_context(context_handle_, uses_blocking_sync, records_timing, interprocess) };
	// Note that, at this point, the event is not associated with this enqueue object's stream.
	this->event(ev);
	return ev;
}

namespace memory {

template <typename T>
inline device_t pointer_t<T>::device() const noexcept
{
	return cuda::device::get(attributes().device);
}

namespace async {

inline void copy(void *destination, const void *source, size_t num_bytes, const stream_t& stream)
{
	detail::copy(destination, source, num_bytes, stream.id());
}

inline void copy(void* destination, const_region_t source, const stream_t& stream)
{
	copy(destination, source.start(), source.size(), stream);
}

inline void copy(region_t destination, const_region_t source, const stream_t& stream)
{
	copy(destination.start(), source, stream);
}

template <typename T, dimensionality_t NumDimensions>
inline void copy(array_t<T, NumDimensions>& destination, const T* source, const stream_t& stream)
{
	detail::copy(destination, source, stream.id());
}

template <typename T, dimensionality_t NumDimensions>
inline void copy(T* destination, const array_t<T, NumDimensions>& source, const stream_t& stream)
{
	detail::copy(destination, source, stream.id());
}

template <typename T>
inline void copy_single(T& destination, const T& source, const stream_t& stream)
{
	detail::copy_single(&destination, &source, sizeof(T), stream.id());
}

} // namespace async

namespace device {

inline region_t allocate(cuda::device_t device, size_t size_in_bytes)
{
	return detail::allocate(device.id(), size_in_bytes);
}

namespace async {

inline region_t allocate(const stream_t& stream, size_t size_in_bytes)
{
	return detail::allocate(stream.context().handle(), stream.id(), size_in_bytes);
}

inline void set(void* start, int byte_value, size_t num_bytes, const stream_t& stream)
{
	detail::set(start, byte_value, num_bytes, stream.id());
}

inline void zero(void* start, size_t num_bytes, const stream_t& stream)
{
	detail::zero(start, num_bytes, stream.id());
}

} // namespace async

namespace peer_to_peer {

inline void copy(
    void *             destination_address,
    const context_t&   destination_context,
    const void *       source_address,
    const context_t&   source_context,
    size_t             num_bytes)
{
    return detail::copy(
        destination_address, destination_context.handle(),
        source_address, source_context.handle(), num_bytes);
}

namespace async {

inline void copy(
	const stream_t&    stream,
    void *             destination_address,
    const context_t&   destination_context,
    const void *       source_address,
    const context_t&   source_context,
    size_t             num_bytes)
{
    return detail::copy(
        stream.id(), destination_address, destination_context.handle(), source_address,
        source_context.handle(), num_bytes);
}

} // namespace async

} // namespace peer_to_peer

/**
 * @brief Create a variant of std::unique_pointer for an array in
 * device-global memory
 *
 * @tparam T  an array type; _not_ the type of individual elements
 *
 * @param device        on which to construct the array of elements
 * @param num_elements  the number of elements to allocate
 * @return an std::unique_ptr pointing to the constructed T array
 */
template<typename T>
inline unique_ptr<T> make_unique(device_t device, size_t num_elements)
{
	static_assert(std::is_array<T>::value, "make_unique<T>(device, num_elements) can only be invoked for T being an array type, T = U[]");
	cuda::device::current::detail::scoped_override_t set_device_for_this_scope(device.id());
	return cuda::memory::detail::make_unique<T, detail::allocator, detail::deleter>(num_elements);
}

/**
 * @brief Create a variant of std::unique_pointer for a single value
 * in device-global memory
 *
 * @tparam T  the type of value to construct in device memory
 *
 * @param device  on which to construct the T element
 * @return an std::unique_ptr pointing to the allocated memory
 */
template <typename T>
inline unique_ptr<T> make_unique(device_t device)
{
	cuda::device::current::detail::scoped_override_t set_device_for_this_scope(device.id());
	return cuda::memory::detail::make_unique<T, detail::allocator, detail::deleter>();
}


} // namespace device

namespace managed {

inline device_t region_t::preferred_location() const
{
	auto device_id = detail::get_scalar_range_attribute<bool>(*this, cudaMemRangeAttributePreferredLocation);
	return cuda::device::get(device_id);
}

inline void region_t::set_preferred_location(device_t& device) const
{
	detail::set_scalar_range_attribute(*this, (cudaMemoryAdvise) cudaMemAdviseSetPreferredLocation, device.id());
}

inline void region_t::clear_preferred_location() const
{
	detail::set_scalar_range_attribute(*this, (cudaMemoryAdvise) cudaMemAdviseUnsetPreferredLocation);
}

inline void advise_expected_access_by(const_region_t region, device_t& device)
{
	detail::set_scalar_range_attribute(region, cudaMemAdviseSetAccessedBy, device.id());
}

inline void advise_no_access_expected_by(const_region_t region, device_t& device)
{
	detail::set_scalar_range_attribute(region, cudaMemAdviseUnsetAccessedBy, device.id());
}

template <typename Allocator>
std::vector<device_t, Allocator> accessors(const_region_t region, const Allocator& allocator)
{
	static_assert(sizeof(cuda::device::id_t) == sizeof(device_t), "Unexpected size difference between device IDs and their wrapper class, device_t");

	auto num_devices = cuda::device::count();
	std::vector<device_t, Allocator> devices(num_devices, allocator);
	auto device_ids = reinterpret_cast<cuda::device::id_t *>(devices.data());


	auto status = cudaMemRangeGetAttribute(
		device_ids, sizeof(device_t) * devices.size(),
		cudaMemRangeAttributeAccessedBy, region.start(), region.size());
	throw_if_error(status, "Obtaining the IDs of devices with access to the managed memory range at " + cuda::detail::ptr_as_hex(region.start()));
	auto first_invalid_element = std::lower_bound(device_ids, device_ids + num_devices, cudaInvalidDeviceId);
	// We may have gotten less results that the set of all devices, so let's whittle that down

	if (first_invalid_element - device_ids != num_devices) {
		devices.resize(first_invalid_element - device_ids);
	}

	return devices;
}

namespace async {

inline void prefetch(
	const_region_t   region,
	cuda::device_t   destination,
	const stream_t&  stream)
{
	detail::prefetch(region, destination.id(), stream.id());
}

} // namespace async


inline region_t allocate(
	cuda::device_t        device,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility)
{
	return detail::allocate(device.id(), num_bytes, initial_visibility);
}

} // namespace managed

namespace mapped {

inline region_pair allocate(
	cuda::device_t&     device,
	size_t              size_in_bytes,
	allocation_options  options)
{
	return cuda::memory::mapped::detail::allocate(device.id(), size_in_bytes, options);
}

} // namespace mapped

} // namespace memory

// kernel_t methods

inline const context_t kernel_t::context() const noexcept
{
	constexpr bool dont_take_ownership { false };
	return context::detail::from_handle(context_handle_, dont_take_ownership);
}

inline const device_t kernel_t::device() const noexcept
{
	return device::detail::wrap(context::detail::get_device_id(context_handle_));
}

#if defined(__CUDACC__)
// Unfortunately, the CUDA runtime API does not allow for computation of the grid parameters for maximum occupancy
// from code compiled with a host-side-only compiler! See cuda_runtime.h for details


inline std::pair<grid::dimension_t, grid::block_dimension_t>
apriori_compiled_kernel_t::min_grid_params_for_max_occupancy(
	memory::shared::size_t   dynamic_shared_memory_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
#if CUDART_VERSION <= 10000
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#else
	int min_grid_size_in_blocks, block_size;
	auto result = cudaOccupancyMaxPotentialBlockSizeWithFlags(
		&min_grid_size_in_blocks, &block_size,
		ptr_,
		static_cast<std::size_t>(dynamic_shared_memory_size),
		static_cast<int>(block_size_limit),
		disable_caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault
		);
	throw_if_error(result,
		"Failed obtaining parameters for a minimum-size grid for kernel " + detail::ptr_as_hex(ptr_) +
		" in context " + detail::ptr_as_hex(context_handle_) + ".");
	return { min_grid_size_in_blocks, block_size };
#endif
}

template <typename UnaryFunction>
std::pair<grid::dimension_t, grid::block_dimension_t>
apriori_compiled_kernel_t::min_grid_params_for_max_occupancy(
	UnaryFunction            block_size_to_dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
#if CUDART_VERSION <= 10000
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#else
	int min_grid_size_in_blocks, block_size;
	auto result = cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
		&min_grid_size_in_blocks, &block_size,
		ptr_,
		block_size_to_dynamic_shared_mem_size,
		static_cast<int>(block_size_limit),
		disable_caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault
		);
	throw_if_error(result,
		"Failed obtaining parameters for a minimum-size grid for kernel " + detail::ptr_as_hex(ptr_) +
		" in context " + detail::ptr_as_hex(context_handle_) + ".");
	return { min_grid_size_in_blocks, block_size };
#endif
}
#endif

inline void apriori_compiled_kernel_t::set_preferred_shared_mem_fraction(unsigned shared_mem_percentage)
{
	if (shared_mem_percentage > 100) {
		throw std::invalid_argument("Percentage value can't exceed 100");
	}
	context::current::detail::scoped_override_t set_context_for_this_scope(context_handle_);
#if CUDART_VERSION >= 9000
	auto result = cudaFuncSetAttribute(ptr_, cudaFuncAttributePreferredSharedMemoryCarveout, shared_mem_percentage);
	throw_if_error(result, "Trying to set the carve-out of shared memory/L1 cache memory");
#else
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#endif
}

namespace stream {

namespace detail {

inline device::id_t device_id_of(stream::id_t stream_id)
{
	return context::detail::get_device_id(context_handle_of(stream_id));
}

inline void record_event_in_current_context(
	device::id_t       current_device_id,
	context::handle_t  current_context_handle_,
	stream::id_t       stream_id,
	event::id_t        event_id)
{
	auto status = cuEventRecord(event_id, stream_id);
	throw_if_error(status,
		   "Failed scheduling event " + cuda::detail::ptr_as_hex(event_id) + " to occur"
		   + " on stream " + cuda::detail::ptr_as_hex(stream_id)
		   + " in CUDA context " + cuda::detail::ptr_as_hex(current_context_handle_)
		   + " on CUDA device" + std::to_string(current_device_id));
}

} // namespace detail

inline stream_t create(
	context_t    context,
	bool         synchronizes_with_default_stream,
	priority_t   priority)
{
	return detail::create(context.device_id(), context.handle(), synchronizes_with_default_stream, priority);
}

inline stream_t create(
	device_t     device,
	bool         synchronizes_with_default_stream,
	priority_t   priority)
{
	return detail::create(device.id(), device.primary_context().handle(), synchronizes_with_default_stream, priority);
}

} // namespace stream

template<typename RawKernel, typename... KernelParameters>
void enqueue_launch(
	bool                    thread_block_cooperation,
	RawKernel               kernel_function,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	static_assert(std::is_function<RawKernel>::value or detail::is_function_ptr<RawKernel>::value,
		"Only a bona fide function can be a CUDA kernel and be launched with this function; "
		"you were attempting to enqueue a launch of something other than a function");
	static_assert(
		cuda::detail::all_true<
			std::is_trivially_copyable<typename std::decay<KernelParameters>::type>::value...
		>::value,
		"All kernel parameter types must be of a trivially copyable (decayed) type." );
	detail::enqueue_launch(
		thread_block_cooperation,
		kernel_function,
		stream.id(),
		launch_configuration,
		std::forward<KernelParameters>(parameters)...);
}

namespace detail {

template<typename... KernelParameters>
void enqueue_launch_helper<apriori_compiled_kernel_t, KernelParameters...>::operator()(
		apriori_compiled_kernel_t  wrapped_kernel,
		const stream_t &           stream,
		launch_configuration_t     launch_configuration,
		KernelParameters &&...     parameters)
{
	using raw_kernel_t = typename kernel::detail::raw_kernel_typegen<KernelParameters ...>::type;
	auto unwrapped_kernel_function = reinterpret_cast<raw_kernel_t>(const_cast<void *>(wrapped_kernel.ptr()));
	// Notes:
	// 1. The inner cast here is because we store the pointer as const void* - as an extra
	//    precaution against anybody trying to write through it. Now, function pointers
	//    can't get written through, but are still for some reason not considered const.
	// 2. We rely on the caller providing us with more-or-less the correct parameters -
	//    corresponding to the compiled kernel function's. I say "more or less" because the
	//    `KernelParameter` pack may contain some references, arrays and so on - which CUDA
	//    kernels cannot accept; so we massage those a bit.

	detail::enqueue_launch(
		wrapped_kernel.thread_block_cooperation(),
		unwrapped_kernel_function,
		stream.id(),
		launch_configuration,
		std::forward<KernelParameters>(parameters)...);
}

template<typename Kernel, typename... KernelParameters>
void enqueue_launch_helper<Kernel, KernelParameters...>::operator()(
	Kernel                  kernel_function,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters &&...  parameters)
{
	static_assert(std::is_function<Kernel>::value or (is_function_ptr<Kernel>::value),
		"No appropriate specialization exists for the type of object passed as the kernel.");
	detail::enqueue_launch(
		thread_blocks_may_not_cooperate,
		kernel_function,
		stream.id(),
		launch_configuration,
		std::forward<KernelParameters>(parameters)...
	);
}

} // namespace detail

namespace memory {
namespace virtual_ {
namespace allocation {

inline device_t properties_t::device() const
{
	return cuda::device::detail::wrap(raw.location.id);
}

} // namespace allocation

inline void set_access_mode(
	region_t fully_mapped_region,
	access_mode_t access_mode,
	const device_t& device)
{
	CUmemAccessDesc desc { { CU_MEM_LOCATION_TYPE_DEVICE, device.id() }, CUmemAccess_flags(access_mode) };
	constexpr const size_t count { 1 };
	auto result = cuMemSetAccess(fully_mapped_region.device_ptr(), fully_mapped_region.size(), &desc, count);
	throw_if_error(result, "Failed setting the access mode to the virtual memory mapping to the range of size "
		+ std::to_string(fully_mapped_region.size()) + " bytes at " + cuda::detail::ptr_as_hex(fully_mapped_region.data()));
}

inline void mapping_t::set_access_mode(access_mode_t access_mode, const device_t& device) const
{
	virtual_::set_access_mode(region_, access_mode, device);
}

inline access_mode_t mapping_t::get_access_mode(const device_t& device) const
{
	CUmemLocation_st location { CU_MEM_LOCATION_TYPE_DEVICE, device.id() };
	unsigned long long flags;
	auto result = cuMemGetAccess(&flags, &location, region_.device_ptr() );
	throw_if_error(result, "Failed determining the access mode for device " + std::to_string(device.id())
						   + " to the virtual memory mapping to the range of size "
						   + std::to_string(region_.size()) + " bytes at " + cuda::detail::ptr_as_hex(region_.data()));
	return access_mode_t(flags);
}

} // namespace virtual_
} // namespace memory

namespace context {

namespace current {

namespace detail {

inline handle_t push_default_if_missing()
{
	auto handle = detail::get_handle();
	if (handle != detail::no_current_context) {
		return handle;
	}
	auto current_device_id = device::current::detail::get_id();
	auto pc_handle = device::primary_context::detail::obtain_and_increase_refcount(current_device_id);
	push(pc_handle);
	return pc_handle;
}

/**
 * @note This specialized scope setter is used in API calls which aren't provided a context
 * as a parameter, and when there is no context that's current. Such API calls are necessarily
 * device-related (i.e. runtime-API-ish), and since there is always a current device, we can
 * (and in fact must) fall back on that device's primary context as what the user assumes we
 * would use.
 */
class scoped_current_device_fallback_t : scoped_override_t {
protected:
public:
	explicit scoped_current_device_fallback_t() :
		scoped_override_t(
			device::primary_context::detail::obtain_and_increase_refcount(device::current::detail::get_id())
		) { }
	~scoped_current_device_fallback_t() = default;
};


} // namespace detail

inline scoped_override_t::scoped_override_t(const context_t& context) : parent(context.handle()) { }
inline scoped_override_t::scoped_override_t(context_t&& context) : parent(context.handle()) { }

} // namespace current

inline context_t create_and_push(
	const device_t&                        device,
	host_thread_synch_scheduling_policy_t  synch_scheduling_policy,
	bool                                   keep_larger_local_mem_after_resize,
	bool                                   allow_pinned_mapped_memory_allocation)
{
	return detail::create_and_push(device.id(), synch_scheduling_policy, keep_larger_local_mem_after_resize, allow_pinned_mapped_memory_allocation);
}

inline context_t create(
	const device_t&                        device,
	host_thread_synch_scheduling_policy_t  synch_scheduling_policy,
	bool                                   keep_larger_local_mem_after_resize,
	bool                                   allow_pinned_mapped_memory_allocation)
{
	auto created = detail::create_and_push(device.id(), synch_scheduling_policy, keep_larger_local_mem_after_resize, allow_pinned_mapped_memory_allocation);
	current::pop();
	return created;
}

} // namespace context

template<typename Kernel, typename... KernelParameters>
inline void launch(
	Kernel                  kernel_function,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	if (context::current::exists()) {
		stream_t stream = context::current::get().default_stream();
		enqueue_launch(kernel_function, stream, launch_configuration, std::forward<KernelParameters>(parameters)...);
	}
	else {
		context::current::detail::scoped_current_device_fallback_t context_for_current_scope;
		stream_t stream = context::current::get().default_stream();
		enqueue_launch(kernel_function, stream, launch_configuration, std::forward<KernelParameters>(parameters)...);
	}
}

namespace memory {

template <typename T>
T* pointer_t<T>::get_for_device() const
{
	context::current::detail::scoped_current_device_fallback_t ensure_we_have_some_context;
	CUdeviceptr device_side_address;
	auto status = cuPointerGetAttribute (&device_side_address, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, memory::device::address(ptr_));
	throw_if_error(status, "Failed obtaining the device-side pointer for pointer value " + cuda::detail::ptr_as_hex(ptr_));
	return reinterpret_cast<T*>(device_side_address);
}

template <typename T>
T* pointer_t<T>::get_for_host() const
{
	context::current::detail::scoped_current_device_fallback_t ensure_we_have_some_context;
	CUdeviceptr host_side_address;
	auto status = cuPointerGetAttribute (&host_side_address, CU_POINTER_ATTRIBUTE_HOST_POINTER, memory::device::address(ptr_));
	throw_if_error(status, "Failed obtaining the host-side pointer for pointer value " + cuda::detail::ptr_as_hex(ptr_));
	return reinterpret_cast<T*>(host_side_address);
}

} // namespace memory

// module_t methods

inline context_t module_t::context() const { return context::detail::from_handle(context_handle_); }
inline device_t module_t::device() const { return device::get(context::detail::get_device_id(context_handle_)); }

} // namespace cuda


#endif // MULTI_WRAPPER_IMPLS_HPP_

