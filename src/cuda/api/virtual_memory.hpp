/**
 * @file virtual_memory.hpp
 */
#ifndef CUDA_API_WRAPPERS_VIRTUAL_MEMORY_HPP
#define CUDA_API_WRAPPERS_VIRTUAL_MEMORY_HPP

#include <cuda/api/types.hpp>
#include <cuda/api/memory.hpp>
#include <cuda.h>

namespace cuda {
// TODO: Perhaps move this down into the device namespace ?
namespace memory {
namespace virtual_ {

class reservation_t;
class allocation_t;
class mapping_t;

namespace detail {

inline void cancel_reservation(memory::region_t reserved) {
    auto status = cuMemAddressFree(memory::device::address(reserved.start()), reserved.size());
    throw_if_error(status, "Failed feeing a reservation of memory region at "
                           + cuda::detail::ptr_as_hex(reserved.data()) + " of size " + std::to_string(reserved.size()));

}

} // namespace detail

using alignment_t = size_t;

enum : alignment_t { trivial_alignment = size_t{1} };

reservation_t reserve(region_t region, alignment_t alignment = trivial_alignment);

class reservation_t {
protected:

    reservation_t(region_t region, alignment_t alignment, bool owning)
        : region_(region), alignment_(alignment), owning_(owning) { }

public:
    friend reservation_t reserve(region_t region, alignment_t alignment);

    reservation_t(reservation_t&& other)  : region_(other.region_), alignment_(other.alignment_), owning_(other.owning_)
    {
        other.owning_ = false;
    }

    ~reservation_t() {
        detail::cancel_reservation(region_);
    }

public: // getters
    bool is_owning() const { return owning_; }
    region_t region() const { return region_; }
    alignment_t alignment() const { return alignment_; }

protected: // data members
    const region_t     region_;
    const alignment_t  alignment_;
    bool               owning_;
};

inline reservation_t reserve(region_t region, alignment_t alignment)
{
    unsigned long flags { 0 };
    CUdeviceptr ptr;
    auto status = cuMemAddressReserve(
    	&ptr, region.size(), alignment,region.device_ptr(), flags);
    throw_if_error(status,
        "Failed feeing a reservation of memory region at "
        + cuda::detail::ptr_as_hex(region.data()) + " of size " + std::to_string(region.size()));
    // TODO: This is strange. If we've specified the memory range already, why even
    // provide addr alignment for that matter?
    if (ptr != region.device_ptr()) {
        throw runtime_error(cuda::status::invalid_value,
            "Reservation result address different from original address"); }
    bool is_owning { true };
    return {region, alignment, is_owning };
}

namespace allocation {

enum class kind_t {
    posix_file_descriptor = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    win32_handle = CU_MEM_HANDLE_TYPE_WIN32,
    win32_kmt = CU_MEM_HANDLE_TYPE_WIN32_KMT,
};

enum class granularity_kind_t : std::underlying_type<CUmemAllocationGranularity_flags_enum>::type {
	minimum_required     = CU_MEM_ALLOC_GRANULARITY_MINIMUM,
	recommended_for_performance = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
};

namespace detail {
template<kind_t Kind> struct shared_handle_type_helper;

template <> struct shared_handle_type_helper<kind_t::posix_file_descriptor> { using type = int; };
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
template <> struct shared_handle_type_helper<kind_t::win32_handle> { using type = HANDLE; };
#endif
// TODO: What about WIN32_KMT?
} // namespace detail

template<kind_t Kind>
using shared_handle_t = typename detail::shared_handle_type_helper<Kind>::type;

// Note: Not inheriting from CUmemAllocationProp_st, since
// that structure is a bit messed up
struct properties_t {
    // Note: Specifying a compression type is currently unsupported,
    // as the driver API does not document semantics for the relevant
    // properties field

public: // getters
    cuda::device_t device() const;

	// TODO: Is this only relevant to requests?
    allocation::kind_t requested_kind() const
    {
		return kind_t(raw.requestedHandleTypes);
    };

protected:
    size_t granularity(granularity_kind_t granuality_kind) const {
        size_t result;
        auto status = cuMemGetAllocationGranularity(&result, &raw,
        	static_cast<CUmemAllocationGranularity_flags>(granuality_kind));
        throw_if_error(status, "Could not determine allocation granularity");
        return result;
    }

public:
	properties_t(CUmemAllocationProp_st raw_properties) : raw(raw_properties)
	{
		if (raw.location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
			throw std::runtime_error("Unexpected allocation type - we only know about devices!");
		}
	}

	properties_t(properties_t&&) = default;
	properties_t(const properties_t&) = default;

public:
	CUmemAllocationProp_st raw;

};

using handle_t = CUmemGenericAllocationHandle;

namespace detail {

allocation_t wrap(handle_t handle, size_t size, bool owning);

} // namespace detail

} // namespace allocation

class allocation_t {
protected: // constructors
    allocation_t(allocation::handle_t handle, size_t size, bool owning)
    : handle_(handle), size_(size), owning_(owning) { }

public: // constructors & destructor
    allocation_t(const allocation_t& other)  : handle_(other.handle_), size_(other.size_), owning_(false)
    { }

    allocation_t(allocation_t&& other)  : handle_(other.handle_), size_(other.size_), owning_(other.owning_)
    {
        other.owning_ = false;
    }

    ~allocation_t() {
        if (not owning_) { return; }
        auto result = cuMemRelease(handle_);
        throw_if_error(result, "Failed making a virtual memory allocation of size " + std::to_string(size_));
    }

public: // non-mutators
    friend allocation_t allocation::detail::wrap(allocation::handle_t handle, size_t size, bool owning);

    size_t size() const { return size_; }
    allocation::handle_t handle() const { return handle_; }
    bool is_owning() const { return owning_; }

    allocation::properties_t properties() const {
		CUmemAllocationProp raw_properties;
        auto status = cuMemGetAllocationPropertiesFromHandle(&raw_properties, handle_);
        throw_if_error(status, "Obtaining the properties of a virtual memory allocation with handle " + std::to_string(handle_));
        return { raw_properties };
    }

    template <allocation::kind_t SharedHandleKind>
    allocation::shared_handle_t<SharedHandleKind> sharing_handle()
    {
        allocation::shared_handle_t<SharedHandleKind> shared_handle_;
        constexpr const unsigned long long flags { 0 };
        auto result = cuMemExportToShareableHandle(&shared_handle_, handle_, SharedHandleKind, flags);
        throw_if_error(result, "Exporting a (generic CUDA) shared memory allocation to a shared handle");
        return shared_handle_;
    }

protected: // data members
    const allocation::handle_t handle_;
    size_t size_;
    bool owning_;
};

namespace allocation {

inline allocation_t create(size_t size, properties_t properties)
{
    constexpr const unsigned long long flags { 0 };
    CUmemGenericAllocationHandle handle;
    auto result = cuMemCreate(&handle, size, &properties.raw, flags);
    throw_if_error(result, "Failed making a virtual memory allocation of size " + std::to_string(size));
    constexpr const bool is_owning { true };
    return detail::wrap(handle, size, is_owning);
}

namespace detail {

inline allocation_t wrap(handle_t handle, size_t size, bool take_ownership)
{
	return {handle, size, take_ownership};
}

inline properties_t properties_of(handle_t handle)
{
    CUmemAllocationProp prop;
    auto result = cuMemGetAllocationPropertiesFromHandle (&prop, handle);
    throw_if_error(result, "Failed obtaining the properties of the virtual memory allocation with handle "
      + std::to_string(handle));
    return { prop };
}

} // namespace detail

/**
 *
 * @note Unfortunately, importing a handle does not tell you how much memory is allocated
 *
 * @tparam SharedHandleKind In practice, a to choose between operating systems, as different
 * OSes would use different kinds of shared handles.
 * @param shared_handle a handle obtained from another process, where it had been
 * exported from a CUDA-specific allocation handle.
 *
 * @return the
 */
template <allocation::kind_t SharedHandleKind>
allocation_t import(shared_handle_t<SharedHandleKind> shared_handle, size_t size, bool take_ownership = false)
{
    constexpr const unsigned long long flags { 0 };
    handle_t result_handle;
    auto result = cuMemImportFromShareableHandle(
        &result_handle, reinterpret_cast<void*>(shared_handle), CUmemAllocationHandleType(SharedHandleKind));
    throw_if_error(result, "Failed importing a virtual memory allocation from a shared handle ");
    return allocation::detail::wrap(result_handle, size, take_ownership);
}

} // namespace allocation

enum access_mode_t : std::underlying_type<CUmemAccess_flags>::type {
    no_access             = CU_MEM_ACCESS_FLAGS_PROT_NONE,
    read_access           = CU_MEM_ACCESS_FLAGS_PROT_READ,
    read_and_write_access = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    rw_access             = read_and_write_access
};

namespace mapping {
namespace detail {

inline mapping_t wrap(region_t region, bool owning = false);

} // namespace detail
} // namespace mapping

class mapping_t {
protected:  // constructors
    mapping_t(region_t region, bool owning) : region_(region), owning_(owning) { }

public: // constructors & destructions

    friend mapping_t mapping::detail::wrap(region_t region, bool owning);

    mapping_t(const mapping_t& other) :
        region_(other.range()), owning_(false) { }

    mapping_t(mapping_t&& other) :
        region_(other.range()), owning_(other.owning_)
    {
        other.owning_ = false;
    }

    region_t range() const { return region_; }
    bool is_owning() const { return owning_; }

	void set_access_mode(access_mode_t access_mode, const device_t& device) const;
	access_mode_t  get_access_mode(const device_t& device) const;

    ~mapping_t()
    {
        if (not owning_) { return; }
        auto result = cuMemUnmap(region_.device_ptr(), region_.size());
        throw_if_error(result,
        	"Failed unmapping the virtual memory range mapped to "
        	+ cuda::detail::ptr_as_hex(region_.data()) + " of size " + std::to_string(region_.size()) + " bytes");
    }

public:
#if CUDA_VERSION >= 11000
	allocation_t allocation() const
    {
		CUmemGenericAllocationHandle allocation_handle;
        auto status = cuMemRetainAllocationHandle(&allocation_handle, region_.data());
        throw_if_error(status," Failed obtaining/retaining the allocation handle for the virtual memory range mapped to "
        	+ cuda::detail::ptr_as_hex(region_.data()) + " of size " + std::to_string(region_.size()) + " bytes");
        constexpr const bool dont_take_ownership { false };
        return allocation::detail::wrap(allocation_handle, region_.size(), dont_take_ownership);
    }
#endif
protected:

    region_t region_;
    bool owning_;

};

namespace mapping {
namespace detail {

mapping_t wrap(region_t range, bool owning) {
	return {range, owning};
}

} // namespace detail

} // namespace mapping

inline mapping_t map(region_t region, const allocation_t& allocation)
{
    size_t offset_into_allocation { 0 }; // not yet supported, but in the API
    constexpr const unsigned long long flags { 0 };
    auto handle = allocation.handle();
    auto status = cuMemMap(region.device_ptr(), region.size(), offset_into_allocation, handle, flags);
    throw_if_error(status, "Failed mapping the virtual memory allocation " + std::to_string(handle)
        + " to the range of size " + std::to_string(region.size()) + " bytes at " +
        cuda::detail::ptr_as_hex(region.data()));
    constexpr const bool is_owning { true };
    return mapping::detail::wrap(region, is_owning);
}

/**
 * Set the access mode from a single device to a region in the (universal) address space
 *
 * @param fully_mapped_region a region in the universal (virtual) address space, which must be
 * covered entirely by virtual memory mappings
 */
void set_access_mode(region_t fully_mapped_region, access_mode_t access_mode, const device_t& device);


} // namespace virtual_
} // namespace memory
} // namespace cuda

#endif //CUDA_API_WRAPPERS_VIRTUAL_MEMORY_HPP
