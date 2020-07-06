/**
 * @file texture_view.hpp
 *
 * @brief Contains a "texture view" class, for hardware-accelerated
 * access to CUDA arrays, and some related standalone functions and
 * definitions.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP
#define CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP

#include <cuda_runtime.h>
#include <cuda/api/array.hpp>
#include <cuda/api/error.hpp>

namespace cuda {

class texture_t;

namespace texture {

using handle_t = cudaTextureObject_t;

/**
 * A simplifying rudimentary wrapper wrapper for the CUDA runtime API's internal
 * "texture descriptor" object, allowing the creating of such descriptors without
 * having to give it too much thought.
 *
 * @todo Could be expanded into a richer wrapper class allowing actual settings
 * of the various fields.
 */
struct descriptor_t : public cudaTextureDesc {
	inline descriptor_t()
	{
		memset(static_cast<cudaTextureDesc*>(this), 0, sizeof(cudaTextureDesc));
		this->addressMode[0] = cudaAddressModeBorder;
		this->addressMode[1] = cudaAddressModeBorder;
		this->addressMode[2] = cudaAddressModeBorder;
		this->filterMode = cudaFilterModePoint;
		this->readMode = cudaReadModeElementType;
		this->normalizedCoords = 0;
	}
};

namespace detail {

inline texture_t wrap(texture::handle_t handle, bool take_ownership) noexcept;

}  // namespace detail

}  // namespace texture

/**
 * @brief Use texture memory for optimized read only cache access
 *
 * This represents a view on the memory owned by a CUDA array. Thus you can
 * first create a CUDA array (\ref cuda::array_t) and subsequently
 * create a `texture_view` from it. In CUDA kernels elements of the array
 * can be accessed with e.g. `float val = tex3D<float>(tex_obj, x, y, z);`,
 * where `tex_obj` can be obtained by the member function `get()` of this
 * class.
 *
 * See also the following sections in the CUDA programming guide:
 *
 * - <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory">texturre and surface memory</a>
 * - <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching">texture fetching</a>
 *
 * @note texture_view's are essentially _owning_ - the view is a resource the CUDA
 * runtime creates for you, which then needs to be freed.
 */
class texture_t {

public:
	bool is_owning() const noexcept { return owning_; }
	texture::handle_t handle() const noexcept { return handle_; }

public: // constructors and destructors

	texture_t(const texture_t& other)
	: handle_(other.handle_), owning_(false) { }

	texture_t(texture_t&& other) noexcept :
		handle_(other.handle_), owning_(other.handle_)
	{
		other.owning_ = false;
	};

public: // operators

	~texture_t()
	{
		if (owning_) {
			auto status = cudaDestroyTextureObject(handle_);
			throw_if_error(status, "failed destroying texture object");
		}
	}

	texture_t& operator=(const texture_t& other) = delete;
	texture_t& operator=(texture_t& other) = delete;

protected: // constructor

	// Usable by the wrap function
	texture_t(texture::handle_t handle , bool take_ownership) noexcept
	: handle_(handle), owning_(take_ownership) { }

public: // friendship

	friend texture_t texture::detail::wrap(texture::handle_t handle, bool take_ownersip) noexcept;

protected:
	texture::handle_t handle_ { } ;
	bool owning_;
};


inline bool operator==(const texture_t& lhs, const texture_t& rhs) noexcept
{
	return lhs.handle() == rhs.handle();
}

inline bool operator!=(const texture_t& lhs, const texture_t& rhs) noexcept
{
	return not (lhs.handle() == rhs.handle());
}

namespace texture {
namespace detail {

inline texture_t wrap(texture::handle_t handle, bool take_ownership) noexcept
{
	return texture_t(handle, take_ownership);
}

} // namespace detail

template <typename T, dimensionality_t NumDimensions>
texture_t texture_view(
	const cuda::array_t<T, NumDimensions>& arr,
	texture::descriptor_t descriptor = texture::descriptor_t())
{
	cudaResourceDesc resource_descriptor;
	std::memset(&resource_descriptor, 0, sizeof(resource_descriptor));
	resource_descriptor.resType = cudaResourceTypeArray;
	resource_descriptor.res.array.array = arr.get();

	handle_t handle;

	const cudaResourceViewDesc* null_resource_view_descriptor { nullptr };
	auto status = cudaCreateTextureObject(&handle, &resource_descriptor, &descriptor, null_resource_view_descriptor);
	throw_if_error(status, "failed creating a CUDA texture object");
	bool is_owning { true };
	return detail::wrap(handle, is_owning);
}

} // namespace texture

}  // namespace cuda

#endif  // CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP
