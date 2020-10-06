/**
 * @file nvrtc/program.hpp
 *
 * @brief A Wrapper class for runtime-compiled (RTC) programs, manipulated using the NVRTC library.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVRTC_PROGRAM_HPP_
#define CUDA_API_WRAPPERS_NVRTC_PROGRAM_HPP_

#include <cuda/nvrtc/compilation_options.hpp>
#include <cuda/nvrtc/error.hpp>
#include <cuda/nvrtc/types.hpp>
#include <cuda/api.hpp>

#include <vector>
#include <iostream>

namespace cuda {

///@cond
class device_t;
class context_t;
class program_t;
///@endcond

/**
 * @brief Real-time compilation of CUDA programs using the NVIDIA NVRTC library.
 */
namespace rtc {

namespace program {

using handle_t = nvrtcProgram;

} // namespace program

/**
 * Wrapper class for a CUDA code module
 *
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to the module is a const-respecting operation on this class.
 */
class program_t {

public: // getters

	const std::string& name() const { return name_; }
	program::handle_t handle() const { return handle_; }

public: // non-mutators

	// Unfortunately, C++'s standard string class is very inflexible,
	// and it is not possible for us to get it to have an appropriately-
	// sized _uninitialized_ buffer. We will therefore have to use
	// a clunkier return type.
	//
	// std::string log() const

	static_vector<char> log() const
	{
		size_t size;
		auto status = nvrtcGetProgramLogSize(handle_, &size);
		throw_if_error(status, "Failed obtaining NVRTC program log size");
		static_vector<char> result(size);
			// TODO: Either use a proper static-vector class with no initialization,
			// or otherwise ensure
		status = nvrtcGetProgramLog(handle_, result.data());
		return result;
	}

	static_vector<char> ptx() const
	{
		size_t size;
		auto status = nvrtcGetPTXSize(handle_, &size);
		throw_if_error(status, "Failed obtaining NVRTC program log size");
		static_vector<char> result(size);
			// This incurs the penalty of zero-filling the buffer; but that's
			// the price of returning an std::string _and_
		status = nvrtcGetPTX(handle_, result.data());
		return result;
	}

	/**
	 * Obtain the mangled/lowered form of an expression registered earlier, after
	 * the compilation
	 *
	 * @param unmangled_name A name of a __global__ or __device__ function or variable.
	 * @return The mangled name (which can actually be used for invoking kernels,
	 * moving data etc.). The memory is owned by the NVRTC program and will be
	 * released when it is destroyed.
	 */
	const char* mangled_form_of(const char* unmangled_name_expression)
	{
		const char* result;
		auto status = nvrtcGetLoweredName(handle_, unmangled_name_expression, &result);
		throw_if_error(status, std::string("Failed obtaining the mangled form of name ")
			+ unmangled_name_expression + "\" in PTX program \"" + name_ + '\"');
		return result;
	}

public: // mutators
	void compile(span<const char*> options)
	{
		auto status = nvrtcCompileProgram(handle_, options.size(), options.data());
		throw_if_error(status, "Failed compiling program \"" + name_ + "\"");
	}

	void compile(const compilation_options_t& options)
	{
		auto marshalled_options = options.marshal();
		compile(marshalled_options.option_ptrs());
	}

	void compile()
	{
		// TODO: Perhaps do something smarter here, e.g. figure out the appropriate compute capabilities?
		compile(compilation_options_t{});
	}

	/**
	 * @brief Register a pre-mangled name of a kernel, to make available for use
	 * after compilation
	 *
	 * @param name The text of an expression, e.g. "my_global_func()", "&f1", "N1::N2::n2",
	 *
	 */
	void register_name_for_lookup(const char* unmangled_name_expression)
	{
		auto status = nvrtcAddNameExpression(handle_, unmangled_name_expression);
		throw_if_error(status, "Failed registering a kernel name with program \"" + name_ + "\"");
	}

protected: // constructors
	program_t(
		program::handle_t handle,
		const char* name,
		bool owning = false) : handle_(handle), name_(name), owning_(owning) { }

public: // constructors and destructor

	program_t(
		const char*  program_name,
		const char*  cuda_source,
		size_t       num_headers,
		const char** header_names,
		const char** header_sources
		) : handle_(), name_(program_name), owning_(true)
	{
		status_t status;
		status = nvrtcCreateProgram(&handle_, cuda_source, program_name, num_headers, header_sources, header_names);
		throw_if_error(status, "Failed creating an NVRTC program (named " + std::string(name_) + ')');
	}

	program_t(const program_t&) = delete;

	program_t(program_t&& other) noexcept
		: handle_(other.handle_), name_(other.name_), owning_(other.owning_)
	{
		other.owning_ = false;
	};

	~program_t()
	{
		if (owning_) {
			auto status = nvrtcDestroyProgram(&handle_);
			throw_if_error(status, "Destroying an NVRTC program");
		}
	}

public: // operators

	program_t& operator=(const program_t& other) = delete;
	program_t& operator=(program_t&& other) = delete;

protected: // data members
	program::handle_t  handle_;
	std::string        name_;
	bool               owning_;
}; // class program_t

namespace program {

// template <>
inline program_t create(
	const char* program_name,
	const char* cuda_source,
	size_t num_headers,
	const char** header_names,
	const char** header_sources)
{
	return program_t(program_name, cuda_source, num_headers, header_names, header_sources);

}


template <typename HeaderNamesFwdIter, typename HeaderSourcesFwdIter>
inline program_t create(
	const char* program_name,
	const char* cuda_source,
	HeaderNamesFwdIter header_names_start,
	HeaderNamesFwdIter header_names_end,
	HeaderSourcesFwdIter header_sources_start)
{
	auto num_headers = header_names_end - header_names_start;
	std::vector<const char*> header_names;
	header_names.reserve(num_headers);
	std::copy_n(header_names_start, num_headers, std::back_inserter(header_names));
	std::vector<const char*> header_sources;
	header_names.reserve(num_headers);
	std::copy_n(header_sources_start, num_headers, std::back_inserter(header_sources));
	return program_t(cuda_source, program_name, num_headers, header_names.data(), header_sources.data());
}

inline program_t create(
	const char* program_name,
	const char* cuda_source,
	span<const char*> header_names,
	span<const char*> header_sources)
{
	return create (
		program_name,
		cuda_source,
		header_names.size(),
		header_names.data(),
		header_sources.data()
	);
}

inline program_t create(const char* program_name, const char* cuda_source)
{
	return create(program_name, cuda_source, 0, nullptr, nullptr);
}

template <typename HeaderNameAndSourceFwdIter>
inline program_t create(
	const char* program_name,
	const char* cuda_source,
	HeaderNameAndSourceFwdIter headers_start,
	HeaderNameAndSourceFwdIter headers_end)
{
	auto num_headers = headers_end - headers_start;
	std::vector<const char*> header_names{};
	std::vector<const char*> header_sources{};
	header_names.reserve(num_headers);
	header_sources.reserve(num_headers);
	for(auto& it = headers_start; it < headers_end; it++) {
		header_names.push_back(it->first);
		header_sources.push_back(it->second);
	}
	return create(cuda_source, program_name, num_headers, header_names.data(), header_sources.data());
}

// Note: This won't work for a string->string map... and we can't use a const char* to const char* map, I think.
template <typename HeaderNameAndSourceContainer>
inline program_t create(
	const char*                   program_name,
	const char*                   cuda_source,
	HeaderNameAndSourceContainer  headers)
{
	return create(cuda_source, program_name, headers.cbegin(), headers.cend());
}

} // namespace program

} // namespace rtc

///@cond
class module_t;
///@endcond
namespace module {

inline module_t create(
	const context_t&       context,
	const rtc::program_t&  compiled_program,
	link::options_t        options = {} )
{
	auto ptx = compiled_program.ptx();
	return module::create(context, memory::region_t{ ptx.data(), ptx.size() }, options);
}

} // namespace module

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVRTC_PROGRAM_HPP_
