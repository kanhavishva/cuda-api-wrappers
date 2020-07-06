/**
 * @file examples/common.hpp
 *
 * @brief Common header for many/most/all CUDA API wrapper example programs.
 */
#ifndef EXAMPLES_COMMON_HPP_
#define EXAMPLES_COMMON_HPP_

#include <cuda/api.hpp>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <string>
#include <system_error>
#include <memory>
#include <cstdlib>
#include <vector>
#include <algorithm>

const char* cache_preference_name(cuda::multiprocessor_cache_preference_t pref)
{
	static const char* cache_preference_names[] = {
		"No preference",
		"Equal L1 and shared memory",
		"Prefer shared memory over L1",
		"Prefer L1 over shared memory",
	};
	return cache_preference_names[(off_t) pref];
}

namespace std {

std::ostream& operator<<(std::ostream& os, cuda::multiprocessor_cache_preference_t pref)
{
	return (os << cache_preference_name(pref));
}

std::ostream& operator<<(std::ostream& os, cuda::context::handle_t handle)
{
	return (os << cuda::detail::ptr_as_hex(handle));
}

std::ostream& operator<<(std::ostream& os, const cuda::context_t& context)
{
	return os << "[device " << context.device_id() << " handle " << context.handle() << ']';
}

std::ostream& operator<<(std::ostream& os, const cuda::device_t& device)
{
	return os << "[id " << device.id() << ']';
}

std::string to_string(const cuda::context_t& context)
{
	std::stringstream ss;
	ss.clear();
	ss << context;
	return ss.str();
}

} // namespace std

[[noreturn]] void die_(const std::string& message)
{
	std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}

#define assert_(cond) \
{ \
	auto evaluation_result = (cond); \
	if (not evaluation_result) \
		die_("Assertion failed at line " + std::to_string(__LINE__) + ": " #cond); \
}


void report_current_context(const std::string& prefix = "")
{
	if (not prefix.empty()) { std::cout << prefix << ", the current context is: "; }
	else std::cout << "The current context is: ";
	if (not cuda::context::current::exists()) {
		std::cout << "(None)" << std::endl;
	}
	else {
		auto cc = cuda::context::current::get();
		std::cout << cc << std::endl;
	}
}



#endif /* EXAMPLES_COMMON_HPP_ */
