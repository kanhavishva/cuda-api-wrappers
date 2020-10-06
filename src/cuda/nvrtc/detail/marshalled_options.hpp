
#ifndef SRC_CUDA_API_DETAIL_MARSHALLED_OPTIONS_HPP_
#define SRC_CUDA_API_DETAIL_MARSHALLED_OPTIONS_HPP_

#include <cuda/nvrtc/types.hpp>

#include <cstdlib>
#include <vector>
#include <string>
#include <cstring>

namespace cuda {

namespace rtc {

namespace detail {


class marshalled_options_size_computer_t {
public:
	using size_type = size_t;

protected:
	size_type buffer_pos_ { 0 };
	size_type num_options_ { 0 };

public:

	void append_to_current(string_view sv)
	{
		buffer_pos_ += sv.size();
	}

	void append_to_current(const char *str)
	{
		append_to_current(string_view{str, std::strlen(str)});
	}

	void append_to_current(char)
	{
		buffer_pos_++;
	}

	template <typename I>
	void append_to_current(typename std::enable_if<std::is_integral<typename std::decay<I>::type>::value>::type v)
	{
		using decayed = typename std::decay<I>::type;
		char int_buffer[std::numeric_limits<decayed>::digits10 + 2];
#if __cplusplus >= 201703L
		auto result = std::to_chars(buffer, buffer + sizeof(buffer), v);
		auto num_chars = result.ptr - int_buffer;
#else
		auto num_chars = snprintf(int_buffer, sizeof(int_buffer), "%lld", (long long) v);
#endif
		buffer_pos_ += num_chars;
	}

	void finalize_current()
	{
		buffer_pos_++; // for a '\0'.
		num_options_++;
	}

	void push_back()
	{
		finalize_current();
	}

	template <typename T, typename... Ts>
	void push_back(T&& e, Ts&&... rest)
	{
		append_to_current(std::forward<T>(e));
		push_back(std::forward<Ts>(rest)...);
	}


public:
	size_t num_options() const { return num_options_; }
	size_t buffer_size() const { return buffer_pos_; }
};

} // namespace detail

/**
 * This class is necessary for realizing everything we need from
 * the marshalled options: Easy access using an array of pointers,
 * for the C API - and RAIIness for convenience and safety when
 * using the wrapper classes.
 */
class marshalled_options_t {
public:
	using size_type = size_t;

	marshalled_options_t(size_type buffer_size, size_type max_num_options)
		: buffer_(buffer_size), option_ptrs_(max_num_options), current_option_(buffer_.data())
	{
		if (max_num_options > 0) { option_ptrs_[0] = buffer_.data(); }
	}
	marshalled_options_t(const marshalled_options_t&) = default;
	marshalled_options_t(marshalled_options_t&&) = default;

protected:
	static_vector<char> buffer_;
	static_vector<char*> option_ptrs_;
	char* current_option_;

	size_type buffer_pos_ { 0 };
	size_type num_options_ { 0 };

public:

	friend class compilation_options_t;

	void append_to_current(string_view sv)
	{
#ifndef NDEBUG
		if (num_options_ >= option_ptrs_.size()) {
			std::runtime_error("Attempt to prepare an option beyond the maximum number of options");
		}
#endif
		std::copy_n(sv.cbegin(), sv.size(), buffer_.data() + buffer_pos_);
		buffer_pos_ += sv.size();
	}

	void append_to_current(const char *str)
	{
		append_to_current(string_view{str, std::strlen(str)});
	}

	void append_to_current(char ch)
	{
#ifndef NDEBUG
		if (num_options_ >= option_ptrs_.size()) {
			std::runtime_error("Attempt to prepare an option beyond the maximum number of options");
		}
#endif
		buffer_[buffer_pos_++] = ch;
	}

	template <typename I>
	void append_to_current(typename std::enable_if<std::is_integral<typename std::decay<I>::type>::value>::type v)
	{
#ifndef NDEBUG
		if (num_options_ >= option_ptrs_.size()) {
			std::runtime_error("Attempt to prepare an option beyond the maximum number of options");
		}
#endif
		using decayed = typename std::decay<I>::type;

		char int_buffer[std::numeric_limits<decayed>::digits10 + 2];
#if __cplusplus >= 201703L
		auto result = std::to_chars(buffer, buffer + sizeof(buffer), v);
		auto num_chars = result.ptr - int_buffer;
#else
		auto num_chars = snprintf(int_buffer, sizeof(int_buffer), "%d", v);
#endif
		std::copy_n(int_buffer, num_chars, buffer_.data() + buffer_pos_);
		buffer_pos_ += num_chars;
	}

	void finalize_current()
	{
#ifndef NDEBUG
		if (num_options_ >= option_ptrs_.size()) {
			std::runtime_error("Attempt to insert options beyond the maximum supported by this structure (and probably - a duplicate option)");
		}
#endif
		buffer_[buffer_pos_++] = '\0';
		option_ptrs_[num_options_++] = current_option_;
		current_option_ = buffer_.data() + buffer_pos_;
	}

	void push_back()
	{
		finalize_current();
	}

	template <typename T, typename... Ts>
	void push_back(T&& e, Ts&&... rest)
	{
		append_to_current(std::forward<T>(e));
		push_back(std::forward<Ts>(rest)...);
	}

public:
	span<const char*> option_ptrs() const { return span<const char*>{const_cast<const char**>(option_ptrs_.data()), num_options_}; }
};

} // namespace rtc

} // namespace cuda


#endif /* SRC_CUDA_API_DETAIL_MARSHALLED_OPTIONS_HPP_ */

