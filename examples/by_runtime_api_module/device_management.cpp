/**
 * An example program utilizing most/all calls from the CUDA
 * Runtime API module:
 *
 *   Device Management
 *
 * but does not include the API calls relating to IPC (inter-
 * process communication) - sharing pointers to device memory
 * and events among operating system process. That should be
 * covered by a different program.
 *
 */
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <cstdlib>
#include <cassert>
#include <cuda/api/device.hpp>
#include <cuda/api/devices.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/miscellany.hpp>
#include <cuda/api/pci_id_impl.hpp>
#include <cuda/api/peer_to_peer.hpp>
#include <cuda/api/multi_wrapper_impls.hpp>

[[noreturn]] void die_(const std::string& message)
{
	std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}

#define assert_(cond) \
{ \
	auto evaluation_result = (cond); \
	if (not evaluation_result) \
		die_(#cond); \
}

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
} // namespace std

int main(int argc, char **argv)
{
	if (cuda::device::count() == 0) {
		die_("No CUDA devices on this system");
	}

	// TODO: cudaChooseDevice

	// Being very cavalier about our command-line arguments here...
	cuda::device::id_t device_id =  (argc > 1) ?
		std::stoi(argv[1]) : cuda::device::default_device_id;

	if (cuda::device::count() <= device_id) {
		die_("No CUDA device with ID " + std::to_string(device_id));
	}

	auto device = cuda::device::get(device_id);

	std::cout << "Using CUDA device " << device.name() << " (having device ID " << device.id() << ")\n";

	if (device.id() != device_id) {
		die_("The device's reported ID and the ID for which we created the device differ: "
			+ std::to_string(device.id()) + " !=" +  std::to_string(device_id));
	}

	if (device.id() != device.memory().device_id()) {
		die_("The device's reported ID and the device's memory object's reported device ID differ: "
			+ std::to_string(device.id()) + " !=" +  std::to_string(device.memory().device_id()));
	}

	// Attributes and properties
	// ---------------------------

	auto max_registers_per_block = device.get_attribute(cudaDevAttrMaxRegistersPerBlock);
	std::cout
		<< "Maximum number of registers per block on this device: "
		<< max_registers_per_block << "\n";
	assert_(device.properties().regsPerBlock == max_registers_per_block);

	// PCI bus IDs
	// --------------------

	auto pci_id = device.pci_id();
	std::string pci_id_str(pci_id);

	std::cout << "The device's PCI id is: " << pci_id_str << '\n';

	cuda::outstanding_error::ensure_none(cuda::do_clear_errors);

	auto re_obtained_device = cuda::device::get(pci_id_str);
	assert_(re_obtained_device == device);

	// Memory - total and available
	// -----------------------------------

	auto device_global_mem = device.memory();
	auto total_memory = device_global_mem.amount_total();
	auto free_memory = device_global_mem.amount_total();

	std::cout 
		<< "Device " << std::to_string(device.id()) << " reports it has "
		<< free_memory << " bytes free out of " << total_memory << " bytes total global memory "
		<< "(" << (total_memory - free_memory) << " bytes used).\n";

	assert_(free_memory <= total_memory);


	// Specific attributes and properties with their own API calls:
	// L1/shared mem (CacheConfig), shared memory bank size (SharedMemConfig)
	// and stream priority range
	// ----------------------------------------------------------------

	auto cache_preference = device.cache_preference();
	std::cout << "The cache preference for device " << device.id() << " is: " << cache_preference << ".\n";

	auto new_cache_preference =
		cache_preference == cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory ?
		cuda::multiprocessor_cache_preference_t::prefer_shared_memory_over_l1 :
		cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory;
	device.set_cache_preference(new_cache_preference);
	cache_preference = device.cache_preference();
	assert_(cache_preference == new_cache_preference);
	std::cout << "The cache preference for device " << device.id() << " has now been set to: " << new_cache_preference << ".\n";

	auto shared_mem_bank_size = device.shared_memory_bank_size();
	shared_mem_bank_size =
		(shared_mem_bank_size == cudaSharedMemBankSizeFourByte) ?
			cudaSharedMemBankSizeEightByte : cudaSharedMemBankSizeFourByte;
	device.set_shared_memory_bank_size(shared_mem_bank_size);
	auto stream_priority_range = device.stream_priority_range();
	if (stream_priority_range.is_trivial()) {
		std::cout << "Device " << device.id() << " does not support stream priorities. "
			"All streams will have the same (default) priority.\n";
	}
	else {
		std::cout << "Streams on device " << device.id() << " have priorities between "
			<< stream_priority_range.least << " (highest numeric value, least prioritized) and "
			<< std::to_string(stream_priority_range.greatest) << "(lowest numeric values, most prioritized).\n";
		assert(stream_priority_range.least > stream_priority_range.greatest);
	}

	// Resource limits
	// --------------------

	auto printf_fifo_size = device.get_limit(cudaLimitPrintfFifoSize);
	std::cout << "The printf FIFO size for device " << device_id << " is " << printf_fifo_size << ".\n";
	decltype(printf_fifo_size) new_printf_fifo_size =
		(printf_fifo_size <= 1024) ?  2 * printf_fifo_size : printf_fifo_size - 512;
	device.set_limit(cudaLimitPrintfFifoSize, new_printf_fifo_size);
	printf_fifo_size = device.get_limit(cudaLimitPrintfFifoSize);
	assert_(printf_fifo_size == new_printf_fifo_size);

	// Flags - yes, yet another kind of attribute/property
	// ----------------------------------------------------

	std::cout << "Device " << device.id() << " uses a"
		<< (device.synch_scheduling_policy() ? " synchronous" : "n asynchronous")
		<< " scheduling policy.\n";
	std::cout << "Device " << device.id() << " is set to "
		<< (device.keeping_larger_local_mem_after_resize() ? "keep" : "discard")
		<< " shared memory allocation after launch.\n";
	std::cout << "Device " << device.id()
		<< " is set " << (device.can_map_host_memory() ? "to allow" : "not to allow")
		<< " pinned mapped memory.\n";
	// TODO: Change the settings as well obtaining them

	// Peer-to-peer
	// --------------------

	auto device_count = cuda::device::count();
	if (device_count > 1) {
		// This makes assumptions about the valid IDs and their use
		auto peer_id = (argc > 2) ?
			std::stoi(argv[2]): (device.id() + 1) % cuda::device::count();
		auto peer_device = cuda::device::get(peer_id);
		if (device.can_access(peer_device)) {
			auto atomics_supported_over_link = cuda::device::peer_to_peer::get_attribute(
				cudaDevP2PAttrNativeAtomicSupported, cuda::device::get(device_id), cuda::device::get(peer_id));
			std::cout
				<< "Native atomics are " << (atomics_supported_over_link ? "" : "not ")
				<< "supported over the link from device " << device_id
				<< " to device " << peer_id << ".\n";
			device.disable_access_to(peer_device);
			// TODO: Try some device-to-device access here, expect an exception
			device.enable_access_to(peer_device);
			// TODO: Try some device-to-device access here
		}
	}

	// Current device manipulation
	// ----------------------------

	if (device_count > 1) {
		auto device_0 = cuda::device::get(0);
		auto device_1 = cuda::device::get(1);
		cuda::device::current::set(device_0);
		assert_(cuda::device::current::get() == device_0);
		assert_(cuda::device::current::detail::get_id() == device_0.id());
		cuda::device::current::set(device_1);
		assert_(cuda::device::current::get() == device_1);
		assert_(cuda::device::current::detail::get_id() == device_1.id());
	}

	try {
		cuda::device::current::detail::set(device_count);
		die_("Should not have been able to set the current device to "
			+ std::to_string(device_count) + " since that's the device count, and "
			+ "the maximum valid ID should be " + std::to_string(device_count - 1)
			+ " (one less)");
	}
	catch(cuda::runtime_error& e) {
		assert_(e.code() == cuda::status::invalid_device);
		// We expected to get this exception, just clear it
		cuda::outstanding_error::clear();
	}

	// Iterate over all devices
	// ------------------------

	auto devices = cuda::devices();
	assert_(devices.size() == cuda::device::count());
	std::cout << "The devices container has " << devices.size() << " element" << (devices.size() == 1 ? "" : "s") << ".\n";
	std::cout << "Let's count the device IDs... ";
	for(auto device : cuda::devices()) {
		std::cout << (int) device.id() << ' ';
		device.synchronize();
	}
	std::cout << "done.\n";



	// Synchronize and reset
	// --------------------

	device.synchronize();
	device.reset();

	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
