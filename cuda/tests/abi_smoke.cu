#include "gb10x_cuda_abi.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string_view>

namespace {

int fail(const char *message) {
    std::cerr << "gb10x_cuda_abi_smoke: " << message << '\n';
    return 1;
}

int require_ok(const char *operation, gb10x_cuda_status status) {
    if (status == GB10X_CUDA_STATUS_OK) {
        return 0;
    }

    std::cerr << "gb10x_cuda_abi_smoke: " << operation
              << " failed: " << gb10x_cuda_status_string(status) << '\n';
    return 1;
}

} // namespace

int main() {
    if (gb10x_cuda_abi_version() != GB10X_CUDA_ABI_VERSION) {
        return fail("unexpected ABI version");
    }

    gb10x_cuda_abi_info abi{};
    abi.struct_size = sizeof(abi);
    if (require_ok("gb10x_cuda_get_abi_info", gb10x_cuda_get_abi_info(&abi)) != 0) {
        return 1;
    }
    if (abi.struct_size != sizeof(abi) || abi.abi_version != GB10X_CUDA_ABI_VERSION) {
        return fail("ABI information does not describe ABI v1");
    }
    if (abi.target_sm_major != 12 || abi.target_sm_minor != 1 ||
        abi.target_sm_variant != GB10X_CUDA_SM_VARIANT_A) {
        return fail("library was not built for sm_121a");
    }

    gb10x_cuda_device_info device{};
    device.struct_size = sizeof(device);
    if (require_ok("gb10x_cuda_probe_device", gb10x_cuda_probe_device(0, &device)) != 0) {
        return 1;
    }
    if (device.struct_size != sizeof(device)) {
        return fail("device information has an unexpected struct size");
    }
    if (device.compute_major != 12 || device.compute_minor != 1) {
        return fail("device does not report compute capability 12.1");
    }

    const auto name_end = std::find(
        std::begin(device.name), std::end(device.name), static_cast<std::uint8_t>(0));
    if (name_end == std::end(device.name)) {
        return fail("device name is not nul terminated");
    }
    const auto name_length = static_cast<std::size_t>(name_end - std::begin(device.name));
    const std::string_view name(reinterpret_cast<const char *>(device.name), name_length);
    if (name.find("GB10") == std::string_view::npos) {
        return fail("device name does not identify GB10");
    }

    std::cout << "name=" << name << '\n'
              << "compute_capability=" << device.compute_major << '.' << device.compute_minor
              << '\n'
              << "cuda_runtime_header_version=" << abi.cuda_runtime_header_version << '\n'
              << "cuda_runtime_loaded_version=" << abi.cuda_runtime_loaded_version << '\n'
              << "total_global_memory_bytes=" << device.total_global_memory_bytes << '\n'
              << "l2_cache_bytes=" << device.l2_cache_bytes << '\n'
              << "persisting_l2_max_bytes=" << device.persisting_l2_max_bytes << '\n';
    return 0;
}
