cmake_minimum_required(VERSION 3.24)

project(GB10XCuda LANGUAGES CXX)

option(GB10X_CUDA_ENABLE "Enable the native GB10X CUDA targets" OFF)
option(
    GB10X_CUDA_NATIVE_TESTS
    "Register the GPU-executing ABI smoke test"
    OFF
)

if(GB10X_CUDA_NATIVE_TESTS AND NOT GB10X_CUDA_ENABLE)
    message(FATAL_ERROR "GB10X_CUDA_NATIVE_TESTS requires GB10X_CUDA_ENABLE=ON")
endif()

if(NOT GB10X_CUDA_ENABLE)
    message(STATUS "GB10X CUDA targets are disabled")
    return()
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "GB10X CUDA requires Linux, found ${CMAKE_SYSTEM_NAME}")
endif()
if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "GB10X CUDA requires a native aarch64 build")
endif()
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    message(
        FATAL_ERROR
        "GB10X CUDA requires native aarch64, found ${CMAKE_SYSTEM_PROCESSOR}"
    )
endif()
if(
    DEFINED CMAKE_CUDA_ARCHITECTURES
    AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL ""
    AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "OFF"
)
    message(
        FATAL_ERROR
        "GB10X requires CMAKE_CUDA_ARCHITECTURES to be empty or OFF"
    )
endif()
set(
    GB10X_CUDA_ARCHITECTURE_FLAG_REGEX
    "(^|[ \t\r\n;])(--gpu-architecture|--gpu-code|--generate-code|-arch|-code|-gencode)(=|[ \t\r\n;]|$)"
)

function(gb10x_reject_cuda_architecture_flags source value)
    if("${value}" MATCHES "${GB10X_CUDA_ARCHITECTURE_FLAG_REGEX}")
        message(
            FATAL_ERROR
            "GB10X owns CUDA architecture selection; remove architecture flags from ${source}"
        )
    endif()
endfunction()

if(DEFINED CMAKE_CUDA_FLAGS)
    gb10x_reject_cuda_architecture_flags(
        CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS}"
    )
endif()
if(DEFINED CMAKE_CUDA_FLAGS_INIT)
    gb10x_reject_cuda_architecture_flags(
        CMAKE_CUDA_FLAGS_INIT
        "${CMAKE_CUDA_FLAGS_INIT}"
    )
endif()
if(DEFINED ENV{CUDAFLAGS})
    gb10x_reject_cuda_architecture_flags(
        "CUDAFLAGS environment variable"
        "$ENV{CUDAFLAGS}"
    )
endif()

get_cmake_property(gb10x_cache_variables CACHE_VARIABLES)
foreach(gb10x_cache_variable IN LISTS gb10x_cache_variables)
    if(gb10x_cache_variable MATCHES "^CMAKE_CUDA_FLAGS_.+")
        gb10x_reject_cuda_architecture_flags(
            "${gb10x_cache_variable}"
            "${${gb10x_cache_variable}}"
        )
    endif()
endforeach()

set(
    CMAKE_CUDA_ARCHITECTURES
    OFF
    CACHE STRING "GB10X disables CMake CUDA architecture defaults"
    FORCE
)

enable_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    message(
        FATAL_ERROR
        "GB10X requires NVIDIA nvcc, found ${CMAKE_CUDA_COMPILER_ID}"
    )
endif()

function(gb10x_require_nvcc_token option expected)
    execute_process(
        COMMAND "${CMAKE_CUDA_COMPILER}" "${option}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "nvcc ${option} failed (${result}): ${error}")
    endif()
    string(
        REGEX MATCH
        "(^|[ \t\r\n])${expected}($|[ \t\r\n])"
        found
        "${output}"
    )
    if(found STREQUAL "")
        message(
            FATAL_ERROR
            "nvcc ${option} does not report ${expected}: ${output}"
        )
    endif()
endfunction()

gb10x_require_nvcc_token(--list-gpu-code sm_121a)
gb10x_require_nvcc_token(--list-gpu-arch compute_121a)

find_package(CUDAToolkit REQUIRED)

set(GB10X_CUDA_CODEGEN_FLAG "--generate-code=arch=compute_121a,code=sm_121a")

function(gb10x_apply_cuda_target_contract target)
    set_property(TARGET "${target}" PROPERTY CUDA_ARCHITECTURES OFF)
    target_compile_options(
        "${target}"
        PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${GB10X_CUDA_CODEGEN_FLAG}>"
    )
    get_target_property(gb10x_target_type "${target}" TYPE)
    if(NOT gb10x_target_type STREQUAL "STATIC_LIBRARY")
        target_link_options(
            "${target}"
            PRIVATE "$<DEVICE_LINK:${GB10X_CUDA_CODEGEN_FLAG}>"
        )
    endif()
endfunction()

add_library(gb10x_cuda STATIC common/src/gb10x_cuda_abi.cu)
target_include_directories(
    gb10x_cuda
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/common/include"
)
target_link_libraries(gb10x_cuda PUBLIC CUDA::cudart)
target_compile_features(gb10x_cuda PUBLIC cxx_std_17)
gb10x_apply_cuda_target_contract(gb10x_cuda)

if(GB10X_CUDA_NATIVE_TESTS)
    enable_testing()
    add_executable(gb10x_cuda_abi_smoke tests/abi_smoke.cu)
    target_link_libraries(gb10x_cuda_abi_smoke PRIVATE gb10x_cuda)
    target_compile_features(gb10x_cuda_abi_smoke PRIVATE cxx_std_17)
    gb10x_apply_cuda_target_contract(gb10x_cuda_abi_smoke)
    add_test(NAME gb10x_cuda_abi_smoke COMMAND gb10x_cuda_abi_smoke)
endif()
