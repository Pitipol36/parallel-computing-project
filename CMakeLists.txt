cmake_minimum_required(VERSION 3.24)

# Default to the GPU in this machine; override with -DCMAKE_CUDA_ARCHITECTURES=75 etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_bvh_raytracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(RT_PTXAS_VERBOSE "Print register/local-memory usage of each kernel" OFF)

# Host code: best single-thread code generation. No fast-math anywhere, so the CPU and
# GPU results can be compared pixel for pixel.
if(MSVC)
  set(RT_CXX_OPT /O2 /arch:AVX2 /fp:precise)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  set(RT_CXX_OPT -O3 -march=native)
endif()

add_library(rt_core STATIC
  src/mesh.cpp
  src/ply_loader.cpp
  src/bvh_build.cpp
  src/camera.cpp
  src/render_cpu.cpp
  src/render_gpu.cu
  src/image.cpp
  src/verify.cpp
)
target_include_directories(rt_core PUBLIC include)
target_compile_options(rt_core PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:${RT_CXX_OPT}>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr>
)
if(RT_PTXAS_VERBOSE)
  target_compile_options(rt_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()
if(MSVC)
  # Pass the host optimisation flags through nvcc to cl as well.
  target_compile_options(rt_core PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>)
endif()

add_executable(raytracer src/main.cpp)
target_link_libraries(raytracer PRIVATE rt_core)

add_executable(rt_tests tests/test_main.cpp)
target_link_libraries(rt_tests PRIVATE rt_core)

enable_testing()
add_test(NAME unit_tests COMMAND rt_tests)
