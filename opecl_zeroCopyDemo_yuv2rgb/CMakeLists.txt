cmake_minimum_required(VERSION 3.5)
enable_testing()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

####################
#  Project
project(zero_copy_demo)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libopencl-stub/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-party/libdrm/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-party/libdrm/include/libdrm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-party/rknnrt/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-party/rknnrt/include/rknn_runtime)


#add_definitions(-Wall -Wextra -Wno-unused-function -fPIC)
add_definitions(-fPIC)

####################
#  Language
set(STDC_HEADERS 1)
set(CMAKE_CXX_STANDARD 14)
set(HAVE_CXX14 TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3"          CACHE STRING "Flags used by the CXX compiler during DEBUG builds.")
set(CMAKE_CXX_FLAGS_RELEASE        "-O2 -DNDEBUG"     CACHE STRING "Flags used by the CXX compiler during RELEASE builds.")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -DNDEBUG -g3" CACHE STRING "Flags used by the CXX compiler during RELWITHDEBINFO builds.")
option(ENABLE_WERROR "Compile with -Werror enabled (default is OFF)" OFF)
if (ENABLE_WERROR)
    add_compile_options(-Werror)
endif()


set(libopencl_SRCS
    libopencl-stub/src/libopencl.cc
)

add_library(OpenCL SHARED ${libopencl_SRCS})
if(NOT ANDROID)
    target_link_libraries(OpenCL "-Wl,--allow-shlib-undefined" dl)
endif()

####################
# demo
set(zero_copy_demo_SRCS
    main.cc
    opencl_wrapper.cc
    drm_alloc.cc
)

if(ANDROID)
    set(ahb_SRCS 
        AHardwareBuffer_wrapper.cc
    )
endif()

add_executable(zero_copy_demo ${zero_copy_demo_SRCS} ${ahb_SRCS})
add_dependencies(zero_copy_demo OpenCL)
target_link_libraries(zero_copy_demo "-Wl,--allow-shlib-undefined" ${CMAKE_CURRENT_BINARY_DIR}/libOpenCL.so dl)

if(ANDROID)
    target_link_libraries(zero_copy_demo "-Wl,--allow-shlib-undefined" android)
endif()

install(TARGETS zero_copy_demo RUNTIME DESTINATION bin)
