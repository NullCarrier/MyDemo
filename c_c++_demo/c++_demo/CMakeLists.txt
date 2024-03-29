cmake_minimum_required(VERSION 3.10)

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


####################
#  Project
project(C_Program)
# included directory and files
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)




####################
# setting c++ 11 standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDRAD 11)

#setting c++ compiler flags
add_compile_options(-Wall -Wextra -Wno-unused-function -fPIC -lrt)

#set(STDC_HEADERS 1)
# set(CMAKE_C_STANDARD )
# set(HAVE_CXX17 TRUE)
# set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3"          CACHE STRING "Flags used by the CXX compiler during DEBUG builds.")
# set(CMAKE_CXX_FLAGS_RELEASE        "-O2 -DNDEBUG"     CACHE STRING "Flags used by the CXX compiler during RELEASE builds.")
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -DNDEBUG -g3" CACHE STRING "Flags used by the CXX compiler during RELWITHDEBINFO builds.")

option(ENABLE_WERROR "Compile with -Werror enabled (default is OFF)" OFF)
if (ENABLE_WERROR)
    add_compile_options(-Werror)
endif()

# include pthread lib
# set(CMAKE_USE_PTHREADS_INIT true)
# set(CMAKE_USE_PTHREADS_INIT ON)

# find_package(Threads REQUIRED)


####################
# demo source files
set(C++_Source_File
    src/main.cpp
)


add_executable(main ${C++_Source_File} )

#target_link_libraries(main ${Threads} Threads::Threads)
#add_library(yuv2yuvfr_demo SHARED ${zero_copy_demo_SRCS} ${ahb_SRCS})
#add_dependencies(nis_sharpen_demo Vulkan)

# install demo into designated location
install(TARGETS main RUNTIME DESTINATION bin)
