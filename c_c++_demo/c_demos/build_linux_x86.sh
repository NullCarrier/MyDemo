#!/bin/bash


BUILD_DIR=build_linux_x86-64

#PATH_VULKAN_LIB=/home/irish/vulkan_offscreen_demo/vulkan_offscreen_demo/libs/
#PATH_VULKAN_LIB=/usr/lib/x86_64-linux-gnu/libvulkan.so.1.2.131 

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
pushd $BUILD_DIR

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=install \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..
      # -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-gnu.toolchain.cmake \
make -j4
make install

popd
