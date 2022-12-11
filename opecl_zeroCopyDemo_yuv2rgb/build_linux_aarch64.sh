#!/bin/bash


BUILD_DIR=build_linux_aarch64

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
pushd $BUILD_DIR

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=install \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-gnu.toolchain.cmake \
      -DRKNN_RT_ONLY=ON \
      ..

make -j4
make install

popd

