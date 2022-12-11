#!/bin/bash


export ANDROID_NDK="/opt/tools/rk/android-ndk-r17/"

BUILD_DIR=build_android_linux_arm64_v8a

#rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
pushd $BUILD_DIR

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=install \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-26 -DANDROID_STL=c++_static \
      ..

make -j4
make install

popd

