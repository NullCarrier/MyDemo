/*
 * @Author: Chao Li 
 * @Date: 11/12/2022
 * @Description: TO DO 
*/

#ifndef AHARDWAREBUFFER_WRAPPER_H
#define AHARDWAREBUFFER_WRAPPER_H

#include <android/hardware_buffer.h>

AHardwareBuffer *createAHardwareBuffer(int width, int height, int format, int flags);
void releaseAHardwareBuffer(AHardwareBuffer *ahb);
int AHardwareBufferToCPU(AHardwareBuffer *ahb, void *dst, int size);
int CPUToAHardwareBuffer(AHardwareBuffer *ahb, void *src, int size);
void *mapAHardwareBuffer(AHardwareBuffer *ahb, int usages);
void unmapAHardwareBuffer(AHardwareBuffer *ahb);

#endif
