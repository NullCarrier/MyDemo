/*
 * @Author: Chao Li 
 * @Date: 11/12/2022
 * @Description: TO DO 
*/

#include <string.h>
#include <stdio.h>

#include "AHardwareBuffer_wrapper.h"


AHardwareBuffer *createAHardwareBuffer(int width, int height, int format, int flags) {
    AHardwareBuffer_Desc usage;
    int ret = 0;

    usage.format = format;
    usage.height = height;
    usage.width = width;
    usage.layers = 1;
    usage.rfu0 = 0;
    usage.rfu1 = 0;
    usage.stride = -1;
    usage.usage = flags;

    AHardwareBuffer* ahb;
    ret = AHardwareBuffer_allocate(&usage, &ahb);

    if (ret != 0) {
        printf("AHardwareBuffer_allocate fail, ret=%d\n", ret);
        return nullptr;
    }

    AHardwareBuffer_describe(ahb, &usage);

    printf("createAHardwareBuffer:w=%d,h=%d, stride=%d\n", usage.width, usage.height, usage.stride);

    return ahb;
}

void releaseAHardwareBuffer(AHardwareBuffer *ahb) {
    if (ahb == nullptr) {
        return;
    }

    AHardwareBuffer_release(ahb);
}

int AHardwareBufferToCPU(AHardwareBuffer *ahb, void *dst, int size) {
    void *ahbVirtualAddress;
    // int32_t BytesPerPixel;
    // int32_t BytesPerStride;
    int ret = 0;

    if ((ahb == nullptr) || (dst == nullptr) || (size <= 0)) {
        return -1;
    }

    ret = AHardwareBuffer_lock(ahb, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, nullptr, 
            (void**)&ahbVirtualAddress);
    
    if (ret != 0) {
        printf("Lock AHardwareBuffer fail!\n");
        return -1;
    }

    // TODO
    // check size and Stride
    
    //copy
    memcpy(dst, ahbVirtualAddress, size);

    AHardwareBuffer_unlock(ahb, nullptr);

    return 0;
}

int CPUToAHardwareBuffer(AHardwareBuffer *ahb, void *src, int size) {
    void *ahbVirtualAddress;
    // int32_t BytesPerPixel;
    // int32_t BytesPerStride;
    int ret = 0;

    if ((ahb == nullptr) || (src == nullptr) || (size <= 0)) {
        return -1;
    }

    ret = AHardwareBuffer_lock(ahb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, 
            (void**)&ahbVirtualAddress);
    
    if (ret != 0) {
        printf("Lock AHardwareBuffer fail!\n");
        return -1;
    }

    // TODO
    // check size and Stride
    
    //copy
    memcpy(ahbVirtualAddress, src, size);

    AHardwareBuffer_unlock(ahb, nullptr);

    return 0;
}

void * mapAHardwareBuffer(AHardwareBuffer *ahb, int usages) {
    void *ahbVirtualAddress;

    int ret = 0;

    if (ahb == nullptr) {
        return nullptr;
    }

    ret = AHardwareBuffer_lock(ahb, usages, -1, nullptr, 
            (void**)&ahbVirtualAddress);
    
    if (ret != 0) {
        printf("Lock AHardwareBuffer fail!\n");
        return nullptr;
    }

    return ahbVirtualAddress;
}

void unmapAHardwareBuffer(AHardwareBuffer *ahb) {
    AHardwareBuffer_unlock(ahb, nullptr);
}
