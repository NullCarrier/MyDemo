/*
 * @Author: Chao Li 
 * @Date: 11/12/2022
 * @Description: TO DO 
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstddef>
#include <cmath>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "opencl_wrapper.h"
#include "drm_alloc.h"

#if defined(__ANDROID__)
#include "AHardwareBuffer_wrapper.h"
#endif

using namespace std;

const char * yuv2rgb_str  =	"__kernel void yuv2rgb(__global uchar * yuv, __global uchar * rgb)"
							"{"
							" unsigned int i =get_global_id(0);"
							" unsigned int j =get_global_id(1);"
							" unsigned int W =get_global_size(0);"
							" unsigned int H =get_global_size(1);"
							" uchar y,u,v,r,g,b;"
							" y=yuv[j*W+i];"
							"unsigned int tmp1=W*H+(j>>1)*W+i-i%2;"
							" u=yuv[tmp1];"
							" v=yuv[tmp1+1];"
							" r=y+1.403f*(v-128);"
							" g=y-0.343f*(u-128)-0.714f*(v-128);"
							" b=y+1.770f*(u-128);"
							" unsigned int tmp3=(j*W+i)*3;"
							" rgb[tmp3]=r;"
							" rgb[tmp3+1]=g;"
							" rgb[tmp3+2]=b;"
							"}";

int NUM_ELEMENTS_X = 320;
int NUM_ELEMENTS_Y = 320;
int NUM_ELEMENTS = (NUM_ELEMENTS_X*NUM_ELEMENTS_Y*3);

#define ENABLE_DRM   1
#if defined(__ANDROID__)
#define ENABLE_AHB   1   //Android Hardware buffer
#endif

// #define ENABLE_CPU   1

class DrmObject {
public:
    int drm_buffer_fd;
    int drm_buffer_handle;
    size_t actual_size;
    uint8_t * drm_buf;
};

class RCLContext
{
public:
    cl_device_id devices;
    cl_context context;
    cl_command_queue commandQueue;
    cl_kernel kernel[1];
    cl_program program;
    unsigned int numberOfMemoryObjects;
    cl_mem *memoryObjects;

    //input object for drm
#if ENABLE_DRM
    DrmObject drm_yuv;
    DrmObject drm_rgb;
#endif

    // input object for android hardware buffer
#if ENABLE_AHB
    AHardwareBuffer *ahb_yuv;
    AHardwareBuffer *ahb_rgb;
#endif

};

int64_t getCurrentTimeUS()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}


int rcl_init(RCLContext *rcl_context) {
    cl_int errorNumber;
    char kernel_name[] = "yuv2rgb";
    int width = NUM_ELEMENTS_X;
    int height = NUM_ELEMENTS_Y;
    int channel = 3;


    if (!createContext(&rcl_context->context, &rcl_context->devices)) {
        goto error;
    }

    if (!createProgram(rcl_context->context, rcl_context->devices, (const char **)&yuv2rgb_str, NULL, &rcl_context->program)) {
        goto error;
    }

    if (!createCommandQueue(rcl_context->context, &rcl_context->commandQueue, &rcl_context->devices)) {
        goto error;
    }

    //create kernel
    rcl_context->kernel[0] = clCreateKernel(rcl_context->program, kernel_name, &errorNumber);
    if (rcl_context->kernel[0] == NULL) {
        printf("Couldn't create kernel!\n");
        goto error;
    }

    //create cl_mem objects
    rcl_context->numberOfMemoryObjects = 2;
    rcl_context->memoryObjects = new cl_mem[rcl_context->numberOfMemoryObjects];

    // create from drm
#if ENABLE_DRM
    rcl_context->drm_yuv.drm_buf = (uint8_t *)drm_buf_alloc(width, height, channel*8, 
            &rcl_context->drm_yuv.drm_buffer_fd, &rcl_context->drm_yuv.drm_buffer_handle, 
            &rcl_context->drm_yuv.actual_size);
    rcl_context->drm_rgb.drm_buf = (uint8_t *)drm_buf_alloc(width, height, channel*8, 
            &rcl_context->drm_rgb.drm_buffer_fd, &rcl_context->drm_rgb.drm_buffer_handle, 
            &rcl_context->drm_rgb.actual_size);

    rcl_context->memoryObjects[0] = createCLMemFromDma(rcl_context->context,
                                        rcl_context->devices,
                                        CL_MEM_READ_ONLY,
                                        rcl_context->drm_yuv.drm_buffer_fd,
                                        width*height*3/2);
    rcl_context->memoryObjects[1] = createCLMemFromDma(rcl_context->context,
                                        rcl_context->devices,
                                        CL_MEM_WRITE_ONLY,
                                        rcl_context->drm_rgb.drm_buffer_fd,
                                        width*height*3);
#endif

    // create from AHardwareBuffer 
#if ENABLE_AHB
    rcl_context->ahb_yuv = createAHardwareBuffer(width, height, AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM,
                    AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN);
    rcl_context->ahb_rgb = createAHardwareBuffer(width, height, AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM,
                    AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN);

    rcl_context->memoryObjects[0] = createCLMemFromAHB(rcl_context->context,
                                        rcl_context->devices,
                                        CL_MEM_READ_ONLY,
                                        rcl_context->ahb_yuv);
    rcl_context->memoryObjects[1] = createCLMemFromAHB(rcl_context->context,
                                        rcl_context->devices,
                                        CL_MEM_WRITE_ONLY,
                                        rcl_context->ahb_rgb);
#endif

#if ENABLE_CPU
    rcl_context->memoryObjects[1] = clCreateBuffer(rcl_context->context, CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE |CL_MEM_ALLOC_HOST_PTR,
                                       NUM_ELEMENTS * sizeof(char), NULL, &errorNumber);
#endif

    if (!checkSuccess(errorNumber))
    {
        goto error;
    }

    return 0;
error:
    cleanUpOpenCL(rcl_context->context, rcl_context->commandQueue, rcl_context->program, 
    rcl_context->kernel, sizeof(rcl_context->kernel)/sizeof(cl_kernel),
    rcl_context->memoryObjects, rcl_context->numberOfMemoryObjects);
    cerr << "rcl_init failed " << __FILE__ << ":"<< __LINE__ << endl;
    return -1;
}

void rcl_deinit(RCLContext *rcl_context) {
    // release resources
    cleanUpOpenCL(rcl_context->context, rcl_context->commandQueue, rcl_context->program, 
        rcl_context->kernel, sizeof(rcl_context->kernel)/sizeof(cl_kernel),
        rcl_context->memoryObjects, rcl_context->numberOfMemoryObjects);

    if (rcl_context->memoryObjects != nullptr) {
        delete rcl_context->memoryObjects;

        rcl_context->memoryObjects = nullptr;
    }

#if ENABLE_DRM
    if (rcl_context->drm_yuv.drm_buf) {
        drm_buf_destroy(rcl_context->drm_yuv.drm_buffer_fd, rcl_context->drm_yuv.drm_buffer_handle, 
            rcl_context->drm_yuv.drm_buf, rcl_context->drm_yuv.actual_size);
    }

    if (rcl_context->drm_rgb.drm_buf) {
        drm_buf_destroy(rcl_context->drm_rgb.drm_buffer_fd, rcl_context->drm_rgb.drm_buffer_handle, 
            rcl_context->drm_rgb.drm_buf, rcl_context->drm_rgb.actual_size);
    }
#endif

#if ENABLE_AHB
    if (rcl_context->ahb_yuv) {
        releaseAHardwareBuffer(rcl_context->ahb_yuv);
        rcl_context->ahb_yuv = nullptr;
    }

    if (rcl_context->ahb_rgb) {
        releaseAHardwareBuffer(rcl_context->ahb_rgb);
        rcl_context->ahb_rgb = nullptr;
    }
#endif

}

#define USEMMAP

int rcl_yuv2rgb(RCLContext *rcl_context, unsigned char *cpu_yuv, unsigned char *cpu_rgb) {
    cl_int errorNumber;
    int width = NUM_ELEMENTS_X;
    int height = NUM_ELEMENTS_Y;

#if ENABLE_CPU
    rcl_context->memoryObjects[0] = clImportMemoryARM(rcl_context->context,
                                        CL_MEM_READ_ONLY,
                                        NULL,
                                        cpu_yuv,
                                        width*height*3/2,
                                        &errorNumber );
#endif

    /* Set the kernel argument */
    errorNumber = clSetKernelArg(rcl_context->kernel[0], 0, sizeof(cl_mem), &rcl_context->memoryObjects[0]);
    errorNumber = clSetKernelArg(rcl_context->kernel[0], 1, sizeof(cl_mem), &rcl_context->memoryObjects[1]);

    /* Execute the OpenCL kernel */
    size_t global_item_size[2] = { (size_t)width, (size_t)height};
    /* can be tuned for improving performance */
    size_t local_item_size[2] = {8, 8};
    // errorNumber = clEnqueueNDRangeKernel(rcl_context->commandQueue, rcl_context->kernel[0], 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    errorNumber = clEnqueueNDRangeKernel(rcl_context->commandQueue, rcl_context->kernel[0], 2, NULL, global_item_size, NULL, 0, NULL, NULL);

    /* Wait for kernel execution completion. */
    if (!checkSuccess(clFinish(rcl_context->commandQueue)))
    {
        cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
#if ENABLE_CPU
#ifdef USEMMAP
    void *gpu_cpu_rgb = clEnqueueMapBuffer(rcl_context->commandQueue, rcl_context->memoryObjects[1], CL_TRUE, CL_MAP_READ, 0,NUM_ELEMENTS * sizeof(char),0,0,0, &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cerr << "Failed to map output buffer. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    // memcpy(cpu_rgb, gpu_cpu_rgb, width*height*3);
    clEnqueueUnmapMemObject(rcl_context->commandQueue, rcl_context->memoryObjects[1], gpu_cpu_rgb,0,0,0);
#else
    errorNumber = clEnqueueReadBuffer(rcl_context->commandQueue, rcl_context->memoryObjects[1], CL_TRUE, 0, width*height*3 * sizeof(char), cpu_rgb, 0, NULL, NULL);
#endif
#endif

#if ENABLE_CPU
    clReleaseMemObject(rcl_context->memoryObjects[0]);
    rcl_context->memoryObjects[0] = nullptr;
#endif
    return 0;
}

int NV21_T_RGB(unsigned int width , unsigned int height , unsigned char *yuyv , unsigned char *rgb)
{
    const int nv_start = width * height ;
    int  i, j, index = 0, rgb_index = 0;
    unsigned char y, u, v;
    int r, g, b, nv_index = 0;
	
    for(i = 0; i <  height ; i++)
    {
		for(j = 0; j < width; j ++){
			//nv_index = (rgb_index / 2 - width / 2 * ((i + 1) / 2)) * 2;
			nv_index = i / 2  * width + j - j % 2;

			y = yuyv[rgb_index];
			u = yuyv[nv_start + nv_index ];
			v = yuyv[nv_start + nv_index + 1];			
		
			r = y + (140 * (v-128))/100;  //r
			g = y - (34 * (u-128))/100 - (71 * (v-128))/100; //g
			b = y + (177 * (u-128))/100; //b
				
			if(r > 255)   r = 255;
			if(g > 255)   g = 255;
			if(b > 255)   b = 255;
       		if(r < 0)     r = 0;
			if(g < 0)     g = 0;
			if(b < 0)     b = 0;
			
			index = rgb_index % width + (height - i - 1) * width;
			rgb[index * 3+0] = b;
			rgb[index * 3+1] = g;
			rgb[index * 3+2] = r;
			rgb_index++;
		}
    }
    return 0;
}

int main(int argc, char *argv[]) {
    RCLContext *rcl_context = new RCLContext();
    unsigned char *cpu_yuv = nullptr;
    unsigned char *cpu_rgb = nullptr;

    if (argc == 3) {
        NUM_ELEMENTS_X = atoi(argv[1]);
        NUM_ELEMENTS_Y = atoi(argv[2]);
        NUM_ELEMENTS = (NUM_ELEMENTS_X*NUM_ELEMENTS_Y*3);
    }

    rcl_init(rcl_context);

#if ENABLE_CPU
    cpu_yuv = (unsigned char *)calloc(NUM_ELEMENTS, sizeof(char));
    cpu_rgb = (unsigned char *)calloc(NUM_ELEMENTS, sizeof(char));
#endif

#if ENABLE_DRM
    cpu_yuv = rcl_context->drm_yuv.drm_buf;
    cpu_rgb = rcl_context->drm_rgb.drm_buf;
#endif

#if ENABLE_AHB
    cpu_yuv = (unsigned char *)mapAHardwareBuffer(rcl_context->ahb_yuv, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN);
#endif
  

    FILE *fp=fopen("nv12", "r");
    fread(cpu_yuv, 1, NUM_ELEMENTS/2,fp);
    fclose(fp);

    int64_t oldTime;
    int64_t newTime;
    int total_cnt = 1000;
    int cnt = 0;

    // warm up run
    oldTime = getCurrentTimeUS();
    rcl_yuv2rgb(rcl_context, cpu_yuv, cpu_rgb);
    newTime = getCurrentTimeUS();

    float one_time_cost = (newTime-oldTime)/1000.0;

    if ( one_time_cost > 10.0 ) {
        total_cnt = 500;
    } else if ( one_time_cost > 5.0 ) {
        total_cnt = 2000;
    } else {
        total_cnt = 5000;
    }

    //start test
    oldTime = getCurrentTimeUS();
    while (cnt++ < total_cnt)
    {
       rcl_yuv2rgb(rcl_context, cpu_yuv, cpu_rgb);
    }      
    newTime = getCurrentTimeUS();

    printf("rcl_yuv2rgb(%dx%d): total use %.2f ms, avg %.2f ms\n", 
        NUM_ELEMENTS_X, NUM_ELEMENTS_Y,
        (newTime-oldTime)/1000.0, (newTime-oldTime)/1000.0/total_cnt);

#if ENABLE_CPU

    // warm up
    oldTime = getCurrentTimeUS();
    NV21_T_RGB(NUM_ELEMENTS_X, NUM_ELEMENTS_Y, cpu_yuv, cpu_rgb);
    newTime = getCurrentTimeUS();
    
    one_time_cost = (newTime-oldTime)/1000.0;

    if ( one_time_cost > 15.0 ) {
        total_cnt = 500;
    } else if ( one_time_cost > 10.0 ) {
        total_cnt = 1000;
    } else if ( one_time_cost > 5.0 ) {
        total_cnt = 5000;
    } else {
        total_cnt = 10000;
    }

    // pure cpu test
    cnt = 0;
    oldTime = getCurrentTimeUS();

    while (cnt++ < total_cnt) {
        NV21_T_RGB(NUM_ELEMENTS_X, NUM_ELEMENTS_Y, cpu_yuv, cpu_rgb);
    }

    newTime = getCurrentTimeUS();

    printf("cpu_yuv2rgb(%dx%d): total use %.2f ms, avg %.2f ms\n", 
        NUM_ELEMENTS_X, NUM_ELEMENTS_Y,
        (newTime-oldTime)/1000.0, (newTime-oldTime)/1000.0/total_cnt);
#endif

#if ENABLE_AHB
    cpu_rgb = (unsigned char *)mapAHardwareBuffer(rcl_context->ahb_rgb, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN);
#endif

    fp=fopen("rgb","w");
    fwrite(cpu_rgb, 1, NUM_ELEMENTS,fp);
    fclose(fp);    

#if ENABLE_CPU
    if (cpu_yuv != NULL) {
        free(cpu_yuv);
    }
    if (cpu_rgb != NULL) {
        free(cpu_rgb);
    }

#endif
#if ENABLE_AHB
    unmapAHardwareBuffer(rcl_context->ahb_yuv);
    unmapAHardwareBuffer(rcl_context->ahb_rgb);
#endif

    rcl_deinit(rcl_context);
    delete rcl_context;

    return 0;
}
