/*
 * @Author: Chao Li 
 * @Date: 11/12/2022
 * @Description: TO DO 
*/

#ifndef __DRM_ALLOC_H__
#define __DRM_ALLOC_H__

void* drm_buf_alloc(int TexWidth, int TexHeight,int bpp, int *fd, int *handle, size_t *actual_size);
int drm_buf_destroy(int buf_fd, int handle, void *drm_buf, size_t size);

#endif