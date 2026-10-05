#include <cuda_runtime.h>
__global__ void k_setgrid(cudaGraphDeviceNode_t node, unsigned x, unsigned y, unsigned z, int* rc) {
    *rc = (int)cudaGraphKernelNodeSetGridDim(node, dim3(x,y,z));
}
__global__ void k_setparam(cudaGraphDeviceNode_t node, size_t off, unsigned val, int* rc) {
    *rc = (int)cudaGraphKernelNodeSetParam(node, off, &val, sizeof(unsigned));
}
extern "C" int launch_set_grid(void* node, unsigned x, unsigned y, unsigned z) {
    int *d; if (cudaMalloc(&d,4)!=cudaSuccess) return -100;
    k_setgrid<<<1,1>>>((cudaGraphDeviceNode_t)node, x,y,z, d);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) { cudaFree(d); return -200 - (int)le; }
    if (cudaDeviceSynchronize()!=cudaSuccess) { cudaFree(d); return -300; }
    int h=-1; cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost); cudaFree(d); return h;
}
extern "C" int launch_set_param(void* node, size_t off, unsigned val) {
    int *d; if (cudaMalloc(&d,4)!=cudaSuccess) return -100;
    k_setparam<<<1,1>>>((cudaGraphDeviceNode_t)node, off, val, d);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) { cudaFree(d); return -200 - (int)le; }
    if (cudaDeviceSynchronize()!=cudaSuccess) { cudaFree(d); return -300; }
    int h=-1; cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost); cudaFree(d); return h;
}
