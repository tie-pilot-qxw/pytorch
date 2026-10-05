#include <cuda_runtime.h>
// Reads the node handle from device memory (filled in AFTER capture), so this kernel
// can be captured into the same graph as the node it will patch.
__global__ void k_patch(cudaGraphDeviceNode_t* nodeslot, const size_t* offs,
                        const unsigned* vals, int n, int* rcout) {
    cudaGraphDeviceNode_t nd = *nodeslot;
    if (nd == 0) { if (rcout) *rcout = -1; return; }
    int rc = 0;
    for (int i = 0; i < n; ++i) {
        unsigned v = vals[i];
        cudaError_t e = cudaGraphKernelNodeSetParam(nd, offs[i], &v, sizeof(unsigned));
        if (e != cudaSuccess) rc = (int)e;
    }
    if (rcout) *rcout = rc;
}
extern "C" void launch_patch(void* nodeslot, void* offs, void* vals, int n, void* rcout, void* stream) {
    k_patch<<<1,1,0,(cudaStream_t)stream>>>((cudaGraphDeviceNode_t*)nodeslot,
        (const size_t*)offs, (const unsigned*)vals, n, (int*)rcout);
}
