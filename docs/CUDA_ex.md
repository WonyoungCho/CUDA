# Hello World!

```c
#include <stdio.h>
__global__ void helloFromGPU(void)
{
    printf("Hello World From GPU!\n");
}
int main(void)
{
    printf("Hello World from CPU!\n");
    helloFromGPU<<<1,10>>>();
    cudaDeviceReset();
    return 0;
}
```
```sh
nvcc -arch=sm_70 hello.cu -o a
```

- **Batch file**
```sh
#!/bin/sh
#SBATCH -J gpu_05
#SBATCH --time=00:05:00
#SBATCH -p dual_v100_node
#SBATCH -o result_%j.out
##SBATCH -e result_%j.err
#SBATCH --gres=gpu:1
srun ./a
```
- **Running**
```sh
$ sbatch run.sh
```
```sh
$ cat result_19790.out
==========================================
SLURM_JOB_ID = 19790
SLURM_NODELIST = tesla27
==========================================
Hello World from CPU!
Hello World From GPU!
Hello World From GPU!
Hello World From GPU!
Hello World From GPU!
Hello World From GPU!
Hello World From GPU!
Hello World From GPU!
Hello World From GPU!
Hello World From GPU!
Hello World From GPU!
```

# Kernel function

Function | Run on | Call from| Return type
:-:|:-:|:-|:-:
`__global__` | Device | Host <br> Device in up to Compute capability 3.5 | void
`__device__` | Device | Device <br> 그리드와 블록을 지정할 수 없다. |
`__host__` | Host | Host | optional

- **Example - device**
```cu
$ cat devicekernel.cu
#include <stdio.h>
__global__ void helloFromHost();
__device__ int helloFromDevice(int tid);
int main()
{
        helloFromHost<<<1,5>>>();
        cudaDeviceReset();
        return 0;
}
__global__ void helloFromHost()
{
        int tid=threadIdx.x;
        printf("Hello world From __global__ kernel: %d\n",tid);
    int tid1=helloFromDevice(tid);
    printf("tid1 : %d\n",tid1);
}

__device__ int helloFromDevice(int tid)
{
        printf("Hello world Form __device__ kernel: %d\n",tid);
        return tid+1;
}
```
```sh
==========================================
SLURM_JOB_ID = 19814
SLURM_NODELIST = tesla24
==========================================
Hello world From __global__ kernel: 0
Hello world From __global__ kernel: 1
Hello world From __global__ kernel: 2
Hello world From __global__ kernel: 3
Hello world From __global__ kernel: 4
Hello world Form __device__ kernel: 0
Hello world Form __device__ kernel: 1
Hello world Form __device__ kernel: 2
Hello world Form __device__ kernel: 3
Hello world Form __device__ kernel: 4
tid1 : 1
tid1 : 2
tid1 : 3
tid1 : 4
tid1 : 5
```

- **Example -host & device**
```cu
#include <stdio.h>
__host__ __device__ void Print()
{
        printf("Hello World\n");
}
__global__ void Wrapper()
{
        Print();
}
int main()
{
        Print();
        printf("==================\n");
        Wrapper<<<1,5>>>();
        cudaDeviceReset();
        return 0;
}
```
```sh
==========================================
SLURM_JOB_ID = 19819
SLURM_NODELIST = tesla24
==========================================
Hello World
==================
Hello World
Hello World
Hello World
Hello World
Hello World
```

# Memory copy
```cu
#include <stdio.h>
__device__ void PrintArray(int tid, int *A)
{
    printf("A[%d]=%d\n",tid,A[tid]);
        if(tid==0) printf("======================\n");
}
__global__ void Print(int *A)
{
        int tid=threadIdx.x;
        PrintArray(tid,A);
}
int main()
{
        int *d1_A, *d2_A, *h_A, *h_B;
        int size=5;
        int i;
        h_A=(int*)malloc(size*sizeof(int));
        h_B=(int*)malloc(size*sizeof(int));
        for(i=0;i<size;i++) h_A[i]=i;
        // Allocate Device memories............
        cudaSetDevice(0);
        cudaMalloc((int**)&d1_A,size*sizeof(int));
        cudaSetDevice(1);
        cudaMalloc((int**)&d2_A,size*sizeof(int));
        //.........................................

        // Data Transfer : Host -> device 0
        cudaSetDevice(0);
        cudaMemcpy(d1_A,h_A,size*sizeof(int), cudaMemcpyHostToDevice);
//      cudaMemcpy(d1_A,h_A,size*sizeof(int), cudaMemcpyDefault);
        Print<<<1,5>>>(d1_A);
        cudaDeviceSynchronize();

        // Data Transfer : Device 0 -> Device 1
        cudaMemcpy(d2_A,d1_A,size*sizeof(int), cudaMemcpyDeviceToDevice);
//      cudaMemcpy(d2_A,d1_A,size*sizeof(int), cudaMemcpyDefault);
        cudaSetDevice(1);
        Print<<<1,5>>>(d2_A);
        cudaDeviceSynchronize();

        // Data Transfer : Device 2 -> Host
        cudaMemcpy(h_B,d2_A,size*sizeof(int),cudaMemcpyDeviceToHost);
//      cudaMemcpy(h_B,d2_A,size*sizeof(int),cudaMemcpyDefault);
        for(i=0;i<size;i++) printf("h_B[%d]=%d\n",i,h_B[i]);
        cudaFree(d1_A); cudaFree(d2_A);
        cudaDeviceReset();
        return 0;
}
```
```sh
==========================================
SLURM_JOB_ID = 19823
SLURM_NODELIST = tesla24
==========================================
A[0]=0
A[1]=1
A[2]=2
A[3]=3
A[4]=4
======================
A[0]=0
A[1]=1
A[2]=2
A[3]=3
A[4]=4
======================
h_B[0]=0
h_B[1]=1
h_B[2]=2
h_B[3]=3
h_B[4]=4
```

# Vector addition
```cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
/*
#define CHECK(call)                                                    \
{                                                                      \
        const cudaError_t error = call;                                    \
        if (error != cudaSuccess)                                          \
        {                                                                                                                      \
                fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);         \
                fprintf(stderr, "code: %d, reason: %s\n", error,               \
                cudaGetErrorString(error));                                    \
                exit(1);                                                       \
        }                                                                  \
}
*/
inline void CHECK(const cudaError_t error)
{
        if(error != cudaSuccess)
        {
                fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
                fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));
                exit(1);
        }
}

double cpuTimer()
{
        struct timeval tp;
        gettimeofday(&tp, NULL);
        return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void initialData(float *arr, int size)
{
        time_t t;
        srand((unsigned)time(&t));  // seed
        for(int i=0;i<size;i++)
                arr[i]=(float)(rand())/RAND_MAX;
}
void AddVecOnHost(float *A, float *B, float *C, const int size)
{
#pragma omp parallel for
        for(int i=0;i<size;i++)
                C[i] = A[i] + B[i];
}

__global__ void AddVecOnGPU(float *A, float *B, float *C, const int size)
{
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if(idx<size) C[idx] = A[idx] + B[idx];
}

void checkResult(float *host, float *gpu, const int N)
{
        double epsilon = 1.0e-8;
        bool match = 1;
        for(int i=0;i<N;i++)
        {
                if(abs(host[i] - gpu[i]) > epsilon)
                {
                        match = 0;
                        printf("Vector do not match!\n");
                        printf("host %5.2f, gpu %5.2f at current %d\n", host[i], gpu[i], i);
                        break;
                }
        }
        if(match) printf("Vectors match.\n");
}
int main(int argc, char **argv)
{
//      int nSize = 1<<23;   //16M
        int nSize = 1<<24;   //16M
        printf("Vector size : %d\n", nSize);
/*********** on HOST *******************/
        // malloc host memory
        size_t nBytes = nSize*sizeof(float);

        float *h_A, *h_B, *hostResult, *gpuResult;
        h_A = (float*)malloc(nBytes);
        h_B = (float*)malloc(nBytes);
        hostResult = (float*)malloc(nBytes);
        gpuResult = (float*)malloc(nBytes);

        double iStart, iEnd;
        double ElapsedTime;

        initialData(h_A, nSize);
        initialData(h_B, nSize);

        memset(hostResult, 0, nBytes);
        memset(gpuResult, 0, nBytes);

        iStart=cpuTimer();
        AddVecOnHost(h_A, h_B, hostResult, nSize);
        iEnd = cpuTimer();
        ElapsedTime = iEnd - iStart;
        printf("Elapsed Time in AddVecOnHost : %f\n",ElapsedTime);
/*****************************************/

/********** ON GPU **********************/
        // malloc device global memory
        float *d_A, *d_B, *d_C;
        CHECK(cudaMalloc((float**)&d_A, nBytes));
        CHECK(cudaMalloc((float**)&d_B, nBytes));
        CHECK(cudaMalloc((float**)&d_C, nBytes));

        // Data transfer : Host --> Device
        CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

        // dimension of thread block and grid
        dim3 block(256);
        dim3 grid((nSize+block.x-1)/block.x);

        // create tow events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        iStart = cpuTimer();
        float Etime;
        cudaEventRecord(start);
        AddVecOnGPU<<<grid, block>>>(d_A, d_B, d_C, nSize);
        CHECK(cudaDeviceSynchronize());
//      ElapsedTime = cpuTimer() - iStart;
        cudaEventRecord(stop);
        ElapsedTime = cpuTimer() - iStart;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&Etime, start, stop);
        printf("Elapsed Time in AddVecOnGPU<<<%d, %d>>> : %f ms\n", grid.x, block.x, Etime);
//      printf("GPU Timer : %f ms , CPU Timer : %f ms\n",Etime, ElapsedTime*1000.0);

        CHECK(cudaMemcpy(gpuResult, d_C, nBytes, cudaMemcpyDeviceToHost));
/****************************************/

        // check results
        checkResult(hostResult, gpuResult, nSize);

        // memory deallocate
        free(h_A),      free(h_B),      free(hostResult),       free(gpuResult);
        CHECK(cudaFree(d_A)),   CHECK(cudaFree(d_B)),   CHECK(cudaFree(d_C));
        return 0;
}
```
```sh
Vector size : 16777216
Elapsed Time in AddVecOnHost : 0.066860
Elapsed Time in AddVecOnGPU<<<65536, 256>>> : 0.355040 ms
Vectors match.
```
