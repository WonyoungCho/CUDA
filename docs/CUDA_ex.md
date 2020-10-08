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
nvcc -arch=sm_75 hello.cu -o a
```
> - RTX 2080 Ti : 7.5


- **Batch file**
```batch
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
`__global__` | Device | - Host <br> - Device in up to Compute capability 3.5 | void
`__device__` | Device | - Device <br> - 그리드와 블록을 지정할 수 없다. |
`__host__` | Host | Host | optional

- **Example - device**
```c
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
```c
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
```c
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
    //  cudaMemcpy(d1_A,h_A,size*sizeof(int), cudaMemcpyDefault);
    Print<<<1,5>>>(d1_A);
    cudaDeviceSynchronize();

    // Data Transfer : Device 0 -> Device 1
    cudaMemcpy(d2_A,d1_A,size*sizeof(int), cudaMemcpyDeviceToDevice);
    //  cudaMemcpy(d2_A,d1_A,size*sizeof(int), cudaMemcpyDefault);
    cudaSetDevice(1);
    Print<<<1,5>>>(d2_A);
    cudaDeviceSynchronize();

    // Data Transfer : Device 1 -> Host
    cudaMemcpy(h_B,d2_A,size*sizeof(int),cudaMemcpyDeviceToHost);
    //  cudaMemcpy(h_B,d2_A,size*sizeof(int),cudaMemcpyDefault);
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
```c
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
    {                                                                  \
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
//  int nSize = 1<<23;   //16M
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
//  ElapsedTime = cpuTimer() - iStart;
    cudaEventRecord(stop);
    ElapsedTime = cpuTimer() - iStart;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Etime, start, stop);
    printf("Elapsed Time in AddVecOnGPU<<<%d, %d>>> : %f ms\n", grid.x, block.x, Etime);
//  printf("GPU Timer : %f ms , CPU Timer : %f ms\n",Etime, ElapsedTime*1000.0);

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

# Matrix multiplication
```c
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

inline void CHECK(const cudaError_t error)
{
    if(error !=cudaSuccess)
    {
            fprintf(stderr, "Error: %s:%d, ",__FILE__,__LINE__);
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));
            exit(1);
    }
}
double cpuTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void initialData(float *arr, const int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for(int i=0;i<size;i++)
        arr[i]= (float)(rand())/RAND_MAX;
}
void MatMulOnCPU(float *A, float *B, float *C, const int Arows, const int Acols, const int Bcols)
{
    float sum;
    for(int i=0;i<Arows;i++)
    {
        for(int j=0;j<Bcols;j++)
        {
            sum = 0.0f;
            for(int k=0;k<Acols;k++)
                {
                sum += A[i*Acols+k]*B[k*Bcols+j];
                }
                    C[i*Bcols+j]=sum;
        }
    }
}
__global__ void MatMultOnGPU(float *A, float *B, float *C, const int Arows, const int Acols, const int Bcols)
{
    int tx = blockDim.x*blockIdx.x + threadIdx.x;   // col of C
    int ty = blockDim.y*blockIdx.y + threadIdx.y;   // row of C
    int tid = ty*Bcols+tx;


    float sum=0.0f;
    if(tx < Bcols && ty <Arows )
    {
        for(int i=0;i<Acols;i++)
        {
            sum += A[ty*Acols + i]*B[i*Bcols+tx];
        }
            C[tid]=sum;
        }
}

void checkResult(float *host, float *gpu, const int N)
{
    double epsilon = 1.0e-8;
    bool match = 1;
    for(int i=0;i<N;i++)
    {
        if(abs(host[i]-gpu[i])>epsilon)
        {
            match = 0;
            printf("Matrices do not match!\n");
            printf("host %10.7f, gpu %10.7f at current %d\n", host[i], gpu[i], i);
            break;
        }
    }
    if(match)printf("Matrices match.\n");
}
int main(int argc, char **argv)
{
    double Start, ElapsedTime;
    float ETime;
    float *MatA, *MatB, *MatC, *gpu_MatC;
    int Arows=300, Acols=200, Bcols=400;
    int threads_x=32, threads_y=32;
    if(argc>1) Arows=atoi(argv[1]);
    if(argc>2) Acols=atoi(argv[2]);
    if(argc>3) Bcols=atoi(argv[3]);
    if(argc>4) threads_x = atoi(argv[4]);
    if(argc>5) threads_y = atoi(argv[5]);
    /************ ON CPU **************/
    MatA=(float*)malloc(Arows*Acols*sizeof(float));
    MatB=(float*)malloc(Acols*Bcols*sizeof(float));
    MatC=(float*)malloc(Arows*Bcols*sizeof(float));
    gpu_MatC=(float*)malloc(Arows*Bcols*sizeof(float));

    initialData(MatA, Arows*Acols);
    initialData(MatB, Acols*Bcols);

    Start=cpuTimer();
    MatMulOnCPU(MatA, MatB, MatC, Arows, Acols, Bcols);
    ElapsedTime=cpuTimer()-Start;
    printf("Elapsed Time on CPU : %f sec\n",ElapsedTime);
    /**********************************/

    /************ ON GPU **************/
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((float**)&d_MatA, Arows*Acols*sizeof(float)));
    CHECK(cudaMalloc((float**)&d_MatB, Acols*Bcols*sizeof(float)));
    CHECK(cudaMalloc((float**)&d_MatC, Arows*Bcols*sizeof(float)));

    // create two events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    CHECK(cudaMemcpy(d_MatA,MatA, Arows*Acols*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB,MatB, Acols*Bcols*sizeof(float),cudaMemcpyHostToDevice));
    dim3 block(threads_x,threads_y,1);
    dim3 grid((Bcols+block.x-1)/block.x, (Arows+block.y-1)/block.y, 1);
    MatMultOnGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, Arows, Acols, Bcols);
    CHECK(cudaMemcpy(gpu_MatC, d_MatC, Arows*Bcols*sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ETime, start, stop);
    printf("Elapsed Time on GPU : %f sec\n",ETime*1e-3);
    /**********************************/
    checkResult(MatC, gpu_MatC, Arows*Bcols);
    free(MatA),     free(MatB),     free(MatC),     free(gpu_MatC);
    CHECK(cudaFree(d_MatA)), CHECK(cudaFree(d_MatB)), CHECK(cudaFree(d_MatC));

    CHECK(cudaDeviceReset());
    return 0;
}
```
```sh
$ nvcc -arch=sm_70 -fmad=false -o a MatMul.cu
```
```sh
$ nvprof ./a
Elapsed Time on CPU : 0.113034 sec
==19259== NVPROF is profiling process 19259, command: ./a
Elapsed Time on GPU : 0.000876 sec
Matrices match.
==19259== Profiling application: ./a
==19259== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   40.71%  51.072us         2  25.536us  22.272us  28.800us  [CUDA memcpy HtoD]
                   29.95%  37.567us         1  37.567us  37.567us  37.567us  [CUDA memcpy DtoH]
                   29.34%  36.799us         1  36.799us  36.799us  36.799us  MatMultOnGPU(float*, float*, float*, int, int, int)
      API calls:   66.67%  579.62ms         3  193.21ms  8.0520us  579.15ms  cudaMalloc
                   32.99%  286.81ms         1  286.81ms  286.81ms  286.81ms  cudaDeviceReset
                    0.11%  977.90us       188  5.2010us     199ns  197.38us  cuDeviceGetAttribute
                    0.08%  737.75us         3  245.92us  124.43us  470.33us  cudaMemcpy
                    0.07%  605.73us         2  302.86us  302.10us  303.62us  cuDeviceTotalMem
                    0.05%  409.69us         3  136.56us  15.931us  226.15us  cudaFree
                    0.01%  89.901us         1  89.901us  89.901us  89.901us  cudaLaunch
                    0.01%  88.392us         2  44.196us  39.750us  48.642us  cuDeviceGetName
                    0.00%  23.613us         2  11.806us  6.8740us  16.739us  cudaEventRecord
                    0.00%  18.100us         6  3.0160us     203ns  16.462us  cudaSetupArgument
                    0.00%  14.179us         2  7.0890us  1.3900us  12.789us  cudaEventCreate
                    0.00%  12.035us         1  12.035us  12.035us  12.035us  cudaEventElapsedTime
                    0.00%  5.6540us         1  5.6540us  5.6540us  5.6540us  cudaEventSynchronize
                    0.00%  4.3410us         3  1.4470us     255ns  3.3950us  cuDeviceGetCount
                    0.00%  2.2400us         4     560ns     239ns     998ns  cuDeviceGet
                    0.00%  2.1440us         1  2.1440us  2.1440us  2.1440us  cudaConfigureCall
```

# Occupancy

- **Example 1**
```c
#include <stdio.h>
#include <cuda_runtime.h>
// device code
__global__ void MyKernel(int *d, int *a, int *b)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    d[idx]=a[idx] +b[idx];
}

// host code
int main()
{
    int numBlocks;
    int blockSize=32;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, MyKernel,blockSize,0);
    int activeWarps = numBlocks*(blockSize/prop.warpSize);
    int maxWarps = prop.maxThreadsPerMultiProcessor/prop.warpSize;
    double occupancy = (double)activeWarps/maxWarps * 100;
    printf("Max # of Blocks : %d\n",numBlocks);
    printf("ActiveWarps : %d\n",activeWarps);
    printf("MaxWarps : %d\n",maxWarps);
    printf("Occupancy = %5.2f %\n",occupancy);
    return 0;
}
```

```sh
$ nvcc nvcc -arch=sm_70 --ptxas-options=-v -o a occupancy1.cu
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z8MyKernelPiS_S_' for 'sm_70'
ptxas info    : Function properties for _Z8MyKernelPiS_S_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, 376 bytes cmem[0]
```
- `--ptxas-options=-v`를 이용해 얻은 리소스 사용량을 Occupancy calculator를 이용하여 Occupancy를 얻을 수 있다.

```sh
Max # of Blocks : 32
ActiveWarps : 32
MaxWarps : 64
Occupancy = 50.00 %
```

- **Example 2**
```c
#include <stdio.h>
#include <cuda_runtime.h>
// Device code
__global__ void MyKernel(int *array, int arrayCount)
{
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   if (idx < arrayCount) {
      array[idx] *= array[idx];
   }
}
// Host code
int main(int argc, char **argv)
{
   int blockSize;
   int minGridSize;
   int arrayCount=1000;
   cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,(void*)MyKernel,0,arrayCount);
   printf("minGridSize : %d \n",minGridSize);
   printf("blockSize : %d\n",blockSize);
   return 0;
}
```
```sh
minGridSize : 320
blockSize : 512
```

# Reduction
```c
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

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

void initialData(int *arr, int size)
{
    time_t t;
    srand((unsigned)time(&t));  // seed
    for(int i=0;i<size;i++)
        arr[i]=(int)(rand() & 0xFF);
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid=threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x*blockDim.x;

    // boundary check
    if(idx>=n) return;

    // in-place reduction in global memory
    for(int stride=blockDim.x/2;stride>0;stride>>=1)
    {
        if(tid<stride)
            idata[tid]+= idata[tid+stride];
            __syncthreads();
    }
    // write result for this block to global mem
    if(tid==0) g_odata[blockIdx.x]=idata[0];
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x*blockDim.x;

    // boundary check
    if(idx>=n) return;

    // in-place reduction in global memory
    for(int stride=1;stride<blockDim.x; stride *=2)
    {
        if((tid%(2*stride))==0)
            idata[tid] += idata[tid+stride];
        // synchrnoize within block
        __syncthreads();
    }
    // write result for this block to global mem
    if(tid==0) g_odata[blockIdx.x] = idata[0];
}

int recursiveReduce(int *data, int const size)
{
    // terminate check
    if(size==1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for(int i=0;i<stride;i++)
        data[i] += data[i+stride];
    // call recursively
    return recursiveReduce(data, stride);
}

int main(void)
{
    int cpu_sum, gpu_sum;

    // initialize
    int size = 1<<24;       // 16M
    printf("array size %d\n", size);

    // execution configuration
    int blocksize = 512;
    dim3 block(blocksize,1);
    dim3 grid((size+block.x-1)/block.x,1);

    // allocate host memory
    size_t bytes = size*sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x*sizeof(int));
    int *tmp=(int*)malloc(bytes);

    // allocate device memory
    int *d_idata, *d_odata;
    cudaMalloc((void**)&d_idata, bytes);
    cudaMalloc((void**)&d_odata, grid.x*sizeof(int));

    initialData(h_idata, size);
    memcpy(tmp, h_idata, bytes);
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    // cpu reduction
    double iStart = cpuTimer();
    cpu_sum = recursiveReduce(tmp, size);
    double ElapsedTime = cpuTimer()-iStart;
    printf("CPU reduction : \t\t%d, Elapsed Time : %f sec\n", cpu_sum, ElapsedTime);

    /********** GPU **************/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0;i<grid.x;i++) gpu_sum += h_odata[i];
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ETime;
    cudaEventElapsedTime(&ETime, start, stop);
    printf("gpu reduction(Neighbored) : \t%d, Elapsed Time : %f sec\n", gpu_sum, ETime*1e-3f);
    /***********************************/

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);

    gpu_sum=0;
    
    for(int i=0;i<grid.x;i++) gpu_sum += h_odata[i];
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ETime, start, stop);
    printf("gpu reduction(Interleaved) : \t%d, Elapsed Time : %f sec\n", gpu_sum, ETime*1e-3f);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_idata), free(h_odata), free(tmp);
    cudaFree(d_idata), cudaFree(d_odata);
    return 0;
}
```
```sh
$ nvprof --metrics inst_per_warp ./a
array size 16777216
==21199== NVPROF is profiling process 21199, command: ./a
CPU reduction : 8389106.000000, Elapsed Time : 0.063129 sec
gpu reduction : 8389123.000000, Elapsed Time : 0.026782 sec
==21199== Profiling application: ./a
==21199== Profiling result:
==21199== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-PCIE-16GB (0)"
    Kernel: reduceInterleaved(float*, float*, unsigned int)
          1                             inst_per_warp                     Instructions per warp  132.875000  132.875000  132.875000
```

# Vector produc with Reduction

```c
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#define N       (1024*1024*16)

__device__ int tmpC[N];

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
void initialData(int *arr, int size)
{
    float tmp;
    time_t t;
    srand((unsigned)time(&t));  // seed
    for(int i=0;i<size;i++){
        tmp=(float)(10.0f*rand()/RAND_MAX);
        arr[i]=(int)(tmp);
    }
}
int VecProdOnCPU(int *A, int *B, int const size)
{
    int tmp=0;
    for(int i=0;i<size;i++)
        tmp += A[i]*B[i];
    return tmp;
}

__global__ void VecProdOnGPU(int *A, int *B, int *g_odata, int const size)
{
    unsigned int tid=threadIdx.x;
    unsigned int idx=blockIdx.x*blockDim.x+threadIdx.x;

    tmpC[idx]=A[idx]*B[idx];
    __syncthreads();
    int *idata=tmpC+blockIdx.x*blockDim.x;

    // boundary check
    if(idx>=size) return;

    // in-place reduction in global mem
    for(int stride=blockDim.x/2;stride>0;stride>>=1)
    {
        if(tid<stride)
            idata[tid]+=idata[tid+stride];
            __syncthreads();
    }
    // write result for this block to global mem
    if(tid==0) g_odata[blockIdx.x]=idata[0];
}

int main(void)
{
    int cpu_result, gpu_result;

    // initialize
    int size = N;
    printf("vector length : %d\n", size);

    // execution configuration
    int blocksize=512;
    dim3 block(blocksize, 1);
    dim3 grid((size+block.x-1)/block.x,1);

    // allocate host memory
    size_t bytes=size*sizeof(int);
    int *h_A = (int*)malloc(bytes);
    int *h_B = (int*)malloc(bytes);
    int *tmp_A = (int*)malloc(bytes);
    int *tmp_B = (int*)malloc(bytes);
    int *h_AB=(int*)malloc(grid.x*sizeof(int));

    // allocate device memory
    int *d_A, *d_B, *d_AB;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_AB, grid.x*sizeof(int));

    initialData(h_A, size);
    initialData(h_B, size);
    memcpy(tmp_A, h_A, bytes);
    memcpy(tmp_B, h_B, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // cpu calculate
    double iStart=cpuTimer();
    cpu_result=VecProdOnCPU(tmp_A, tmp_B, size);
    double ElapsedTime=cpuTimer()-iStart;
    printf(" Result on CPU : %d, Elapsed Time %f sec\n", cpu_result,ElapsedTime);

    /********** GPU ***********/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    VecProdOnGPU<<<grid, block>>>(d_A,d_B,d_AB,size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ETime;
    cudaEventElapsedTime(&ETime, start, stop);
    cudaMemcpy(h_AB, d_AB, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_result=0;
    for(int i=0;i<grid.x;i++) gpu_result += h_AB[i];
    printf(" Result on GPU : %d, Elapsed Time %f sec\n", gpu_result,ETime*1e-3f);

    cudaEventDestroy(start),  cudaEventDestroy(stop);

    free(h_A), free(h_B), free(tmp_A), free(tmp_B), free(h_AB);
    cudaFree(d_A), cudaFree(d_B), cudaFree(d_AB);

    return 0;
}
```
```sh
 vector length : 16777216
 Result on CPU : 478182126, Elapsed Time 0.056005 sec
 Result on GPU : 478182126, Elapsed Time 0.000441 sec
```

# Using shared memory
```c
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
inline void CHECK(const cudaError_t error)
{
    if(error !=cudaSuccess)
    {
        fprintf(stderr, "Error: %s:%d, ", __FILE__,__LINE__);
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}
double cpuTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
void initialData(float *arr, const int size)
{
    time_t t;
    srand((unsigned int)time(&t));
    for(int i=0;i<size;i++)
        arr[i]=(float)(rand())/RAND_MAX;
}
void checkResult(float *host, float *gpu, const int N)
{
    double epsilon = 1.0e-8;
    bool match=1;
    for(int i=0;i<N;i++)
    {
        if(abs(host[i]-gpu[i])>epsilon)
        {
            match=0;
            printf("Matrices do not match!\n");
            printf("host %10.7f, gpu %10.7f at current %d\n", host[i], gpu[i], i);
            break;
        }
    }
    if(match)printf("Matrices match.\n");
}
void MatMulOnCPU(float *A, float *B, float *C, const int nrows, const int ncols)
{
    float sum;
    for(int i=0;i<nrows;i++)
    {
        for(int j=0;j<nrows;j++)
        {
            sum = 0.0f;
            for(int k=0;k<ncols;k++)
            {
                sum += A[i*ncols+k]*B[k*nrows+j];
            }
            C[i*nrows+j]=sum;
        }
    }
}
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread idx
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA*BLOCK_SIZE*by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin +wA-1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first-submatrix of B processed by the block
    int bBegin = BLOCK_SIZE*bx;
    // Step size used to interate through the sub-matrix of B
    int bStep = BLOCK_SIZE*wB;

    // Csub is used to store the element of the block sub-matrix
    // that is compute dby the thread;
    float Csub = 0.0f;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for(int a = aBegin, b=bBegin; a<=aEnd;  a+=aStep, b+=bStep)
    {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx]=A[a+wA*ty+tx];
        Bs[ty][tx]=B[b+wB*ty+tx];
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes on element of the blocksub-matrix
#pragma unroll
        for(int k=0;k<BLOCK_SIZE;++k)
            Csub += As[ty][k]*Bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    int c = wB*BLOCK_SIZE*by + BLOCK_SIZE*bx;
    C[c+wB*ty + tx] =Csub;
}

int main()
{
    double Start, ElapsedTime;
    int block_size=32;
    int nrows=10*block_size;
    int ncols=20*block_size;
    float *MatA, *MatB, *MatC, *gpu_MatC;
    float *d_MatA, *d_MatB, *d_MatC;

    int size = nrows*ncols;
    /********** ON CPU **************/
    MatA=(float*)malloc(size*sizeof(float));
    MatB=(float*)malloc(size*sizeof(float));
    MatC=(float*)malloc(nrows*nrows*sizeof(float));
    gpu_MatC=(float*)malloc(nrows*nrows*sizeof(float));

    initialData(MatA, size);
    initialData(MatB, size);

    Start=cpuTimer();
    MatMulOnCPU(MatA, MatB, MatC, nrows, ncols);
    ElapsedTime=cpuTimer()-Start;
    printf("Elapsed Time on CPU: %f\n",ElapsedTime);
    /********************************/

    /********** ON GPU *****************/
    cudaMalloc((void**)&d_MatA, size*sizeof(float));
    cudaMalloc((void**)&d_MatB, size*sizeof(float));
    cudaMalloc((void**)&d_MatC, nrows*nrows*sizeof(float));

    Start=cpuTimer();
    // copy host memory to device
    cudaMemcpy(d_MatA, MatA, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, MatB, size*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(block_size, block_size);     // sub-matrix dimension
    dim3 grid((nrows+block.x-1)/block.x, (nrows+block.y-1)/block.y);        // submatrix dimension of C
    matrixMulCUDA<32><<<grid,block>>>(d_MatC, d_MatA, d_MatB, ncols, nrows);
    cudaMemcpy(gpu_MatC,d_MatC, nrows*nrows*sizeof(float), cudaMemcpyDeviceToHost);
    ElapsedTime=cpuTimer()-Start;
    printf("Elapsed Time on GPU : %f\n",ElapsedTime);

    checkResult(MatC, gpu_MatC,nrows*nrows);

    free(MatA), free(MatB), free(MatC),     free(gpu_MatC);
    cudaFree(d_MatA),       cudaFree(d_MatB),       cudaFree(d_MatC);
    cudaDeviceReset();
    return 0;
}
```
```sh
Elapsed Time on CPU: 0.335707
Elapsed Time on GPU : 0.002112
Matrices match.
```
