# Introduction
---

## Terminology
- GPU에서 실행되는 함수를 커널(kernel)함수라고 부른다.
- 커널 호출에 의해 생성된 모든 스레드를 그리드(Grid)라고 한다.
- 그리드는 많은 스레드 블록으로 구성된다.
- 스레드 블록은 스레드들의 묶음이다.
- 스레드 블록의 스레드들은 32개의 스레드로 구성된 워프(Warp) 단위로 실행된다.
- 즉, 워프는 SM에서 실행의 단위가 된다.

## CUDA Runtime API
```c
__host__cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
```
> : 지정된 디바이스의 정보를 반환
> - `prop` : 지정된 디바이스에 대한 속성
> - `device` : 속성을 얻고자 하는 디바이스 번호

## CUDA Kernel Function
Function | Run on | Call from| Return type
:-:|:-:|:-|:-:
`__global__` | Device | Host <br> Device in up to Compute capability 3.5 | void
`__device__` | Device | Device <br> 그리드와 블록을 지정할 수 없다. |
`__host__` | Host | Host | optional

## CUDA Kernel 호출
``cu
function_name<<<grid,block>>>(argument list);
```
> - `grid` : 블록의 개수인 그리드의 크기
> - `block` : 각 블록에서 스레드의 개수인 블록의 크기
```


## Partition
```sh
$ sinfo
PARTITION        AVAIL  TIMELIMIT  NODES  STATE NODELIST
single_k40_node     up 1-00:00:00      2  down* tesla[11,17]
single_k40_node     up 1-00:00:00      9  alloc tesla[01-08,10]
single_k40_node     up 1-00:00:00      1   idle tesla09
single_k40_node     up 1-00:00:00      5   down tesla[12-16]
dual_v100_node      up 1-00:00:00     10  alloc tesla[18-23,25-28]
dual_v100_node      up 1-00:00:00      1   idle tesla24
single_v100_node    up 1-00:00:00      2   idle tesla[29-30]
skl_node            up 1-00:00:00      4  alloc skl[04-05,08-09]
skl_node            up 1-00:00:00      5   idle skl[01-03,06-07]
bigmem_node         up 1-00:00:00      2   idle bigmem,bigmem2
```
```sh
$ sinfo -Nel
Thu Nov 29 09:58:25 2018
NODELIST   NODES        PARTITION       STATE CPUS    S:C:T MEMORY TMP_DISK WEIGHT AVAIL_FE REASON
bigmem         1      bigmem_node        idle   40   4:10:1 516860        0      1   (null) none
bigmem2        1      bigmem_node        idle   56   4:14:1 775143        0      1   (null) none
skl01          1         skl_node        idle   36   2:18:1 192000        0      1 Xeon6140 none
skl02          1         skl_node        idle   36   2:18:1 192000        0      1 Xeon6140 none
skl03          1         skl_node        idle   36   2:18:1 192000        0      1 Xeon6140 none
skl04          1         skl_node   allocated   36   2:18:1 192000        0      1 Xeon6140 none
skl05          1         skl_node   allocated   36   2:18:1 192000        0      1 Xeon6140 none
skl06          1         skl_node        idle   36   2:18:1 192000        0      1 Xeon6140 none
skl07          1         skl_node        idle   36   2:18:1 192000        0      1 Xeon6140 none
skl08          1         skl_node   allocated   36   2:18:1 192000        0      1 Xeon6140 none
skl09          1         skl_node   allocated   36   2:18:1 192000        0      1 Xeon6140 none
tesla01        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla02        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla03        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla04        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla05        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla06        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla07        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla08        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla09        1  single_k40_node        idle   20   2:10:1 129031        0      1 TeslaK40 none
tesla10        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla11        1  single_k40_node       down*   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla12        1  single_k40_node        down   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla13        1  single_k40_node        down   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla14        1  single_k40_node        down   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla15        1  single_k40_node        down   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla16        1  single_k40_node        down   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla17        1  single_k40_node       down*   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla18        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla19        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla20        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla21        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla22        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla23        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla24        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla25        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla26        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla27        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla28        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla29        1 single_v100_node        idle   20   2:10:1 100000        0      1 TeslaV10 none
tesla30        1 single_v100_node        idle   20   2:10:1 100000        0      1 TeslaV10 none
```

## Device
```c
$ cat DeviceQuery.cu
#include <cuda_runtime.h>
#include <stdio.h>
int main(void)
{
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        printf("Device : \"%s\"\n", deviceProp.name);

        int driverVersion = 0, runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("driverVersion : %d\n", driverVersion);
        printf("runtimeVersion : %d\n", runtimeVersion);
        printf("\tCUDA Driver Version / Runtime Version  %d.%d / %d.%d\n",
                        driverVersion/1000, (driverVersion%100)/10,
                        runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("\tCUDA Capability Major/Minor version number : %d.%d\n",
                        deviceProp.major, deviceProp.minor);
        printf("\tTotal amount of global memory : %.2f GBytes (%llu bytes)\n",
                        (float)deviceProp.totalGlobalMem/(pow(1024.0,3)),
                        (unsigned long long) deviceProp.totalGlobalMem);
        printf("\tGPU Clock rate :\t%.0f MHz(%0.2f GHz)\n",
                        deviceProp.clockRate*1e-3f, deviceProp.clockRate*1e-6f);
        printf("\tMemory Clock rate :\t%.0f Mhz\n", deviceProp.memoryClockRate*1e-3f);
        printf("\tMemory Bus Width :\t%d-bit\n", deviceProp.memoryBusWidth);
        if(deviceProp.l2CacheSize)
                printf("\tL2 Cache Size:\t%d bytes\n",deviceProp.l2CacheSize);
        printf("\tTotal amount of constant memory:\t%lu bytes\n",deviceProp.totalConstMem);
        printf("\tTotal amount of shared memory per block:\t%lu bytes\n",deviceProp.sharedMemPerBlock);
        printf("\tTotal number of registers available per block:\t%d\n",deviceProp.regsPerBlock);
        printf("\tWarp Size:\t%d\n",deviceProp.warpSize);
        printf("\tMaximum number of threads per multiprocessor:\t%d\n",deviceProp.maxThreadsPerMultiProcessor);
        printf("\tMaximum number of thread per block:\t%d\n",deviceProp.maxThreadsPerBlock);
        printf("\tMaximum sizes of each dimension of a block:\t%d x %d x %d\n",
                        deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("\tMaximum sizes of each dimension of a grid:\t%d x %d x %d\n",
                        deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
        exit(EXIT_SUCCESS);

        return 0;

}
```
```sh
$ sbatch sun.sh
==========================================
SLURM_JOB_ID = 19799
SLURM_NODELIST = tesla27
==========================================
Device : "Tesla V100-PCIE-16GB"
driverVersion : 9000
runtimeVersion : 9000
        CUDA Driver Version / Runtime Version  9.0 / 9.0
        CUDA Capability Major/Minor version number : 7.0
        Total amount of global memory : 15.77 GBytes (16936861696 bytes)
        GPU Clock rate :        1380 MHz(1.38 GHz)
        Memory Clock rate :     877 Mhz
        Memory Bus Width :      4096-bit
        L2 Cache Size:  6291456 bytes
        Total amount of constant memory:        65536 bytes
        Total amount of shared memory per block:        49152 bytes
        Total number of registers available per block:  65536
        Warp Size:      32
        Maximum number of threads per multiprocessor:   2048
        Maximum number of thread per block:     1024
        Maximum sizes of each dimension of a block:     1024 x 1024 x 64
        Maximum sizes of each dimension of a grid:      2147483647 x 65535 x 65535
```

## C로 작성된 파일을 Cuda로 compile할 때
```c
nvcc -arch=sm_70 -x cu hello.c -o a
```
