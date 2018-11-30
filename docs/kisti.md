# Specification
```c
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

# Login
```sh
======================== KAT User Notification ========================
 * Any unauthorized attempts to use/access the system can be
   investigated and prosecuted by the related Act

 * Beta service
  - 2017-10-16 09:00 ~ 2017-12-15 18:00 (1st)
  - 2018-05-02 10:00 ~ 2018-07-31 15:00 (2nd)
  - 2018-08-01 10:00 ~ 2018-11-30 17:00 (3rd)

 * Hardware
  - tesla[01-30] HOST: Intel Xeon Ivy Bridge (E5-2670) 2 socket,128GB DDR3
  - bigmem: Intel Xeon Westmere (E7-4870) 4 socket,512GB DDR3
  - bigmem2: Intel Xeon Broadwell (E7-4830) 4 socket,768GB DDR4
  - skl[01-10]: Intel Xeon Skylake (Gold 6140) 2 socket,192GB DDR4

 * Software Stack
  - tesla[01-30],bigmem: Centos 6.4,MLNX-OFED 2.2,Lustre 1.8.9
  - bigmem2,skl[01-10]: Centos 6.6,MLNX-OFED 2.4,Lustre 1.8.9

 * Login, Debugging Node
  - Login: login-tesla[01-02]
  - Debugging: skl10

 * Policy on User Jobs
  - Wall Time Clock Limit: 24h (common)
                   |Max Running| Max Active Jobs |Max Available|   GPU
                   |   Jobs    |(running+waiting)|     Gpus    |
    - - - - - - - -|- - - - - -| - - - - - - - - |- - - - - - -|- - - - -
  - tesla[01-17]   |     4     |        4        |      5      | K40  1EA
  - tesla[18-28]   |     8     |        8        |     16      | V100 2EA
  - tesla[29,30]   |     2     |        2        |      2      | V100 1EA
  - bigmem,bigmem2 |     2     |        4        |      5      |    -
  - skl[01-09]     |     2     |        4        |      5      |    -

 * Preventive Maintenance
  - 2018-09-12 09:00 ~ 21:00
  - 2018-10-11 09:00 ~ 21:00

 * Failures of Shared Filesystem
  - 2018-09-27 11:00 ~ 17:30 (/scratch2)

 * Failures of Login/Debugging nodes

 * More detail can be found on http://helpdesk.ksc.re.kr
```

# Partition
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
