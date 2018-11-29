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
$ cat sun.sh
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
$ sbatch sun.sh
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
## CUDA Kernel code
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
