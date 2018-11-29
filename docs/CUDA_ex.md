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
