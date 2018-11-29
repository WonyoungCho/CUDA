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

## C로 작성된 파일을 Cuda로 compile할 때
```c
nvcc -arch=sm_70 -x cu hello.c -o a
```

## Batch file
```batch
#!/bin/sh
#SBATCH -J gpu_05
#SBATCH --time=00:05:00
#SBATCH -p dual_v100_node
#SBATCH -o result.out
##SBATCH -e result.err
#SBATCH --gres=gpu:1
srun ./a
```
