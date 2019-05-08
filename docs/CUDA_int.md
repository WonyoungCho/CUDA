# Introduction
---

## Terminology
- GPU에서 실행되는 함수를 커널(kernel)함수라고 부른다.
- 커널 호출에 의해 생성된 모든 스레드를 그리드(Grid)라고 한다.
- 그리드는 많은 스레드 블록으로 구성된다.
- 스레드 블록은 스레드들의 묶음이다. (CUDA core = thread)
- 스레드 블록의 스레드들은 32개의 스레드로 구성된 워프(Warp) 단위로 실행된다.
- 즉, 워프는 SM에서 실행의 단위가 된다. (Stream Multiprocessor(SM) : GPU의 기본단위)


## Structure
아래와 같은 하위 관계를 가지고 있다.
> - Device : GPU를 가리킨다.
> - Grid : Device가 
> - Block
> - Thread

## Runtime API
```c
__host__cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
```
 : 지정된 디바이스의 정보를 반환

> - `prop` : 지정된 디바이스에 대한 속성
> - `device` : 속성을 얻고자 하는 디바이스 번호

```c
__host____device__cudaErro_t cudaMalloc(void **devPtr, size_t_size)
```
> - `devPtr` :  디바이스 메모리에 할당될 포인터
> - `size` : 요청되는 byte 단위의 할당크기

```c
__host____device__cudaErro_t cudaFree(void *devPtr)
```
> - `devPtr` : 해제될 메모리에 대한 디바이스 포인터

## Kernel Function
Function | Run on | Call from| Return type
:-:|:-:|:-|:-:
`__global__` | Device | Host <br> Device in up to Compute capability 3.5 | void
`__device__` | Device | Device <br> 그리드와 블록을 지정할 수 없다. |
`__host__` | Host | Host | optional

## Kernel 호출
```c
function_name<<<nBlock,nThread>>>(argument list);
```
> - `nBlock` : 블록의 개수
> - `nThread` : 각 블록에서 스레드의 개수


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
