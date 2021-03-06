**PuCUDA**는 **Python**을 기본으로 하여 **GPU** 에서 작업이 작동하도록 하는 방식이다. **GPU** 계산이 되는 **kernel** 부분의 명령어는 **c언어** 로 작성해야 한다.

# Module

- CUDA 를 사용하기 위한 첫 번째는, **kernel**로 보내기 위한 과정들을 자동으로 초기화 하는 작업이다.
```python
import pycuda.autoinit
```

- 다음으로는 **GPU kernel**에서 작동할 명령이 들어가는 모듈을 불러오는 일이다.
```python
from pycuda.compiler import SourceModule
```

- Python을 사용하다보면 `numpy array`를 많이 사용하는데, 이것을 GPU에서도 거의 동일하게 사용할 수 있다.
```python
from pycuda import gpuarray
```

따라서 프로그래밍 할 때는 위 세개의 모듈을 불러오고 시작한다.
```python
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
```

# Indexing thread

작업을 **GPU kernel**로 넘겨주고 **kernel**에서 데이터를 다루려면, 계산이 이루어지는 **thread**의 **indexing** 이 중요하다.


# Kernel function
```python
mod = SourceModule("""
  __global__ void kernel(int *a)
  {
    int idx = threadIdx.x;
    a[idx] += 10;
  }
  """)
```

# Kernel 호출
```python
kernel = mod.get_function("kernel")

kernel(a_gpu, block=(10,1,1), grid=(1,1))
```

# GPUArray

- **CUDA C** 처럼 `pycuda.driver`를 불러와 메모리를 할당하고 GPU에 넘겨주는 방식으로 프로그래밍 할 수 있다.
```python
import numpy
import pycuda.driver as cuda

mod = SourceModule("""
  __global__ void kernel(int *a)
  {
    int idx = threadIdx.x;
    a[idx] += 10;
  }
  """)

kernel = mod.get_function("kernel")

a = numpy.zeros(10).astype(numpy.int)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

kernel(a_gpu, block=(10,1,1), grid=(1,1))

a_result = numpy.empty_like(a)
cuda.memcpy_dtoh(a_result, a_gpu)

a_gpu.free()

print(a_result)
```
위에서는 CPU에서 만든 array를 GPU에 메모리를 할당(`cuda.mem_alloc`)하고, GPU로 보내주는 작업(`cuda.memcpy_htod`)을 하고, 작업이 끝나면 해제하였는데, `gpuarray`를 사용하면 GPU 메모리에 바로 올려 사용하고 해제도 된다.
```python
from pycuda import gpuarray

mod = SourceModule("""
  __global__ void kernel(int *a)
  {
    int idx = threadIdx.x;
    a[idx] += 10;
  }
  """)

kernel = mod.get_function("kernel")

a_gpu = gpuarray.zeros(10,int)

kernel(a_gpu, block=(10,1,1), grid=(1,1))

print(a_gpu.get())
```
