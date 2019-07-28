# Introduction

기본적인 개념과 구동 방식은 CUDA와 같다. Python 과 연계하여 GPU 프로그래밍을 하는 방식이다.

CUDA 를 사용하기 위한 첫 번째는, kernel로 보내기 위한 과정들을 자동으로 초기화 하는 작업이다.
```python
import pycuda.autoinit
```
