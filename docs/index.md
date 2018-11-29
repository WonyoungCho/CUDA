# CUDA


## Module
```sh
Currently Loaded Modulefiles:
  1) compiler/gcc-4.9.4   2) cuda/9.0
```

# alias
```bash
alias cs='cp /home01/kedu05/ex/sun.sh ./'
alias cm='nvcc -arch=sm_70 -o a'
alias rn='sbatch sun.sh'
alias re='cat result.out'
alias sq='squeue'
alias lg='sacct'
```

## Reference
- <http://webedu.ksc.re.kr>
- CUDA C Programming Guide
- CUDA RUNTIME API
- <https://developer.nvidia.com/educators>
- CUDA C Programming
- CUDA 병렬프로그래밍
- CUDA Fortran for Scientists and Engineers
- NVIDIA TESLA V100 GPU ARCHITECTURE(white paper)

## Install

- <https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-centos-7-linux>
- <https://linuxconfig.org/how-to-install-nvidia-cuda-toolkit-on-centos-7-linux>

## Archtecture
```sh
nvcc -arch=sm_##
```
- <https://developer.nvidia.com/cuda-gpus>
