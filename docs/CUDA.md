# alias
```bash
alias cs='cp /home01/kedu05/ex/sun.sh ./'
alias nv='nvcc -arch=sm_70 -o a'
alias rn='sbatch sun.sh'
alias re='cat result.out'
alias sq='squeue'
alias lg='sacct'
```
---

# Install
- <https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-centos-7-linux>
- <https://linuxconfig.org/how-to-install-nvidia-cuda-toolkit-on-centos-7-linux>

# If warning is appeared during `bash NVIDIA-Linux-x86_64-*`
```sh
Error: You apear to be running an X server; please exit X before installing. ...
```
```sh
$ ps ax | grep X
$ sudo kill -9 PID_number
```

# Compute Capability x.x
```sh
nvcc -arch=sm_xx
```
- <https://developer.nvidia.com/cuda-gpus>

# Reference
- <http://webedu.ksc.re.kr>
- CUDA C Programming Guide
- CUDA RUNTIME API
- <https://developer.nvidia.com/educators>
- CUDA C Programming
- CUDA 병렬프로그래밍
- CUDA Fortran for Scientists and Engineers
- NVIDIA TESLA V100 GPU ARCHITECTURE(white paper)