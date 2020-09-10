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

**Cuda driver**
- <https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-centos-7-linux>
- Driver : <https://www.nvidia.com/en-us/drivers/unix/>
- Download : <https://www.nvidia.com/Download/index.aspx>
```
$ wget https://us.download.nvidia.com/XFree86/Linux-x86_64/450.66/NVIDIA-Linux-x86_64-450.66.run
$ sudo yum groupinstall "Development Tools"
$ sudo yum install -y kernel-devel epel-release
$ sudo yum install -y dkms
$ sudo emacs /etc/default/grub
GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet nouveau.modeset=0" # nouveau.modeset=0 must be in this line.
$ sudo grub2-mkconfig -o /boot/grub2/grub.cfg # BIOS
$ sudo grub2-mkconfig -o /boot/efi/EFI/centos/grub.cfg # EFI

$ sudo reboot now

$ sudo systemctl isolate multi-user.target
$ sudo bash NVIDIA-Linux-x86_64-*
YES
```

**Cuda toolkit**
- <https://linuxconfig.org/how-to-install-nvidia-cuda-toolkit-on-centos-7-linux>
- Toolkit : <https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/>
```
$ wget https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-10.2.89-1.x86_64.rpm
$ sudo rpm -i cuda-repo-*.rpm
$ sudo yum install cuda
$ emacs ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
$ source ~/.bashrc
$ nvcc --version
$ nvidia-smi
```


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
