sudo apt update
sudo apt upgrade -y
sudo apt autoremove
wget https://developer.download.nvidia.com/compute/cuda/13.3.0/local_installers/cuda_13.3.0_610.43.02_linux.run
sudo sh cuda_13.3.0_610.43.02_linux.run
nvcc --version
nvidia-smi