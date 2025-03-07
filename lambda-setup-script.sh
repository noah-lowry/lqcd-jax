
#!/bin/sh
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
sh Miniconda3-latest-Linux-aarch64.sh
conda init
exec
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create -n lattice-qcd Python=3.11 ipykernel jupyter pip -y
conda activate lattice-qcd
conda install numpy scipy matplotlib tqdm -y
pip install -U "jax[cuda12]"
pip install jax-tqdm
