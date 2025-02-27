# RSR-MJX

This repository contains the codebase for the paper "RSR: An Real-Sim-Real (RSR) Loop Framework for Generalizable Robotic Policy Transfer with Differentiable Simulation" submitted to IROS2025. In this work, we propose a framework that leverages real-world data to adjust the parameters of the differentiable simulator - MJX. 

## Setup Instructions

### 1. Create an environment
To begin, create and activate a new Conda environment:
```bash
conda create -n unienv python=3.10
conda activate unienv
```
### 2. Setup the dependencies
Create a file named ``requirements.txt`` and add the following content:
```bash
numpy==1.26.4
scipy==1.12.0
optax==0.2.4
brax==0.12.1
jax==0.4.29
nvidia-cudnn-cu12==9.1.0.70
nvidia-cuda-cupti-cu12
--find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
jaxlib==0.4.29+cuda12.cudnn91
mujoco==3.2.4
mujoco-mjx==3.2.4
```

Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Tune your environmental parameters in MJX with real-world data

### 
