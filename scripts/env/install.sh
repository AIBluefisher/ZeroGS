# conda create -n zero_gs python=3.9
# conda activate zero_gs

# install pytorch
# Ref: https://pytorch.org/get-started/previous-versions/

# CUDA 11.7
# conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda

# Basic packages.
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg easydict \
            kornia lpips tensorboard visdom tensorboardX matplotlib plyfile trimesh h5py pandas \
            omegaconf PyMCubes Ninja pyransac3d einops pyglet pre-commit pylint GPUtil \
            open3d pyrender
pip install timm==0.6.7
pip install -U scikit-learn
pip install git+https://github.com/jonbarron/robust_loss_pytorch
pip install torch-geometric==2.4.0

conda install pytorch3d -c pytorch3d
conda install conda-forge::opencv
conda install pytorch-scatter -c pyg
conda remove ffmpeg --force

# Third-parties.

cd submodules/dsacstar
python setup.py install

cd ../../
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization

mkdir 3rd_party && cd 3rd_party

git clone https://github.com/cvg/sfm-disambiguation-colmap.git
cd sfm-disambiguation-colmap
python -m pip install -e .
cd ..

# HLoc is used for extracting keypoints and matching features.
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
cd ..

# Tiny-cuda-cnn & nerfacc
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# nerfacc
# pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.1_cu117.html
# or install the latest version
# pip install git+https://github.com/KAIR-BAIR/nerfacc.git
# To install a specified version:
# pip install nerfacc==0.3.5 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.1_cu117.html
pip install nerfacc==0.3.5 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu118.html

# Install CURope
cd croco/models/curope/
python setup.py build_ext --inplace
