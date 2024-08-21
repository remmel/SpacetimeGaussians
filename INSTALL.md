Installation and running tested on Ubuntu 22.04 LTS with RTX 3060 - 12 GB.
It should be possible to run everything by copy-pasting below commands

# Install libraries

```shell
sudo apt install nvidia-cuda-toolkit
sudo apt install gcc-10 g++-10
```

```shell
conda create -n spacetime python=3.7.13 -y
conda activate spacetime
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y

```

```shell
# Install for Gaussian Rasterization (Ch9) - Ours-Full
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9
# Install for Gaussian Rasterization (Ch3) - Ours-Lite
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch3
# Install for Forward Full - Ours-Full (speed up testing, mlp fused, no sigmoid)
pip install thirdparty/gaussian_splatting/submodules/forward_full
# Install for Forward Lite - Ours-Lite (speed up testing)
pip install thirdparty/gaussian_splatting/submodules/forward_lite

# install simpleknn
pip install thirdparty/gaussian_splatting/submodules/simple-knn
```

```shell
# Install MMCV for CUDA KNN, used for init point sampling, reduce number of points when sfm points are too many
cd thirdparty
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -e .
cd ../../
```

```shell
pip install opencv-python-headless natsort scipy kornia tqdm plyfile scikit-image
```

## Libs for preprocessing
```shell
pip install Pillow
conda config --set channel_priority false
conda install colmap -c conda-forge
```

# Download, Preprocess and Train

## Neural3D - flame_steak

```shell
# Download
cd ..
mkdir -p dataset/Neural3D && cd dataset/Neural3D/
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_steak.zip
unzip flame_steak.zip
cd ../../SpacetimeGaussian

# Preprocess
python script/pre_n3d.py --videopath ../dataset/Neural3D/flame_steak

# Train
python train.py --eval --config configs/n3d_lite/flame_steak.json --model_path log/flame_steak_lite  --source_path ../dataset/Neural3D/flame_steak/colmap_0

# Test
python test.py --eval --skip_train --valloader colmapvalid --configpath configs/n3d_lite/flame_steak.json --model_path log/flame_steak_lite --source_path ../dataset/Neural3D/flame_steak/colmap_0
```

# Immersive Deepview video dataset - 02_Flames

```shell
cd ..
mkdir -p dataset/immersive && cd dataset/immersive/
wget https://storage.googleapis.com/deepview_video_raw_data/02_Flames.zip
unzip 02_Flames.zip
cd ../../SpacetimeGaussian

# Preprocess
python script/pre_immersive_distorted.py --videopath ../dataset/immersive/02_Flames
python script/pre_immersive_undistorted.py --videopath ../dataset/immersive/02_Flames

cp configs/im_view/04_Truck/pickview.pkl ../dataset/immersive/02_Flames_dist/

# Train - undistorted
python train.py --gtmask 1 --config configs/im_undistort_lite/02_Flames.json --model_path log/02_Flames_undist/01 --source_path ../dataset/immersive/02_Flames_undist/colmap_0 

# Train - distorted - not enough GPU
PYTHONDONTWRITEBYTECODE=1 python train_imdist.py --eval --config configs/im_distort_full/02_Flames.json --model_path log/02_Flames/dist --source_path ../dataset/immersive/02_Flames_dist/colmap_0 
```