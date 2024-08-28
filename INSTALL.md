Installation and running tested on (fresh) Ubuntu 22.04 LTS with RTX 3060 - 12 GB.
It should be possible to run everything by copy-pasting below commands

# Install libraries

```shell
sudo apt install nvidia-cuda-toolkit
sudo apt install gcc-10 g++-10 gcc g++ unzip
```

```shell
conda create -n spacetime python=3.7.13 -y
conda activate spacetime
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y

```

```shell
git clone git@github.com:remmel/SpacetimeGaussians.git --recursive
# git clone https://github.com/remmel/SpacetimeGaussians.git --recursive
cd SpacetimeGaussians

# export TORCH_CUDA_ARCH_LIST="7.0" # if compiling on non gpu server
export CC=/usr/bin/gcc-10 # if "error: parameter packs not expanded with ‘...’" or "UserWarning: There are no g++ version bounds defined for CUDA version 11.5"
export CXX=/usr/bin/g++-10
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

pip install -e thirdparty/mmcv -v
```

```shell

pip uninstall opencv-python # when is installed opencv-python??? conda list | grep opencv
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
cd ../../SpacetimeGaussians

# Preprocess
#sudo apt-get install xvfb # on server, needed even with opencv-headless
#xvfb-run python script/pre_n3d.py --videopath ../dataset/Neural3D/flame_steak --downscale 2 
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

# Train - undistorted
python train.py --gtmask 1 --config configs/im_undistort_lite/02_Flames.json --model_path log/02_Flames_undist/01 --source_path ../dataset/immersive/02_Flames_undist/colmap_0 

# Train - distorted - not enough GPU
cp configs/im_view/04_Truck/pickview.pkl ../dataset/immersive/02_Flames_dist/
PYTHONDONTWRITEBYTECODE=1 python train_imdist.py --eval --config configs/im_distort_lite/02_Flames.json --model_path log/02_Flames/dist --source_path ../dataset/immersive/02_Flames_dist/colmap_0 
```

# Troubleshooting

## OSError: [Errno 24] Too many open files
`ulimit -n 1000000`
https://askubuntu.com/questions/1182021/too-many-open-files

## Could not load the Qt platform plugin "xcb"
```
QObject::moveToThread: Current thread (0x55cbf9c4eec0) is not the object's thread (0x55cbf9dd54c0).
Cannot move to target thread (0x55cbf9c4eec0)

qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/remmel/anaconda3/envs/feature_splatting2/lib/python3.7/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb, eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx.
```

```
export QT_QPA_PLATFORM=offscreen
```

using `xvfb` does not seems the work
TODO try `pip uninstall opencv-python` and `pip install opencv-python-headless`