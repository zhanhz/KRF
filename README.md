# KRF
## Installation
- Install CUDA 10.1 / 10.2
- Set up python3 environment from requirement.txt:
  ```shell
  pip3 install -r requirement.txt 
  ```
- Install [apex](https://github.com/NVIDIA/apex):
  ```shell
  git clone https://github.com/NVIDIA/apex
  cd apex
  export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"  # set the target architecture manually, suggested in issue https://github.com/NVIDIA/apex/issues/605#issuecomment-554453001
  pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  cd ..
  ```
- Install [normalSpeed](https://github.com/hfutcgncas/normalSpeed), a fast and light-weight normal map estimator:
  ```shell
  git clone https://github.com/hfutcgncas/normalSpeed.git
  cd normalSpeed/normalSpeed
  python3 setup.py install --user
  cd ..
  ```
- Install tkinter through ``sudo apt install python3-tk``
- Compile for chamfer distance
  ``` 
  cd krf/utils/distance
  python setup.py install
  ```
- Install KNN-CUDA
  ```shell
  cd KNN-CUDA
  make
  make install
  ```
- Compile [RandLA-Net](https://github.com/qiqihaer/RandLA-Net-pytorch) operators:
  ```shell
  cd krf/models/RandLA/
  sh compile_op.sh
  ```

## Create Dataset
- **LineMOD:** Download the preprocessed LineMOD dataset from [onedrive link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yhebk_connect_ust_hk/ETW6iYHDbo1OsIbNJbyNBkABF7uJsuerB6c0pAiiIv6AHw?e=eXM1UE) or [google drive link](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) (refer from [DenseFusion](https://github.com/j96w/DenseFusion)). Unzip it and link the unzipped ``Linemod_preprocessed/`` to ``krf/datasets/linemod/Linemod_preprocessed``:
  ```shell
  ln -s path_to_unzipped_Linemod_preprocessed krf/dataset/linemod/
  ```
  Generate rendered and fused data following [raster_triangle](https://github.com/ethnhe/raster_triangle).

- **YCB-Video:** Download the YCB-Video Dataset from [PoseCNN](https://rse-lab.cs.washington.edu/projects/posecnn/). Unzip it and link the unzipped ```YCB_Video_Dataset``` to ```krf/datasets/ycb/YCB_Video_Dataset```:
  ```shell
  ln -s path_to_unzipped_YCB_Video_Dataset krf/datasets/ycb/
  ```
- You can download pretrained complete networks and generated data [here](https://pan.baidu.com/s/1VIIcRWbkFrABYc4IVChFhw?pwd=d46m ). 

  Then generate colored mesh point cloud for each objects by:
  ```
  python generate_color_pts.py
  ```
- **Generate FFB6D estimate results**

  Download pretrained model [FFB6D-LineMOD](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yhebk_connect_ust_hk/Ehg--MMyNdtLnAEurN0tm_MBQ8u_Lntrl42-BQeXO_8H8Q?e=HsZ2Yi), [FFB6D-YCB](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yhebk_connect_ust_hk/EW7a5w-ytftLgexIyXuIcjwB4o0dWo1hMteMNlA1zgM7Wg?e=UE1WJs), move it to ``train_log/linemod/checkpoints/`` or ``train_log/ycb/checkpoints/``. Then modify ``generate_ds.sh`` and generate estimate results by:
  ```shell
  bash generate_ds.sh
  ```

## Training
- To train the network on YCB Dataset, run the following command:
```shell
bash train_ycb_refine_pcn.sh
```
- To train the network on LineMOD Dataset, run the following command:
```shell  
# commands in train_lm_refine_pcn.sh
n_gpu=6
cls='ape'
#ckpt_mdl="/home/zhanhz/FFB6D/ffb6d/train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_REFINE_best.pth.tar"
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm_refine_pcn.py --gpus=$n_gpu --cls=$cls #-checkpoint $ckpt_mdl
# end

bash train_lm_refine_pcn.sh
```

## Evaluation
- To evaluate our method on YCB Dataset, run the following command:
```shell
python ycb_refine_test.py -gpu=0 -ckpt=CHECKPOINT_PATH -use_pcld -use_rgb
```
- To evaluate our method on Occlusion LineMOD Dataset, run the following command for one class:
```shell
python lm_refine_test.py -gpu=0 -ckpt=CHECKPOINT_PATH -cls='ape' -use_pcld -use_rgb
```
or evaluate all class by:

```shell
bash test_occ_icp.sh
```
