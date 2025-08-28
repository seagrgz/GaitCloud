### This is the repository for WACV2025 paper [GaitCloud: Leveraging Spatial-Temporal Information for LiDAR-Base Gait Recognition with A True-3D Gait Representation](https://openaccess.thecvf.com/content/WACV2025/papers/Zhang_GaitCloud_Leveraging_Spatial-Temporal_Information_for_LiDAR-Base_Gait_Recognition_with_A_WACV_2025_paper.pdf).

# Requirements:
    python 3.9.19  
    pytorch 2.3.1  
    cuda 12.1

# Get started:
## Data preparation
#### 1. Download the open-access dataset [SUSTech1K](https://openxlab.org.cn/datasets/noahshen/SUSTech1K).  
#### 2. Unzip SUSTech1K-pkl.zip into 'dataset/SUSTech1K'. It should be like:  
    dataset
       |--SUSTech1K
       |      |--SUSTech1K-Released-pkl
       |      |      |--subject id
       |      |      |      |--...
#### 3. Set the parameters in **SUSTech1K-voxelize.py** as follow:  
    name = 'any name for voxelized dataset',  
    frame_num = 20,                         # Number of frames used to create a sample.  
    box_size = [1.25,1.25,2],               # Voxelization box size in [height, length, width].  
    res = 0.03125,                          # Voxelize resolution.  
    dilution = 1,                           # Rate of reducing vertical point density.  
    noise = 0,                              # Add Gausian noise to point coordinates.  
    compress = 0,                           # Compress rate on width dimension.  
    tfusion = False                         # If create temporal expended samples.  
#### 4. Set `data_root = 'dataset name'` in `create_set.py`, than run `SUSTech1K-voxelize.py` and `create_set.py` sequentially.  
This will transform the raw point cloud data into GaitCloud representation and create train set and test set in `'dataset name'/tmp`.  
By default, runtime model at the last 10 epochs will be evauated with samples in the test set and the results will be recorded in the experiment folder created during training.
#### \(Optional\) 5. Set `lidargait` in `create_set.py` True to prepare the data for [LidarGait](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_LidarGait_Benchmarking_3D_Gait_Recognition_With_Point_Clouds_CVPR_2023_paper.pdf)   reproduction. \(After step 4 processed.\)  
#### \(Optional\) 6. Set `frame_num = 10` and `tfusion = True` in `SUSTech1K-voxelize.py`. Repeat step 3, 4 to create temporal expanded GaitCloud.  

## Train the models  
&nbsp;&nbsp;&nbsp;&nbsp;cd to the root folder of this repository and run `sh tool/train.sh SUSTech1K [exp_name]`.  
&nbsp;&nbsp;&nbsp;&nbsp;Set **exp_name**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;to **GaitCloud_repro** for the model in this paper.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;to **LidarGait_repro** for our reproduction of LidarGait.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;to **LidarGait3D_repro** for 3D-LidarGait with temporal expanded GaitCloud.  
This will start the model training with config files at `config/SUSTech1K/exp_name`.  
#### \*\* You may need to modify the data root in the config files to meet the customized dataset name. Make sure you already have according data in `datasetname/tmp`. \*\*

## Evaluation
You can check the graphs in the experiment folder which created by the training program, or run `./view_result.py -t 'folder name'` to get a summary.

##  Please contact me at shhhlusi3@gmail.com if you meet any problems.
