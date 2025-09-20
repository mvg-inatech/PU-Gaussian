# PU-Gaussian: Point Cloud Upsampling using 3D Gaussian Representation

Official implementation of **PU-Gaussian: Point Cloud Upsampling using 3D Gaussian Representation**, accepted at the **ICCV 2025 e2e3D Workshop**.

## Authors

**Mahmoud Khater**<sup>1,2</sup>, **Mona Strauss**<sup>1,2</sup>, **Philipp von Olshausen**<sup>2</sup>, **Alexander Reiterer**<sup>1,2</sup>

<sup>1</sup> University of Freiburg, Germany  
<sup>2</sup> Fraunhofer IPM, Freiburg, Germany

## Installation

### Clone the repository
```bash
git clone https://github.com/your-username/PU-Gaussian.git
cd PU-Gaussian
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Compile the submodules

Install the required submodules for point cloud operations:

```bash
cd pointnet2_ops_lib && pip install .
cd ../pointops && pip install .
cd ../utils/chamfer3d && pip install .
```

### Compile the evaluation code for metric calculation (optional)

To calculate the CD, HD and P2F metrics, you need to install the CGAL library (please follow the [PU-GAN](https://github.com/liruihui/PU-GAN) repo) and virtual environment of PU-GCN (please follow the [PU-GCN](https://github.com/guochengqian/PU-GCN) repo) first. And then you also need to compile the `evaluation_code` folder.
```bash
cd evaluation_code
bash compile.sh
```

These commands are tested on an ubuntu system.

## Data Preparation

### Option 1: Quick Start with Pre-processed Test Data

You can directly download the test point clouds and run our inference on them using the model saved in `pretrained_model`.

Download the [PU-GAN test meshes](https://drive.google.com/open?id=1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC) and [PU1K dataset](https://drive.google.com/drive/folders/1k1AR_oklkupP8Ssw6gOrIve0CmXJaSH3?usp=sharing), then unzip them and put them into the `data/PU-GAN` folder and `data/PU1K` folder respectively.


### PU-GAN test point cloud generation

Since the PU-GAN dataset only provides mesh files for test, we first generate test point clouds by Poisson disk sampling.

```bash
# 4X, generated files will be saved at ./data/PU-GAN/test_pointcloud/input_2048_4X by default
python prepare_pugan.py --mode test --input_pts_num 2048 --gt_pts_num 8192
# 16X, generated files will be saved at ./data/PU-GAN/test_pointcloud/input_2048_16X by default
python prepare_pugan.py --mode test --input_pts_num 2048 --gt_pts_num 32768
```

where `input_pts_num` denotes the point number of input low-res point cloud, `gt_pts_num` denotes the point number of ground truth high-res point cloud, and you can modify these two arguments to obtain various upsampling rates or input sizes.

You can also use the `noise_level` argument to generate the noisy low-res input.
```bash
# 4X, noise_level=0.01, generated files will be saved at ./data/PU-GAN/test_pointcloud/input_2048_4X_noise_0.01 by default
python prepare_pugan.py --mode test --input_pts_num 2048 --gt_pts_num 8192 --noise_level 0.01
```

### Option 2: Complete Dataset Generation (for training and evaluation)

#### PU-GAN Dataset Preparation

1. **Download PU-GAN raw meshes**  
   Download the raw meshes from [https://drive.google.com/file/d/1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC/view](https://drive.google.com/file/d/1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC/view) and extract them to `data/PU-GAN`.

2. **Process meshes to point clouds**  
   ```bash
   python prepare_pugan.py --pt train --input_pts_num 2048 --gt_pts_num 40960
   ```
   This will save the point clouds in `data/PU-GAN/train_pointcloud`.

3. **Generate training dataset**  
   ```bash
   python generate_dataset.py --dataset pugan --save_dir data/PU-GAN/train
   ```

#### PU1K Dataset Preparation

The PU1K dataset consists of PU-GAN data and meshes from ShapeNet. In order to generate the same dataset as our approach, PU-GAN dataset must be processed first.

1. **Download PU1K raw meshes**  
   Download the raw meshes from [https://drive.google.com/file/d/1tnMjJUeh1e27mCRSNmICwGCQDl20mFae/view?usp=drive_link](https://drive.google.com/file/d/1tnMjJUeh1e27mCRSNmICwGCQDl20mFae/view?usp=drive_link) and extract them to `data/PU1k_raw_meshes`.

2. **Process training set**  
   ```bash
   python prepare_pu1k.py --pt train --input_pts_num 2048 --gt_pts_num 40960
   ```
   This will save the point clouds in `data/pu1k/train_pointcloud`.

3. **Generate PU1K dataset**  
   ```bash
   python generate_dataset.py --dataset pu1k --save_dir data/PU1k/train
   ```


The final file structure of `data` folder is shown as follow:

```
data  
├───PU-GAN
│   ├───test # test mesh file
│   ├───test_pointcloud # generated test point cloud file
│   │     ├───input_2048_16X
│   │     ├───input_2048_4X
│   │     ├───input_2048_4X_noise_0.01
│   │     ...
│   ├───train_pointcloud # processed training point clouds
│   └───train
│   │     └───PUGAN_x20.h5
├───PU1K
│   ├───test
│   │     ├───input_1024
│   │     ├───input_2048
│   │     ...
│   ├───train_pointcloud # processed training point clouds
│   └───train
│   │     └───pu1k_x20.h5
└───PU1k_raw_meshes # raw meshes for PU1K dataset
```

## Quick Start

We have provided the pretrained models in the `pretrained_model` folder, so you can directly use them to reproduce the results.

### Testing with pretrained models

* PU-GAN
```bash
# 4X, upsampled point clouds will be saved at ./pretrained_model/pugan/test/4X
python test.py --dataset pugan --test_input_path ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048/ --test_gt_path ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192/ --ckpt .pretrained_model/pu_gau2_pu1k_Best.pth --save_dir results/PU-GAN/4x --up_rate 4

# 16X, upsampled point clouds will be saved at ./pretrained_model/pugan/test/16X
python test.py --dataset pugan --test_input_path ./data/PU-GAN/test_pointcloud/input_2048_16X/input_2048/ --test_gt_path ./data/PU-GAN/test_pointcloud/input_2048_16X/gt_32768/ --ckpt pretrained_model/pu_gau2_pu1k_Best.pth --save_dir results/PU-GAN/16x --up_rate 16
```

* PU1K
```bash
# 4X upsampling
python test.py --dataset pu1k --test_input_path ./data/PU1K/test/input_2048/input_2048 --test_gt_path ./data/PU1K/test/input_2048/gt_8192 --ckpt ./pretrained_model/pu_gau2_pu1k_Best.pth --save_dir results/PU1k/4x --up_rate 4
```

### Evaluation

The upsampled point clouds are saved at `./result/[dataset]/test/[save_dir]`. Then you can utilize the following commands to calculate metrics.

```bash
# take 4X upsampled results for example
cd evaluation_code
python write_eval_script.py --dataset pugan --upsampled_pcd_path ../pretrained_model/pugan/test/4X/
bash eval_pugan.sh
```

### Training from scratch

Add the dataset directory (`pugan` or `pu1k`) to `config.py`, then run:
```bash
python train.py
```

## Citation

```bibtex
@inproceedings{khater2025puguassian,
  title={PU-Gaussian: Point Cloud Upsampling using 3D Gaussian Representation},
  author={Khater, Mahmoud and Strauss, Mona and von Olshausen, Philipp and Reiterer, Alexander},
  booktitle={ICCV 2025 e2e3D Workshop},
  year={2025}
}
```

*Note: BibTeX citation will be updated once the paper is published.*



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact the authors or open an issue in this repository.

## Acknowledgments

This repository is heavily dependent on [PU-GCN](https://github.com/guochengqian/PU-GCN), [Grad-PU](https://github.com/yunhe20/Grad-PU), [PU-CRN](https://github.com/wanruzhao/PU-CRN), and [RepKPU](https://github.com/qhanghu/RepKPU) repositories. Do not forget to cite them if this work is beneficial for you.
