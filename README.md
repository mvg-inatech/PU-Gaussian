# PU-Gaussian: Point Cloud Upsampling using 3D Gaussian Representation

Official implementation of **PU-Gaussian: Point Cloud Upsampling using 3D Gaussian Representation**, accepted at **ICCV 2025 e2e3D Workshop**.

## Authors
**Mahmoud Khater**<sup>1,2</sup>, **Mona Strauss**<sup>1,2</sup>, **Philipp von Olshausen**<sup>2</sup>, **Alexander Reiterer**<sup>1,2</sup>  
<sup>1</sup> University of Freiburg, Germany  
<sup>2</sup> Fraunhofer IPM, Freiburg, Germany  

---

## Installation

Clone the repo and install dependencies:
```bash
git clone https://github.com/mvg-inatech/PU-Gaussian.git
cd PU-Gaussian
pip install -r requirements.txt
```

Compile submodules:
```bash
cd pointops && pip install .
cd utils/chamfer3d && pip install .
```

---

## Data Preparation

### Option 1: Quick Test Setup

You can directly download the test point clouds and run inference using the models in `pretrained_model`.

- Download the [PU-GAN dataset](https://drive.google.com/open?id=1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC)  
- Download the [PU1K dataset](https://drive.google.com/file/d/1oTAx34YNbL6GDwHYL2qqvjmYtTVWcELg/view?usp=drive_link)  
- Extract them into `data/PU-GAN` and `data/PU1K` respectively.

- **PU-GAN**: Requires preprocessing from meshes.  
- **PU1K**: No preprocessing required; just place files in the correct folder structure.

#### PU-GAN Test Data Preparation
Generate point clouds from meshes using Poisson disk sampling:
```bash
# 4X upsampling
python prepare_pugan.py --mode test --input_pts_num 2048 --gt_pts_num 8192

# 16X upsampling
python prepare_pugan.py --mode test --input_pts_num 2048 --gt_pts_num 32768
```

Optional: add noise to low-res input:
```bash
python prepare_pugan.py --mode test --input_pts_num 2048 --gt_pts_num 8192 --noise_level 0.01
```

#### PU1K Test Data
Simply unzip the dataset into `data/PU1K` with the required folder structure.

---

## Testing

Use pretrained models in `pretrained_model/` to reproduce our results.

**PU-GAN:**
```bash
# 4X
python test.py --dataset pugan   --test_input_path ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048/   --test_gt_path ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192/   --ckpt pretrained_model/pu_gaussian_pugan_Best.pth   --save_dir results/PU-GAN/4x --up_rate 4

# 16X
python test.py --dataset pugan   --test_input_path ./data/PU-GAN/test_pointcloud/input_2048_16X/input_2048/   --test_gt_path ./data/PU-GAN/test_pointcloud/input_2048_16X/gt_32768/   --ckpt pretrained_model/pu_gaussian_pugan_Best.pth   --save_dir results/PU-GAN/16x --up_rate 16
```

**PU1K:**
```bash
python test.py --dataset pu1k   --test_input_path ./data/PU1K/test/input_2048/input_2048   --test_gt_path ./data/PU1K/test/input_2048/gt_8192   --ckpt pretrained_model/pu_gaussian_pu1k_Best.pth   --save_dir results/PU1K/4x --up_rate 4
```


---

### Option 2: Full Dataset (for training & reproducing results)

#### PU-GAN
```bash
python prepare_pugan.py --mode train --input_pts_num 2048 --gt_pts_num 40960
python generate_dataset.py --dataset pugan --save_dir data/PU-GAN/train
```

#### PU1K
The PU1K dataset includes PU-GAN data and ShapeNet meshes. Since PU1K depends on PU-GAN preprocessing, process PU-GAN first.

Download the raw meshes from [this link](https://drive.google.com/file/d/1tnMjJUeh1e27mCRSNmICwGCQDl20mFae/view?usp=drive_link) and extract them to `data/PU1K_raw_meshes`.

```bash
python prepare_pu1k.py --mode train --input_pts_num 2048 --gt_pts_num 40960
python generate_dataset.py --dataset pu1k --save_dir data/PU1K/train
```

*Note: This step may take some time.*

---

## Expected Data Folder Structure

```
data  
├── PU-GAN
│   ├── test                 # raw test mesh files
│   ├── test_pointcloud      # generated test point clouds
│   │    ├── input_2048_16X
│   │    ├── input_2048_4X
│   │    ├── input_2048_4X_noise_0.01
│   │    ...
│   ├── train_pointcloud     # processed training point clouds
│   └── train
│        └── PUGAN_x20.h5
├── PU1K
│   ├── test
│   │    ├── input_1024
│   │    ├── input_2048
│   │    ...
│   ├── train_pointcloud     # processed training point clouds
│   └── train
│        └── pu1k_x20.h5
└── PU1K_raw_meshes          # raw meshes for PU1K dataset
```

---


## Training

To train from scratch, edit `config.py` with the dataset path and run:
```bash
# pugan training
python train.py --dataset pugan

# pu1k training
python train.py --dataset pu1k
```
---
## Evaluation
For more information on evaluation, please refer to [ [Grad-PU](https://github.com/yunhe20/Grad-PU) ]


## Citation

```bibtex
@inproceedings{khater2025puguassian,
  title={PU-Gaussian: Point Cloud Upsampling using 3D Gaussian Representation},
  author={Khater, Mahmoud and Strauss, Mona and von Olshausen, Philipp and Reiterer, Alexander},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year={2025},
   note={ICCV 2025 e2e3D Workshop, to appear}
}
```
*BibTeX will be updated once the paper is officially published.*

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions or issues, please open an issue in this repository or contact the authors.

---

## Acknowledgments

This project builds upon [PU-GCN](https://github.com/guochengqian/PU-GCN), [Grad-PU](https://github.com/yunhe20/Grad-PU), [PU-CRN](https://github.com/wanruzhao/PU-CRN), and [RepKPU](https://github.com/qhanghu/RepKPU). Please cite them if you use this work.
