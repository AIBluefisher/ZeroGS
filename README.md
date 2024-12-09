# <img src="./assets/imgs/zero_gs_logo.png" style="width:45px;height:auto;margin-bottom:-8px"> ZeroGS: Training 3D Gaussian Splatting from Unposed Images

[[Project Page](https://aibluefisher.github.io/ZeroGS/) | [arXiv](https://arxiv.org/pdf/2411.15779)]

---------------------------

## 🛠️ Installation

Install the conda environment of ZeroGS.

```sh
conda create -n zero_gs python=3.9
conda activate zero_gs
cd ZeroGS/scripts
./scripts/env/install.sh
```

**git hook for code style checking**:
```sh
pre-commit install --hook-type pre-commit
```


## 🚀 Features

- [x] Release [ACE0](https://nianticlabs.github.io/acezero) implementation
    - [ ] Incorporate [GLACE](https://github.com/cvg/glace) into ACE0
- [ ] Release our customized 3D Gaussian Splatting module
    - [ ] Incorporate [Scaffold-GS](https://city-super.github.io/scaffold-gs)
    - [ ] Incorporate [DOGS](https://github.com/aibluefisher/dogs)
- [ ] Release ZeroGS implementation


## 📋 Train & Eval ACE0

We aim at providing a framework which makes it easy to implement your own neural implicit module with this codebase and since this project starts before the code releasing of ACE0, we re-implement ACE0 based on our codebase.

### ⌛Train ACE0

Before training ACE0, please download the [pretrained feature encoder](https://github.com/nianticlabs/ace/blob/main/ace_encoder_pretrained.pt) from ACE, and put it under the folder  `ZeroGS/conerf/model/scene_regressor`.

```bash
conda activate zero_gs
visdom -port=9000 # Keep the port the same as the `visdom_port` provided in the configuration file
cd ZeroGS/scripts/train
./train_ace_zero.sh 0 ace_early_stop_resize_2k_anneal mipnerf360 ace
```
We use `visdom` to visualize the camera pose predictions during training. You can access `https://localhost:9000` to view it.

### 📊 Evaluate ACE0

```bash
conda activate zero_gs
cd ZeroGS/scripts/eval
./eval_ace_zero.sh 0 ace_early_stop_resize_2k_anneal mipnerf360 ace
```
Metrics file and camera poses will be recorded in `eval/val/` folder. Point clouds are recorded in the `eval/val/ACE0_COLMAP` (This folder also contains the model files in COLMAP formats) in `.ply` format.

### 🔢 Hyper Parameters for training ACE0

All the parameters related to train ACE0 are provided the configuration file in `config/ace/mipnerf360.yaml`. Most of the parameters can be kept the same as in this configuration file. However, the parameters listed below need to be adjusted accordingly to obtain better performance:
```yaml
trainer:
  # We can use less iterations for the `garden` scene (i.e. 2000).
  min_iterations_per_epoch: 5000

pose_estimator:
  # Change this to a larger threshold (3000) for the 'garden` scene of the mipnerf360 dataset.
  min_inlier_count: 2000 # minimum number of inlier correspondences when registering an image
```

A larger value in `min_iterations_per_epoch` can make the mapping more accurate, but also lead to longer training time.


## ✏️ Cite

If you find this project useful for your research, please consider citing our paper:
```bibtex
@inproceedings{yuchen2024zerogs,
    title={ZeroGS: Training 3D Gaussian Splatting from Unposed Images},
    author={Yu Chen, Rolandos Alexandros Potamias, Evangelos Ververas, Jifei Song, Jiankang Deng, Gim Hee Lee},
    booktitle={arXiv},
    year={2024},
}
```

## 🙌 Acknowledgements

This work is built upon [ACE](https://nianticlabs.github.io/ace/), [DUSt3R](https://github.com/naver/dust3r), and [Spann3R](https://hengyiwang.github.io/projects/spanner). We sincerely thank all the authors for releasing their code.

## 🪪 License

Copyright © 2024, Chen Yu.
All rights reserved.
Please see the [license file](LICENSE) for terms.
