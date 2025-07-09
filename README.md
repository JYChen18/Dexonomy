# Dexonomy

Official implementation of *Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy* [RSS 2025]. 


[![Project Page](https://img.shields.io/badge/Project-Page-Green.svg)]()
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-orange.svg)]()

## Overview

<div style="text-align: center;">
    <img src="img/teaser.png" width=100% >
</div>

Our algorithm synthesizes **contact-rich, penetration-free, and physically plausible** dexterous grasps for:
- Any grasp type
- Any object
- Any articulated hand

All starting from just **one** human-annotated template *per hand and grasp type*.

#### Supported Features
- Hands: Shadow, Allegro, Leap, MANO, Inspire, Unitree_G1
- Object Scenes: Single (floating & tabletop), Clustered (light clutter)
- Object Assets: Rigid objects (ShapeNet & Objaverse), Articulated objects (PartNet)
- Physics Simulator: MuJoCo
- Grasping Trajectory Synthesis: cuRobo integration


## Installation
Our code is tested on **Ubuntu 20.04** with **NVIDIA GeForce RTX 3090** GPUs and **Intel(R) Xeon(R) Platinum 8255C** CPUs.

```bash
git submodule update --init --recursive --progress

conda create -n dexonomy python=3.10 
conda activate dexonomy

conda install pytorch==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia 

# [Optional] Only necessary for grasping trajectory synthesis
sudo apt install git-lfs
cd third_party/BODex
git lfs pull
pip install -e . --no-build-isolation
cd ...

# Install this package
pip install -e .
```

## Quick Start
### 1. Prepare Object Assets 

Download our pre-processed object assets `DGN_5k_processed.zip` from [Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy), and organize the unzipped folders as below. 
```
assets/object/DGN_5k
|- processed_data
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- scene_cfg
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- valid_split
|  |- all.json
|  |_ ...
```
Alternatively, you can pre-process your own object assets using [MeshProcess](https://github.com/JYChen18/MeshProcess).

### 2. Synthesize Grasps
Run the following commands to synthesize grasps for a specific grasp type:
```bash
python -m dexonomy.main op=tmpl  # Generate initial template from human annotations
python -m dexonomy.script 'template_name=[1_Large_Diameter]'
```

## Tutorial
For a detailed walkthrough of **template annotation** and **code usage**, please refer to [getting_started](https://github.com/JYChen18/Dexonomy/tree/main/getting_started).

## License

This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

## Citation

If you find this work useful for your research, please consider citing:
```
@article{chen2025dexonomy,
        title={Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy},
        author={Chen, Jiayi and Ke, Yubin and Peng, Lin and Wang, He},
        journal={Robotics: Science and Systems},
        year={2025}
      }
```