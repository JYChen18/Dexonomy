# Dexonomy

Official implementation of *Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy*. 

[![Project Page](https://img.shields.io/badge/Project-Page-Green.svg)]()
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-orange.svg)]()

<div style="text-align: center;">
    <img src="img/teaser.png" width=100% >
</div>

For **any grasp type**, **any object**, and **any articulated robotic hand**, our algorithm quickly synthesizes *contact-rich, penetration-free, and physically plausible* dexterous grasps, starting from only **one** human-annotated template per hand and grasp type.


## Installation
```bash
git submodule update --init --recursive --progress

conda create -n dexonomy python=3.10 
conda activate dexonomy

conda install pytorch==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install -e .
```

## Run
1. **Prepare object assets**. Download our pre-processed object assets `DGNv2_processed.zip` from [here](https://disk.pku.edu.cn/link/AA0FA86ED9AC5F4EC2B3BB6AF4100BEEA6) and organize the unzipped folders as below. Alternatively, new object assets can be pre-processed using [MeshProcess](https://github.com/JYChen18/MeshProcess).
```
assets/object/DGN_obj_v2
|- processed_data
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- valid_split
|  |- all.json
|  |_ ...
```

2. **Synthesize grasps** for a specific grasp type.
```
python -m dexonomy.main task=anno2temp   # generate initial template from human annotations
python -m dexonomy.script 'template_name=[1_Large_Diameter]'
```
More detailed instructions can be found in `example`.