# Dexonomy

Official implementation of *Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy*. 

[![Project Page](https://img.shields.io/badge/Project-Page-Green.svg)]()
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-orange.svg)]()

<div style="text-align: center;">
    <img src="img/teaser.png" width=100% >
</div>

For **any** grasp type, object, and articulated robotic hand, our algorithm quickly synthesizes *contact-rich, penetration-free, and physically plausible* dexterous grasps, starting from only one human-annotated template per hand and grasp type.


## Installation
```
git submodule update --init --recursive --progress

conda create -n dexonomy python=3.10 
conda activate dexonomy

conda install pytorch==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install -e .
```

2. Run
```
# Run three tasks in a single command line
python script.py       

# Run each task individually. 
python main.py task=csample     
python main.py task=mjopt     
python main.py task=mjtest     
```

3. Visualization
```
# Get quantitative numbers
python main.py task=stat

# Save rendered images in USD format
python main.py task=usd 

# Save raw 3D meshes in OBJ format
python main.py task=visd debug_template=1_LargeDiameter debug_obj=core_bottle_3b0e35ff08f09a85f0d11ae402ef940e

```