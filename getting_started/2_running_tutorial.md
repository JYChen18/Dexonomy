# Code Tutorial
This tutorial provides a step-by-step guide for using the code and exploring its key features.

## Template Preparation
To convert annotated files into valid grasp templates, run:
```bash
python -m dexonomy.main task=anno2temp hand=allegro
```
- Results are saved to: `init_template_dir` (default: `assets/hand/allegro/init_template`) defined in [dexonomy/config/base.yaml](https://github.com/JYChen18/Dexonomy/blob/main/dexonomy/config/base.yaml#L23)

## Grasp Synthesis
Grasp synthesis consists of three core tasks: `syn_obj`, `syn_hand`, and `syn_test`.

### 1. `syn_obj`: Optimize Object Pose
This task samples and optimizes the object pose while keeping the hand fixed. It uses a single initial template on one GPU:
```bash
python -m dexonomy.main task=syn_obj hand=allegro exp_name=first_try template_name=fingertip_small 'gpu_list=[0]' task.object.cfg_num=20
```
- Valid results are saved to: `init_dir` (default: `output/first_try_allegro/initialization`)
- Logs are saved to: `log_dir` (default: `output/first_try_allegro/log/syn_obj_0/main.log`)


### 2. `syn_hand`: Refine Hand Pose
This task refines the hand's palm pose and joint positions (qpos) in simulation while keeping the object fixed. Currently, it only supports MuJoCo on CPU.
```bash
python -m dexonomy.main task=syn_hand hand=allegro exp_name=first_try
```
- Valid results are saved to: `grasp_dir` (default: `output/first_try_allegro/graspdata`)
- Logs are saved to: `log_dir` (default: `output/first_try_allegro/log/syn_hand_0/main.log`) 

#### Debugging Options
```bash
python -m dexonomy.main task=syn_hand hand=allegro exp_name=first_try skip=False template_name=fingertip_small obj_name=floating_core_bottle_523cddb320608c09a37f3fc191551700_scale005 data_name=0_1_grasp debug_render=True hydra.verbose=true 
```

- `debug_render=True`: Render the refinement process and save GIFs to `debug_dir` (default: `output/first_try_allegro/debug`). (Note: if there is an OpenGL-related error, try to first run `export MUJOCO_GL=egl`)
- `debug_viewer=True`: Open the MuJoCo viewer. (Note: not available on headless devices). 
- `skip=False`: Re-generate existing graspdata files.
- `template_name`, `obj_name`, and `data_name`: Specify which initialization files to process. Each argument is optional and can be used independently.
- `hydra.verbose=true`: Print detailed debug info, such as failure reasons.

All these options are configurable via [dexonomy/config/base.yaml](https://github.com/JYChen18/Dexonomy/blob/main/dexonomy/config/base.yaml)

### 3. `syn_test`: Test Synthesized Grasps
This task evaluates the synthesized grasps in simulation. Currently, it only supports MuJoCo on CPU.

```bash
python -m dexonomy.main task=syn_test hand=allegro exp_name=first_try
```
- Successful grasps are saved to: `succ_dir` (default: `output/first_try_allegro/succgrasp`)
- New grasp templates are saved to: `new_template_dir` (default: `output/first_try_allegro/new_template`)
- The same debugging options as in `syn_hand` are also available here.

### 4. `dexonomy.script`: Run All Three Tasks Together
You can run all above three tasks (`syn_obj`, `syn_hand`, `syn_test`) together using:  
```bash
python -m dexonomy.script hand=allegro exp_name=first_try 'template_name=[fingertip_small,fingertip_mid,fingertip_large]' 'gpu_list=[5,6,7]'
```
- `syn_obj` will distribute multiple templates across GPUs in `gpu_list` and run in parallel.
- `syn_hand` and `syn_test` will automatically check and run as soon as there are unprocessed data from previous stage.

#### Customize Configurations
You can override configuration parameters per task like:
```bash
python -m dexonomy.script hand=allegro exp_name=first_try 'template_name=[fingertip_small,fingertip_mid,fingertip_large]' 'gpu_list=[5,6,7]' +syn_obj.object.cfg_num=20 '+syn_hand.grasp.qp_filter.miu_coef=[0.5, 0.02]'
```


## Visualization
Two visualization modes are available: `vis_usd` and `vis_3d`.

### 1. `vis_usd`: Visualize All Grasps in One USD File
The USD file can be opened by [usdview](https://docs.omniverse.nvidia.com/usd/latest/usdview/index.html).
```bash
python -m dexonomy.main task=vis_usd hand=allegro exp_name=first_try task.max_num=20 task.data_type=init task.check_success=False
```
- `task.max_num`: Limit number of data to visualize.
- `task.data_type`: Specifies which data directory to visualize. 
    - `init`: Initializations in `init_dir` (output of `syn_obj`)
    - `grasp`: Refined grasps in `grasp_dir` (output of `syn_hand`)
    - `succ` (default): Successful grasps in `succ_dir` (output of `syn_test`)
    - `new_template`: Newly generated templates in `new_template_dir` (output of `syn_test`)
- `task.check_success`: Filters the data to visualize
    - `True`: Only include successful data.
    - `False`: Only include failed data.
    - `None` (default):  Include all samples regardless of success
- `template_name`, `obj_name`, and `data_name`: Specify which files to run as before.

### 2. `vis_3d`: Visualize Individually in OBJ Formats
```bash
python -m dexonomy.main task=vis_3d hand=allegro exp_name=first_try task.data_type=init task.object.contact=True task.hand.contact=True 
```
- This mode supports to visualize contacts and hand collision skeleton.
- The same debugging options as in `vis_usd` are also available here.

## Statistics Calculation
Run the following command *at any time* to compute statistics:
```bash
python -m dexonomy.main task=stat hand=allegro exp_name=first_try 
```
- Output will be printed to screen and saved to: `log_dir` (Default: `output/first_try_allegro/log/stat_0/main.log`).
