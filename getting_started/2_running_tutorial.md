# Code Tutorial
This tutorial provides a step-by-step guide for using the code and exploring its key features.

## Template Preparation
To convert annotated files into valid grasp templates, run:
```bash
dexrun op=tmplhand=allegro
```
- Results are saved to: `init_template_dir` (default: `assets/hand/allegro/init_template`) defined in [dexonomy/config/base.yaml](https://github.com/JYChen18/Dexonomy/blob/main/dexonomy/config/base.yaml#L23)

## Grasp Synthesis
Grasp synthesis consists of three core ops: `init`, `grasp`, and `test`.

### 1. `init`: Optimize Object Pose
This op samples and optimizes the object pose while keeping the hand fixed. It uses a single initial template on one GPU:
```bash
dexrun op=init hand=allegro exp_name=first_try template_name=fingertip_small 'init_gpu=[0]' op.object.n_cfg=20
```
- Valid results are saved to: `init_dir` (default: `output/first_try_allegro/initialization`)
- Logs are saved to: `log_dir` (default: `output/first_try_allegro/log/init_0/main.log`)


### 2. `grasp`: Refine Hand Pose
This op refines the hand's palm pose and joint positions (qpos) in simulation while keeping the object fixed. Currently, it only supports MuJoCo on CPU.
```bash
dexrun op=grasp hand=allegro exp_name=first_try
```
- Valid results are saved to: `grasp_dir` (default: `output/first_try_allegro/grasp_data`)
- Logs are saved to: `log_dir` (default: `output/first_try_allegro/log/grasp_0/main.log`) 

#### Debugging Options
```bash
dexrun op=grasp hand=allegro exp_name=first_try skip=False debug_name=core_bottle_523cddb320608c09a37f3fc191551700 debug_render=True hydra.verbose=true 
```

- `debug_render=True`: Render the refinement process and save GIFs to `debug_dir` (default: `output/first_try_allegro/debug`). (Note: if there is an OpenGL-related error, try to first run `export MUJOCO_GL=egl`)
- `debug_view=True`: Open the MuJoCo viewer. (Note: not available on headless devices). 
- `skip=False`: Re-generate existing grasp data files.
- `debug_name=core_bottle_523cddb320608c09a37f3fc191551700`: Specify which initialization files to process. Only those path names that  include `debug_name` would be used.
- `hydra.verbose=true`: Print detailed debug info, such as failure reasons.

All these options are configurable via [dexonomy/config/base.yaml](https://github.com/JYChen18/Dexonomy/blob/main/dexonomy/config/base.yaml)

### 3. `test`: Test Synthesized Grasps
This op evaluates the synthesized grasps in simulation. Currently, it only supports MuJoCo on CPU.

```bash
dexrun op=test hand=allegro exp_name=first_try
```
- Successful grasps are saved to: `succ_grasp_dir` (default: `output/first_try_allegro/succ_grasp`)
- New grasp templates are saved to: `new_template_dir` (default: `output/first_try_allegro/new_template`)
- The same debugging options as in `grasp` are also available here.

### 4. `dexonomy.script`: Run All Three ops Together
You can run all above three ops (`init`, `grasp`, `test`) together using:  
```bash
dexsyn hand=allegro exp_name=first_try 'template_name=[fingertip_small,fingertip_mid,fingertip_large]' 'init_gpu=[5,6,7]'
```
- `init` will distribute multiple templates across GPUs in `init_gpu` and run in parallel.
- `grasp` and `test` will automatically check and run as soon as there are unprocessed data from previous stage.

#### Customize Configurations
You can override configuration parameters per op like:
```bash
dexsyn hand=allegro exp_name=first_try 'template_name=[fingertip_small,fingertip_mid,fingertip_large]' 'init_gpu=[5,6,7]' +init.object.n_cfg=20 '+grasp.grasp.qp_filter.miu_coef=[0.5, 0.02]'
```


## Visualization
Two visualization modes are available: `vis_usd` and `vis_3d`.

### 1. `vis_usd`: Visualize All Grasps in One USD File
The USD file can be opened by [usdview](https://docs.omniverse.nvidia.com/usd/latest/usdview/index.html).
```bash
dexrun op=vusd hand=allegro exp_name=first_try op.n_max=20 op.data=init op.succ=False
```
- `op.n_max`: Limit number of data to visualize.
- `op.data`: Specifies which data directory to visualize. 
    - `init`: Initializations in `init_dir` (output of `init`)
    - `grasp`: Refined grasps in `grasp_dir` (output of `grasp`)
    - `succ_grasp` (default): Successful grasps in `succ_grasp_dir` (output of `test`)
    - `new_template`: Newly generated templates in `new_template_dir` (output of `test`)
- `op.succ`: Filters the data to visualize
    - `True`: Only include successful data.
    - `False`: Only include failed data.
    - `None` (default):  Include all samples regardless of success
- `debug_name`: Specify which files to run as before.

### 2. `vis_3d`: Visualize Individually in OBJ Formats
```bash
dexrun op=v3d hand=allegro exp_name=first_try op.data=init op.object.contact=True op.hand.contact=True 
```
- This mode supports to visualize contacts and hand collision skeleton.
- The same debugging options as in `vis_usd` are also available here.

## Statistics Calculation
Run the following command *at any time* to compute statistics:
```bash
dexrun op=stat hand=allegro exp_name=first_try 
```
- Output will be printed to screen and saved to: `log_dir` (Default: `output/first_try_allegro/log/stat_0/main.log`).
