## File changes

The core changes in the `dexonomy` folder are as follows:

* `config/op/grasp_admm.yaml`
  * `step`: the steps of outer ADMM iterations
  * `substep`: simulation substep for Mujoco, the same as Dexonomy 
  * `step_obj`: inner steps of updating object contact points
  * `step_hand`: inner steps of updating hand contact points
  * `rho_admm`: $\rho$ in ADMM
  * `lr_obj`: learning rate of updating object contact points
  * `square_res`: whether use squared L2-norm or L2-norm of grasp metric. Now turn on for shadow hand, turn off for Allegro hand
  * `fix_hand_local_cp`: whether fix the hand contact points in local link frames
* `config/base.yaml`
  * `legacy_api`: to run on previous Dexonomy dataset
* `op/gen_grasp_admm.py`
  * the core algorithm of ADMM optimization
* `qp/qp_single.py`
  * `ContactQPTorch`: differentiable QP solver in torch
* `sim/mujoco_env.py`
  * `Mujoco_OptEnv:apply_contact_forces_no_normal`: used in the second stage of ADMM
* `script_admm.py`
  * the pipeline for generation with ADMM

## To Run

Run pipeline:
```sh
dexsyn_admm exp_name=test tmpl_name=[1_Large_Diameter,2_Small_Diameter,3_Medium_Wrap,4_Adducted_Thumb,5_Light_Tool,6_Prismatic_4_Finger,7_Prismatic_3_Finger,8_Prismatic_2_Finger] hand=shadow +init.object.n_cfg=100 +init_gpu=[0,1,2,3,4,5,6,7]
```

Run single operation:
```sh
dexrun op=grasp_admm 'exp_name=test' 'hand=shadow'
```


