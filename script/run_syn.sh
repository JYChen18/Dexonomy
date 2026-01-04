# dexsyn_admm n_worker=128 exp_name=admm_0 tmpl_name=[1_Large_Diameter,2_Small_Diameter,3_Medium_Wrap,4_Adducted_Thumb,5_Light_Tool,6_Prismatic_4_Finger,7_Prismatic_3_Finger,8_Prismatic_2_Finger] hand=shadow +init.object.n_cfg=-1 +init_gpu=[0,1,2,3,4,5,6,7]

# dexsyn_admm n_worker=128 exp_name=admm_1 tmpl_name=[9_Palmar_Pinch,10_Power_Disk,11_Power_Sphere,12_Precision_Disk,13_Precision_Sphere,14_Tripod,15_Fixed_Hook,16_Lateral] hand=shadow +init.object.n_cfg=-1 +init_gpu=[0,1,2,3,4,5,6,7]

# dexsyn_admm n_worker=128 exp_name=admm_2 tmpl_name=[17_Index_Finger_Extension,18_Extensior_Type,19_Distal,20_Writing_Tripod,21_Tripod_Variation,22_Parallel_Extension,23_Adduction_Grip,24_Tip_Pinch] hand=shadow +init.object.n_cfg=-1 +init_gpu=[0,1,2,3,4,5,6,7]

# dexsyn_admm n_worker=128 exp_name=admm_3 tmpl_name=[25_Lateral_Tripod,26_Sphere_4_Finger,27_Quadpod,28_Sphere_3_Finger,29_Stick,30_Palmar,31_Ring,32_Ventral] hand=shadow +init.object.n_cfg=-1 +init_gpu=[0,1,2,3,4,5,6,7]

# dexsyn_admm n_worker=128 exp_name=admm_4 tmpl_name=[33_Inferior_Pincer,fingertip_large,fingertip_middle,mug] hand=shadow +init.object.n_cfg=-1 +init_gpu=[0,1,2,3,4,5,6,7]

###########################

dexsyn_admm n_worker=128 exp_name=admm_01_03_0 tmpl_name=[1_Large_Diameter,2_Small_Diameter,3_Medium_Wrap,4_Adducted_Thumb,5_Light_Tool,6_Prismatic_4_Finger,7_Prismatic_3_Finger,8_Prismatic_2_Finger] hand=shadow +init.object.n_cfg=1000 +init_gpu=[0,1,2,3,4,5,6,7]

dexsyn_admm n_worker=128 exp_name=admm_01_03_1 tmpl_name=[9_Palmar_Pinch,10_Power_Disk,11_Power_Sphere,12_Precision_Disk,13_Precision_Sphere,14_Tripod,15_Fixed_Hook,16_Lateral] hand=shadow +init.object.n_cfg=1000 +init_gpu=[0,1,2,3,4,5,6,7]

dexsyn_admm n_worker=128 exp_name=admm_01_03_2 tmpl_name=[17_Index_Finger_Extension,18_Extensior_Type,19_Distal,20_Writing_Tripod,21_Tripod_Variation,22_Parallel_Extension,23_Adduction_Grip,24_Tip_Pinch] hand=shadow +init.object.n_cfg=1000 +init_gpu=[0,1,2,3,4,5,6,7]

dexsyn_admm n_worker=128 exp_name=admm_01_03_3 tmpl_name=[25_Lateral_Tripod,26_Sphere_4_Finger,27_Quadpod,28_Sphere_3_Finger,29_Stick,30_Palmar,31_Ring,32_Ventral] hand=shadow +init.object.n_cfg=1000 +init_gpu=[0,1,2,3,4,5,6,7]

dexsyn_admm n_worker=128 exp_name=admm_01_03_4 tmpl_name=[33_Inferior_Pincer,fingertip_large,fingertip_middle,mug] hand=shadow +init.object.n_cfg=1000 +init_gpu=[0,1,2,3,4,5,6,7]
