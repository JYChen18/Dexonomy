# dexsyn n_worker=128 exp_name=exp_A_pre tmpl_name=[fingertip_small,fingertip_mid] hand=allegro +init.object.n_cfg=-1 +init_gpu=[0,1,2,3,4,5,6,7] +init.epoch=1

# dexsyn_admm n_worker=128 exp_name=exp_A_final tmpl_name=[fingertip_small,fingertip_mid] hand=allegro +init.object.n_cfg=-1 +init_gpu=[0,1,2,3,4,5,6,7] +init.epoch=1

dexsyn_admm exp_name=tmp_01_25 tmpl_name=[fingertip_small,fingertip_mid] hand=allegro +init.object.n_cfg=400 +init_gpu=[0,1,2,3,4,5,6,7] +init.epoch=1 +grasp_admm.grasp.rho_admm=1e2 +grasp_admm.grasp.lr_obj=1e-3 ste

dexsyn_admm exp_name=exp_A_rho_1e3 tmpl_name=[fingertip_small,fingertip_mid] hand=allegro +init.object.n_cfg=4000 +init_gpu=[0,1,2,3,4,5,6,7] +init.epoch=1 +grasp_admm.grasp.rho_admm=1e3

dexsyn_admm exp_name=exp_A_rho_1e1 tmpl_name=[fingertip_small,fingertip_mid] hand=allegro +init.object.n_cfg=400 +init_gpu=[0,1,2,3,4,5,6,7] +init.epoch=1 +grasp_admm.grasp.rho_admm=1e1