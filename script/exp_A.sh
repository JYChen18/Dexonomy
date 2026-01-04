# dexsyn_admm n_worker=128 exp_name=exp_A_1 tmpl_name=[fingertip_small,fingertip_mid] hand=allegro +init.object.n_cfg=-1 +init_gpu=[0,1,2,3,4,5,6,7] +init.epoch=1

# dexsyn_admm n_worker=128 exp_name=exp_A_admm tmpl_name=[fingertip_small,fingertip_mid] hand=allegro +init.object.n_cfg=-1 +init_gpu=[0,1,2,3,4,5,6,7] +init.epoch=1

dexsyn_admm n_worker=128 exp_name=exp_A tmpl_name=[fingertip_small,fingertip_mid] hand=allegro +init.object.n_cfg=-1 +init_gpu=[0,1,2,3,4,5,6,7] +init.epoch=1

# dexsyn n_worker=128 exp_name=exp_A_old tmpl_name=[fingertip_small,fingertip_mid] hand=allegro +init.object.n_cfg=-1 +init_gpu=[0,1,2,3,4,5,6,7] +init.epoch=1