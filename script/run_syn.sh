# rm -rf output/old_allegro
dexsyn exp_name=old tmpl_name=[fingertip_mid] hand=allegro +init.object.n_cfg=20 hydra.verbose=true

# rm -rf output/debug_allegro
# dexsyn_admm exp_name=debug tmpl_name=[fingertip_mid] hand=allegro +init.object.n_cfg=20 hydra.verbose=true

# dexsyn_admm exp_name=debug tmpl_name=[fingertip_mid] hand=allegro +init.object.n_cfg=1 hydra.verbose=true

# dexrun op=stat hand=allegro exp_name=old
# dexrun op=stat hand=allegro exp_name=new