# fingertip_small,fingertip_mid,fingertip_large
exp_name=bulb
hand=allegro
tmpl_name=1_Large_Diameter
obj_name=sem_LightBulb_8338a18d589c26d21c648623457982d0
echo "${exp_name}_${hand}, tmpl_name=${tmpl_name}, obj_name=${obj_name}"

# dexrun op=init hand=$hand exp_name=$exp_name tmpl_name=$tmpl_name 'init_gpu=[0,1,2,3]' "op.object.cfg_path=${asset_root}/object/DGN_5k/scene_cfg/${obj_name}/floating/*.npy"
dexrun op=grasp_admm hand=$hand exp_name=$exp_name
# dexrun op=eval hand=$hand exp_name=$exp_name
# dexrun op=v3d hand=$hand exp_name=$exp_name