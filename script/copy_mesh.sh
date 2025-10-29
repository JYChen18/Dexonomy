obj_name=mug
obj=core_mug_bf2b5e941b43d030138af902bc222a59
path=assets/object/DGN_5k/processed_data/${obj}/mesh/simplified.obj
target=/mnt/home/ruanliangwang/fast-graspd/assets/ycb/${obj_name}.obj
# print the command
echo "cp ${path} ${target}"
cp ${path} ${target}