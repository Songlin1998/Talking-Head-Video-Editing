iters=".....tar"
names="female_asian"
datasets="female_asian"
near=0.5361079454421998
far=1.1361079454421996
path="dataset/finetune_models/${names}/${iters}"
datapath="dataset/${datasets}/0"
bc_type="torso_imgs"
suffix="train"
python NeRFs_lip/render_with_novel_audio_lip.py --need_torso True --config dataset/test_config.txt --expname ${names}_${suffix} --expname_finetune ${names}_${suffix} --render_only --ft_path ${path} --datadir ${datapath} --bc_type ${bc_type} --near ${near} --far ${far}