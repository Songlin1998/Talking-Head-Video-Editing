#!/bin/bash
iters="....tar"
names="test"
datasets="test"
near=0.7371961951255799
far=1.3371961951255797
path="dataset/finetune_models/${names}/${iters}"
datapath="dataset/${datasets}/0"
bc_type="torso_imgs"
suffix="val"
python NeRFs/render_only.py --need_torso True --config dataset/test_config.txt --expname ${names}_${suffix} --expname_finetune ${names}_${suffix} --render_only --ft_path ${path} --datadir ${datapath} --bc_type ${bc_type} --near ${near} --far ${far}