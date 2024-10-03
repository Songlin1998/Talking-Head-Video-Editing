import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


def stcs_cal(video_feats, stcs_path=None, dim=56):
    '''
    Use this if normalizing data is needed, but input shape is different in a batch
    Args:
        video_feats: numpy.ndarray [[T1, 53], [T2, 53], ...] 
        stcs_path: given statistics {min, max, mean, std}
    Return:
        res: stcs_dict, normalized data
    '''
    if stcs_path is None:
        # min, max, mean, std calculation
        mus, sigmas, mins, maxs = [], [], [], []
        # for i in range(dim):
        #     i_feat = [v[:, i] for v in video_feats]  # [[T1,] [T2,], ...] each is numpy.ndarray
        #     i_feat = np.concatenate(i_feat, axis=0)

        mins.append(video_feats.min())
        maxs.append(video_feats.max())
        mus.append(video_feats.mean())
        sigmas.append(video_feats.std())
        
        # store this if needed
        stcs_dict = {
                'min': mins,
                'max': maxs,
                'mean': mus,
                'std': sigmas
            }
    else:
        stcs_dict = np.load(stcs_path, allow_pickle=True).item()
        mus = stcs_dict['mean']
        sigmas = stcs_dict['std']
    
    # normalize data
    # normalized_feats = []
    # for f in video_feats:
    #     feat = (f - mus) / sigmas  # ([T,53] - [1,53]) / [1,53]
    #     normalized_feats.append(feat)
    normalized_feats = (video_feats-mus) /sigmas

    return stcs_dict, normalized_feats

def audio_stcs_cal(video_feats, stcs_path=None, dim=16):
    '''
    Use this if normalizing data is needed, but input shape is different in a batch
    Args:
        video_feats: numpy.ndarray [[T1, 53], [T2, 53], ...] 
        stcs_path: given statistics {min, max, mean, std}
    Return:
        res: stcs_dict, normalized data
    '''
    if stcs_path is None:
        # min, max, mean, std calculation
        mus, sigmas, mins, maxs = [], [], [], []
        # for i in range(29):
        #     i_feat = [v[:,:, i] for v in video_feats]  # [[T1,] [T2,], ...] each is numpy.ndarray
        #     i_feat = np.concatenate(i_feat, axis=0)

        mins.append(video_feats.min())
        maxs.append(video_feats.max())
        mus.append(video_feats.mean())
        sigmas.append(video_feats.std())
        
        # store this if needed
        stcs_dict = {
                'min': mins,
                'max': maxs,
                'mean': mus,
                'std': sigmas
            }
    else:
        stcs_dict = np.load(stcs_path, allow_pickle=True).item()
        mus = stcs_dict['mean']
        sigmas = stcs_dict['std']
    
    # normalize data
    # normalized_feats = []
    # for f in video_feats:
    #     feat = (f - mus) / sigmas  # ([T,53] - [1,53]) / [1,53]
    #     normalized_feats.append(feat)
    normalized_feats = (video_feats-mus) / sigmas

    return stcs_dict, normalized_feats

trans_t = lambda t : torch.Tensor([
    [1,0,0,t],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])) @ c2w
    return c2w

def pose_around(theta1, theta2 , c2w):
    #c2w = rot_theta(theta / 180.0 * np.pi).cpu() @ c2w
    c2w = trans_t(theta1).cpu() @ rot_theta(theta2 / 180.0 * np.pi).cpu() @ c2w
    return c2w


def load_audface_data_lip(basedir, testskip=1, test_file=None, aud_file=None, train_length=None, need_lip=False, need_torso=False, bc_type='torso_bc_imgs', deca_file=None, lip_file=None):
    if test_file is not None:
        
        with open(os.path.join(basedir, test_file)) as fp:
            meta = json.load(fp)
        poses = []
        auds = []
        lips = []
        imgs = []
        mouths = []
        lip_rects = []
        torso_bcs = []
        deca_exps = []
        aud_features = np.load(os.path.join(basedir, aud_file))
        _,aud_features = audio_stcs_cal(aud_features)
        deca_exp_features = np.load(os.path.join(basedir, deca_file))
        _,deca_exp_features = stcs_cal(deca_exp_features)
        lip_features = np.load(os.path.join(basedir, lip_file))
        _,lip_features = stcs_cal(lip_features)

        aud_start = 0
        for frame in meta['frames'][::testskip]:
            poses.append(np.array(frame['transform_matrix']))
            if aud_file == 'aud.npy':
                auds.append(aud_features[min(frame['aud_id'], aud_features.shape[0]-1)])
                deca_exps.append(deca_exp_features[min(frame['aud_id'], deca_exp_features.shape[0]-1)])
                lips.append(lip_features[min(frame['aud_id'], aud_features.shape[0]-1)])
            else:
                auds.append(aud_features[min(aud_start, aud_features.shape[0]-1)])
                deca_exps.append(deca_exp_features[min(aud_start, aud_features.shape[0]-1)])
                lips.append(lip_features[min(aud_start, aud_features.shape[0]-1)])
                aud_start = aud_start+1
            fname = os.path.join(basedir, 'com_imgs', str(frame['img_id']) + '.jpg')
            mouth_fname = os.path.join(basedir, 'mouth', str(frame['img_id']) + '.png')
            imgs.append(fname)
            mouths.append(mouth_fname)
            lip_rects.append(np.array(frame['lip_rect'], dtype=np.int32))
            torso_bcs.append(os.path.join(basedir, bc_type, str(frame['img_id']) + '.jpg'))
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        lips = np.array(lips).astype(np.float32)
        deca_exps = np.array(deca_exps).astype(np.float32)
        lip_rects = np.array(lip_rects).astype(np.int32)
        #target = torch.as_tensor(imageio.imread(image_path)).to(device).float() / 255.0
        bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))
        H, W = bc_img.shape[0], bc_img.shape[1]
        focal, cx, cy = float(meta['focal_len']), float(meta['cx']), float(meta['cy'])
        avg_c2w = poses.mean(0)
        #render_poses = torch.stack([pose_around(angle, -10, avg_c2w) for angle in [0.009, 0.011, 0.013, 0.015, 0.017]], 0, ).cuda()
        render_poses = torch.stack([pose_around(angle, 5, avg_c2w) for angle in [-0.017, -0.009, -0.005, 0.005, 0.009, 0.017]], 0, ).cuda()
        #render_poses = torch.stack([pose_around(angle, -10, avg_c2w) for angle in [0.017, 0.02]], 0, ).cuda()
        if need_lip:
            if need_torso:
                return imgs, poses, auds, bc_img, [H, W, focal, cx, cy], lip_rects, torso_bcs, render_poses, deca_exps, mouths, lips
            else:
                return imgs, poses, auds, bc_img, [H, W, focal, cx, cy], lip_rects, None, render_poses, deca_exps, mouths, lips
        else:
            if need_torso:
                return imgs, poses, auds, bc_img, [H, W, focal, cx, cy], torso_bcs, deca_exps, mouths, lips
            else:
                return imgs, poses, auds, bc_img, [H, W, focal, cx, cy], None, deca_exps, mouths, lips

    #every id has a dir
    id_list = sorted(os.listdir(os.path.join(basedir)))

    id_num = len(id_list)
    metas = {}
    all_imgs = {}
    all_mouths = {}
    all_poses = {}
    all_auds = {}
    all_lips = {}
    all_deca_exps = {}
    all_sample_rects = {}
    all_lip_rects = {}
    counts= {}

    i_split = {}
    bc_img = {}
    all_torso_bcs = {}

    splits = ['train', 'val']
    for i in id_list:#range id
        metas[i] = {}
        for s in splits:
            if s in ['train']:
                with open(os.path.join(basedir, i, 'transforms_{}.json'.format(s)), 'r') as fp:
                    metas[i][s] = json.load(fp)
            else:
                with open(os.path.join(basedir, i, 'transforms_{}.json'.format(s)), 'r') as fp:
                    metas[i][s] = json.load(fp)
        all_imgs[i] = []
        all_mouths[i] = []
        all_poses[i] = []
        all_auds[i] = []
        all_lips[i] = []
        all_deca_exps[i] = []
        all_sample_rects[i] = []
        all_lip_rects[i] = []
        all_torso_bcs[i] = []

        aud_features = np.load(os.path.join(basedir, i, 'aud.npy'))
        _,aud_features = audio_stcs_cal(aud_features)
        deca_exp_features = np.load(os.path.join(basedir, i, 'deca_exp.npy'))
        _,deca_exp_features = stcs_cal(deca_exp_features)
        lip_features = np.load(os.path.join(basedir,i, lip_file))
        _,lip_features = stcs_cal(lip_features)


        counts[i] = [0]
        for s in splits:
            meta = metas[i][s]
            imgs = []
            mouths = []
            poses = []
            auds = []
            lips = []
            deca_exps = []
            sample_rects = []
            lip_rects = []
            #mouth_rects = []
            #exps = []
            torso_bcs = []

            if s == 'train' or testskip == 0:
                skip = 1
            else:
                skip = testskip

            for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, i , 'com_imgs',
                                     str(frame['img_id']) + '.jpg')
                mouth_fname = os.path.join(basedir, i , 'mouth',
                                     str(frame['img_id']) + '.png')
                imgs.append(fname)
                mouths.append(mouth_fname)
                poses.append(np.array(frame['transform_matrix']))
                auds.append(
                    aud_features[min(frame['aud_id'], aud_features.shape[0]-1)])
                deca_exps.append(
                    deca_exp_features[min(frame['aud_id'], deca_exp_features.shape[0]-1)]
                )
                lips.append(
                    lip_features[min(frame['aud_id'], lip_features.shape[0]-1)])
                sample_rects.append(np.array(frame['face_rect'], dtype=np.int32))
                lip_rects.append(np.array(frame['lip_rect'], dtype=np.int32))
                #add bc
                torso_bcs.append(os.path.join(basedir, i , bc_type,
                                     str(frame['img_id']) + '.jpg'))

            imgs = np.array(imgs)
            mouths = np.array(mouths)
            poses = np.array(poses).astype(np.float32)
            auds = np.array(auds).astype(np.float32)
            lips = np.array(lips).astype(np.float32)
            torso_bcs = np.array(torso_bcs)
            counts[i].append(counts[i][-1] + imgs.shape[0])
            all_imgs[i].append(imgs)
            all_mouths[i].append(mouths)
            all_poses[i].append(poses)
            all_auds[i].append(auds)
            all_lips[i].append(lips)
            all_deca_exps[i].append(deca_exps)
            all_sample_rects[i].append(sample_rects)
            all_lip_rects[i].append(lip_rects)
            all_torso_bcs[i].append(torso_bcs)

        i_split[i] = [np.arange(counts[i][j], counts[i][j+1]) for j in range(len(splits))]
        all_imgs[i] = np.concatenate(all_imgs[i], 0)
        all_mouths[i] = np.concatenate(all_mouths[i], 0)
        all_torso_bcs[i] = np.concatenate(all_torso_bcs[i], 0)
        all_poses[i] = np.concatenate(all_poses[i], 0)
        all_auds[i] = np.concatenate(all_auds[i], 0)
        all_lips[i] = np.concatenate(all_lips[i], 0)
        all_deca_exps[i] = np.concatenate(all_deca_exps[i],0)
        all_sample_rects[i] = np.concatenate(all_sample_rects[i], 0)
        all_lip_rects[i] = np.concatenate(all_lip_rects[i], 0)

        bc_img[i] = imageio.imread(os.path.join(basedir, i, 'bc.jpg'))

    H, W = bc_img[i].shape[:2]
    focal, cx, cy = float(meta['focal_len']), float(meta['cx']), float(meta['cy'])
    #if need the lip region, can return the all_lip_rects
    if need_lip:
        return all_imgs, all_poses, all_auds, bc_img, [H, W, focal, cx, cy], all_sample_rects, i_split, id_num, all_lip_rects, all_torso_bcs, all_deca_exps, all_mouths, all_lips
    else:
        return all_imgs, all_poses, all_auds, bc_img, [H, W, focal, cx, cy], all_sample_rects, i_split, id_num, all_torso_bcs, all_deca_exps, all_mouths, all_lips
