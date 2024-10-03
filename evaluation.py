from multiprocessing import reduction
import torch
import numpy as np
import cv2
from PIL import Image
import math
from torchvision.transforms import transforms
import os
from tqdm import tqdm
import lpips
from pytorch_msssim import ms_ssim
from skimage import io

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

dataset = 30 
mse_loss = torch.nn.MSELoss(reduction='mean')
loss_fn = lpips.LPIPS(net='alex', spatial=True) 

for idx in range(500):

    audio_mse = []
    exp_mse = []
    audio_psnr =[]
    exp_psnr = []
    audio_lpips = []
    exp_lpips = []
    audio_ssim = []
    exp_ssim = []

    for i in range(3):

        img_ori_rgb = io.imread(os.path.join(f'.../dataset/eric_{dataset}/0/ori_imgs', f'{i}.jpg'))
        img_audio_rgb = io.imread(os.path.join(f'.../eric_{dataset}_val_{idx}/renderonly_path_000999', f'00{i}.png'))
        img_exp_rgb = io.imread(os.path.join(f'.../eric_{dataset}_val_{idx}/predict_renderonly_path_000150', f'00{i}.png'))

        img_ori_tensor = transforms.ToTensor()(img_ori_rgb) 
        img_audio_tensor = transforms.ToTensor()(img_audio_rgb)
        img_exp_tensor = transforms.ToTensor()(img_exp_rgb)

        img_ori_rgb = img_ori_tensor * 256
        img_audio_rgb = img_audio_tensor * 256
        img_exp_rgb = img_exp_tensor *256

        img_ori_norm = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(img_ori_tensor)
        img_audio_norm = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(img_audio_tensor)
        img_exp_norm = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(img_exp_tensor)

        # mse
        audio_mse.append(mse_loss(img_ori_tensor,img_audio_tensor).item())
        exp_mse.append(mse_loss(img_ori_tensor,img_exp_tensor).item())
        # psnr
        audio_psnr.append(psnr(img_ori_rgb.numpy(),img_audio_rgb.numpy()))
        exp_psnr.append(psnr(img_ori_rgb.numpy(),img_exp_rgb.numpy()))
        # lpips
        audio_lpips.append(loss_fn.forward(img_ori_norm.unsqueeze(0),img_audio_norm.unsqueeze(0)).mean().item())
        exp_lpips.append(loss_fn.forward(img_ori_norm.unsqueeze(0),img_exp_norm.unsqueeze(0)).mean().item())
        # ssim
        audio_ssim.append(ms_ssim(img_ori_tensor.unsqueeze(0),img_audio_tensor.unsqueeze(0), data_range=1, size_average=False).item())
        exp_ssim.append(ms_ssim(img_ori_tensor.unsqueeze(0),img_exp_tensor.unsqueeze(0), data_range=1, size_average=False).item())

    print('==================================================================')
    print(f'{idx}_audio_mse:',sum(audio_mse)/len(audio_mse))
    print(f'{idx}_exp_mse:',sum(exp_mse)/len(exp_mse))

    print(f'{idx}_audio_psnr:',sum(audio_psnr)/len(audio_psnr))
    print(f'{idx}_exp_psnr:',sum(exp_psnr)/len(exp_psnr))

    print(f'{idx}_audio_lpips:',sum(audio_lpips)/len(audio_lpips))
    print(f'{idx}_exp_lpips:',sum(exp_lpips)/len(exp_lpips))

    print(f'{idx}_audio_ssim:',sum(audio_ssim)/len(audio_ssim))
    print(f'{idx}_exp_ssim:',sum(exp_ssim)/len(exp_ssim))