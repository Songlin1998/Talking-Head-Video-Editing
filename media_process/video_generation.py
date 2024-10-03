from moviepy.editor import AudioFileClip, VideoFileClip
import imageio.v2 as imageio
from tqdm import tqdm
import os
import cv2

id = 'female_black'
save_folder = '.../dataset/edit_'+id
novel_frames_path = '.../dataset/finetune_models/female_black_train/predict_renderonly_path_000197'

num = 250
imgs = []
start_point = 197
end_point = 210
length = 21
for img_idx in tqdm(range(num)):

    img_name = str(img_idx) + '.jpg'
    img_file_path = os.path.join(f'.../dataset/{id}/0/com_imgs',img_name)
    
    if img_idx==start_point:
        for img_idx_novel in range(length):
                img_file_path = os.path.join(novel_frames_path,'{:03d}.png'.format(img_idx_novel))
                img = imageio.imread(img_file_path)
                imgs.append(img)
    if start_point<=img_idx<end_point:
        continue
        
    print(img_idx)
    img = imageio.imread(img_file_path)
    imgs.append(img)

imageio.mimwrite(os.path.join(save_folder, id+'.mp4'), imgs, fps=25, quality=8)

visual_frames = VideoFileClip(os.path.join(save_folder, id+'.mp4')) 
audio_frames = AudioFileClip(os.path.join(save_folder, 'audio_new.wav')) 

video = visual_frames.set_audio(audio_frames)
video.write_videofile(os.path.join(save_folder, id+'_new.mp4')) 