from moviepy.editor import AudioFileClip, VideoFileClip
import imageio.v2 as imageio
from tqdm import tqdm
import os
import cv2




save_folder = '/hd2/yangsonglin/myDFRF/dataset/finetune_models/male_black_train/dfrf_predict_renderonly_path_000088/'

visual_frames = VideoFileClip('/hd2/yangsonglin/myDFRF/dataset/finetune_models/male_black_train/dfrf_predict_renderonly_path_000088/video.mp4') # 视频文件路径
audio_frames = AudioFileClip('/hd2/yangsonglin/myDFRF/dataset/male_black/0/aud.wav') # 语音文件路径

video = visual_frames.set_audio(audio_frames)
video.write_videofile(os.path.join(save_folder, 'dfrf.mp4')) # 保存视频文件的路径
# VideoFileClip('/hd2/yangsonglin/myDFRF/male_black_myDFRF.mp4').speedx(0.7).write_videofile('/hd2/yangsonglin/myDFRF/male_black_myDFRF_slow.mp4')