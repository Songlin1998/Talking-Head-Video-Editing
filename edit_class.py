import tensorflow as tf
from data_process.util import extract_frames_video_all
import os
import json
import numpy as np
import cv2
from data_loader.neuralrender import NeuralRender_DataSample
from moviepy.editor import AudioFileClip, ImageSequenceClip
import moviepy.editor as mp
import torch
from audio_util.deepspeech_features.deepspeech_features import conv_audios_to_deepspeech
from audio_util.deepspeech_features.deepspeech_store import get_deepspeech_model_file
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '8'
#只输出fatal信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tools.DECA.decalib.deca import DECA
from tools.DECA.decalib.utils.config import cfg as deca_cfg
from utils.torch_functions import Normalize
from network.net_rnn import TextureGenerator
from network.model_nr import maskErosion
from data_process.util import estimate_inverse_crop_params, inverse_crop
from se2e_network.model_s2e_transformer import Speech2Expression_Transformer
from scipy.io import wavfile
import json
import librosa
import wave, audioop
from pydub import AudioSegment
import math
from tqdm import tqdm

class edit_prepare_data(object):
    '''
    ### (1) Parameter Description

    - **video_path**: Path to the original video file.
    - **frames_path**: Path to save the frames of the original video.
    - **audio_path**: Path to save the audio file of the original video.
    - **edit_T**: Sequence length for se2e and e2v processing.
    - **ori_dict_path**: Transcription of the original video audio (word, start time, duration).
    - **edit_text_path**: Path to the edited text.
    - **edit_part_audio_path**: Audio file of the edited word.
    - **edit_part_dict_path**: Transcription of the edited word's audio (word, start time, duration).
    - **num_frames_ori**: Total number of frames (to synchronize the total frames of audio and video).
    - **duration_total**: Total duration of the video (e.g., 20.190 seconds).

    ### (2) Function Description

    - **change_video_fps(self)**: Change the video frame rate (fps = 25).
    - **get_video_frames(self)**: Extract video frames with shape `[T_total, 3, 256, 256]`.
    - **get_video_params(self)**: Obtain DECA parameters of the video frames.
    - **get_audio_ds_frames(self)**: Extract DS embeddings for the audio frames `[T_total, 29]`.
    - **get_edit_audio_frames_start_length(self)**: Get the starting frame and duration for inserting the edited word into the original video.
    - **get_expression_frames(self)**: Extract expression coefficients of the video frames `[T_total, 53]`.

    ### Sampling Fixed-Length Sequences for se2e and e2v Inputs

    #### For se2e:
    - **get_edit_audio_frames(self)**: Extract fixed-length audio embeddings `[T, 29]`.
    - **get_edit_expression_frames(self)**: Extract fixed-length expression coefficients embeddings `[T, 53]`.

    #### For e2v:
    - **get_edit_video_frames(self)**: Extract fixed-length video frames `[T, 3, 256, 256]`.
    - **get_edit_video_params(self)**: Extract fixed-length video frame rendering coefficients.

    '''
    def __init__(self, video_path, frames_path, audio_path, edit_T, ori_dict_path, edit_text_path, 
                    edit_part_audio_path, edit_part_dict_path, num_frames_ori, duration_total= 20.190
                    ):
        
        self.source_test_data = NeuralRender_DataSample(frames_path, image_size=256, tempo_extent=1, # 相当于每个batch都为1
                                            params_only=False, use_audio=False)
        
        self.T_total = len(self.source_test_data)
        self.audio_path = audio_path
        self.T = edit_T 
        self.ori_dict_path = ori_dict_path
        self.edit_text_path = edit_text_path
        self.edit_part_audio_path = edit_part_audio_path
        self.edit_part_dict_path = edit_part_dict_path
        self.num_frames_ori = num_frames_ori
        self.duration_total = duration_total

    
    def change_video_fps(self):
        video_input_path=".../dataset/Obama_00.mp4"
        video_output_path="...dataset/Obama_25.mp4"
        clip = mp.VideoFileClip(video_input_path)
        clip.write_videofile(video_output_path, fps=25)

    def get_video_frames(self):
        
        video_frames_list = []
        for i in range(self.T_total):
            video_frames_list.append(self.source_test_data[i]['face'])  # [T_total, 3, 256, 256]
        video_frames = torch.cat(video_frames_list,dim=0) # [T_total, 3, 256, 256]
        return video_frames
    
    def get_video_params(self):
  
        video_params = {}
        for key in self.source_test_data[0]['params'].keys():
            single_video_params_list = []
            for i in range(self.T_total):
                single_video_params_list.append(self.source_test_data[i]['params'][key].cuda()) 
            single_ideo_params = torch.cat(single_video_params_list,dim=0)
            video_params.update({key:single_ideo_params})
        return video_params

    def get_audio_ds_frames(self):
        audio_frames_list = conv_audios_to_deepspeech(audios=self.audio_path,
                                                    num_frames=self.T_total+1,
                                                    deepspeech_pb_path = get_deepspeech_model_file())
        audio_frames = torch.Tensor(audio_frames_list) # [T_total, 29]
        return  audio_frames
    
    def get_edit_audio_frames_start_length(self):

        ori_dict_file = open(self.ori_dict_path, "rb")
        ori_dict = json.load(ori_dict_file)

        edit_text_file = open(self.edit_text_path,'r')
        edit_text = list(filter(None,edit_text_file.read().split(" ")))


        edit_part_dict_file = open(self.edit_part_dict_path, "rb")
        edit_part_dict = json.load(edit_part_dict_file)

        duration_edit = edit_part_dict['words'][0]['duration']
        duration_before_edit = 0 
        
        for i in range(len(ori_dict['words'])):
            if edit_text[i] != ori_dict['words'][i]['word']:
                duration_before_edit = ori_dict['words'][i]['start_time']
                break
        
        start = int((duration_before_edit/self.duration_total) * self.num_frames_ori)
        length = int(duration_edit * 25)
        edit_audio_frames_list = conv_audios_to_deepspeech(audios=self.edit_part_audio_path,
                                                    num_frames=length+1, # 为了和video frmase保持一致
                                                    deepspeech_pb_path = get_deepspeech_model_file())
        edit_audio_frames = torch.Tensor(edit_audio_frames_list) # [T_total, 29]
        
        return edit_audio_frames, start, length
    
    def get_expression_frames(self):
        expression_frames = torch.cat((self.source_test_data[0]['params']['exp'],self.source_test_data[0]['params']['pose'][:,3:]),1)
        for j in range(1,self.T_total):
            expression_frame = torch.cat((self.source_test_data[j]['params']['exp'],self.source_test_data[j]['params']['pose'][:,3:]),1)
            expression_frames = torch.cat((expression_frames,expression_frame),0)
        return expression_frames
    
    def get_edit_audio_frames(self, audio_frames, edit_audio_frames, start, length):
        
        edit_audio_frames = torch.cat((audio_frames[start-(self.T-length)//2 : start, :],
                                        edit_audio_frames,
                                        audio_frames[start : start-(self.T-length)//2 + (self.T-length), :]),0)
        return edit_audio_frames
    
    def get_edit_expression_frames(self, expression_frames, start, length):

        edit_expression_frames = torch.cat((expression_frames[start-(self.T-length)//2 : start, :],
                                        torch.zeros(length,53),
                                        expression_frames[start : start-(self.T-length)//2 + (self.T-length), :]),0)
        return edit_expression_frames
    
    def get_edit_video_frames(self, video_frames, start, length):
        edit_video_frames = torch.cat((video_frames[start-(self.T-length)//2 : start, :, :, :],
                                        video_frames[start-1, :, :, :].repeat(length//2,1,1,1),
                                        video_frames[start, :, :, :].repeat(length-length//2,1,1,1),
                                        video_frames[start : start-(self.T-length)//2 + (self.T-length), :, :, :]),0)
        return edit_video_frames

    def get_edit_video_params(self, video_params, start, length, predict_exp_pose):
        edit_video_params = {}
        for key in video_params.keys():
            if key == 'exp': 
                single_param = torch.cat((
                    video_params['exp'][start-(self.T-length)//2 : start,:],   
                    video_params['exp'][start-1, :].repeat(length//2,1),
                    video_params['exp'][start, :].repeat(length-length//2,1),
                    video_params['exp'][start : start-(self.T-length)//2 + (self.T-length),:]
                ),0)
            elif key == 'pose':
                after = torch.cat((
                    video_params['pose'][start-(self.T-length)//2 : start, 3:], 
                    video_params['pose'][start-1, 3:].repeat(length//2,1),
                    video_params['pose'][start, 3:].repeat(length-length//2,1),
                    video_params['pose'][start : start-(self.T-length)//2 + (self.T-length), 3:] 
                ),0)
                before = torch.cat((
                    video_params['pose'][start-(self.T-length)//2 : start, :3], 
                    video_params['pose'][start-1, :3].repeat(length//2,1), 
                    video_params['pose'][start, :3].repeat(length-length//2,1), 
                    # video_params['pose'][0:9, :3],
                    video_params['pose'][start: start-(self.T-length)//2 + (self.T-length), :3] 
                ),0)
                single_param = torch.cat((before,after),1)
            elif key == 'light':
                single_param = torch.cat((
                    video_params['light'][start-(self.T-length)//2 : start,:,:],
                    video_params['light'][start-1, :,:].repeat(length//2,1,1),
                    video_params['light'][start, :,:].repeat(length-length//2,1,1),
                    video_params['light'][start : start-(self.T-length)//2 + (self.T-length),:,:]
                ),0)
            else:
                single_param = torch.cat((
                    video_params[key][start-(self.T-length)//2 : start,:],
                    video_params[key][start-1, :].repeat(length//2,1),
                    video_params[key][start, :].repeat(length-length//2,1),
                    video_params[key][start : start-(self.T-length)//2 + (self.T-length),:]
                ),0)
            edit_video_params.update({key:single_param})
        return edit_video_params

def _toTensor(x):
    return torch.Tensor(x)

def ProcessVisDict(visdict, crop_boxs, target_size, model_imgsize=256):
    ret_shape, ret_lower_face_mask, ret_mouth_mask = [], [], []
    bs = visdict['shape_detail_images'].size(0)
    for idx in range(bs):
        crop_box = crop_boxs[idx]
        input_shape = visdict['shape_detail_images'][idx].permute(1,2,0).cpu().numpy() # (T, 3, 512, 512)
        input_lower_face_mask = visdict['mask_dict']['lower_face'][idx, 0].cpu().numpy()
        input_bg_mask = 1-input_lower_face_mask # (b, 1, 512, 512)
        input_mouth_mask = visdict['mask_dict']['mouth'][idx, 0].cpu().numpy()
        mask = np.stack([input_lower_face_mask, input_bg_mask, input_mouth_mask])
        mask = (mask * 255).astype(np.uint8)
        mask = mask.transpose(1,2,0)
        input_shape = (input_shape*255).astype(np.uint8)

        tform = estimate_inverse_crop_params(input_shape, crop_box)
        new_shape = cv2.resize(inverse_crop(input_shape, tform, target_size), (model_imgsize, model_imgsize))
        new_mask = cv2.resize(inverse_crop(mask, tform, target_size), (model_imgsize, model_imgsize))

        ret_shape.append(_toTensor(new_shape).permute(2,0,1)/255.0)
        new_mask = _toTensor(new_mask/255.0)
        ret_lower_face_mask.append(new_mask[:, :, 0].unsqueeze(0))
        ret_mouth_mask.append(new_mask[:, :, 2].unsqueeze(0))
    return torch.stack(ret_shape), torch.stack(ret_lower_face_mask), torch.stack(ret_mouth_mask)


class edit_inference(object):
    def __init__(self,
                se2e_model_path,
                e2v_model_path):
        
        # load se2e
        N_ACOUSTIC = 29 
        self.model_se2e = Speech2Expression_Transformer(n_acoustic=N_ACOUSTIC+53, n_expression=53, downsample=False).cuda()
        checkpoint = torch.load(se2e_model_path, map_location='cpu')
        self.model_se2e.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model_se2e = self.model_se2e.cuda()
        self.model_se2e.eval()
        # load e2v
        device = 'cuda'
        deca_cfg.dataset.image_size = 256
        deca_cfg.model.use_tex = False
        self.deca = DECA(config=deca_cfg, device=device)
        Norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), device=device)
        Norm_rev = Normalize((-1, -1, -1), (2, 2, 2), device=device)

        ckpt = torch.load(e2v_model_path) # log_prefix
        self.e2v = TextureGenerator(ngf=48, dec_rnn=True, scale=1, mouth_only=False, norm_type='instance', use_dropout=False).to(device)
        self.e2v.load_state_dict(ckpt['model_G'], strict=True)
        self.e2v = self.e2v.eval()

    def expression_predict(self, edit_audio_frames, edit_expression_frames):
        input_audio_expression = torch.cat((edit_audio_frames,edit_expression_frames),1).unsqueeze(0)
        predict_exp_pose = self.model_se2e(input_audio_expression.cuda()).squeeze(0)
        return predict_exp_pose
    
    def video_predict(self,edit_video_params,edit_video_frames):
        hidden = None
        previous_input_shape = None

        _, transfer_visdict = self.deca.decode(edit_video_params)
        
        input_frames = edit_video_frames.unsqueeze(0) 
        
        input_shape, input_lower_face_mask, input_mouth_mask = ProcessVisDict(transfer_visdict, 
                                                                            crop_boxs=edit_video_params['crop_box'], 
                                                                            target_size=256, model_imgsize=256)
        
        input_shape = input_shape.unsqueeze(0) 
        input_bg_mask = ~(input_lower_face_mask.unsqueeze(0).bool()) 
        bs, t, c, h, w = input_bg_mask.shape
        _bg_mask = input_bg_mask.reshape(bs*t, c, h, w)
        erosion=0.3
        input_bg_mask_erosion = maskErosion(_bg_mask, erosionFactor=erosion).cuda()
        input_bg_mask_erosion = input_bg_mask_erosion.reshape(bs, t, c, h, w)
        
        # render to talking face
        fake, hidden = self.e2v.forward(imgs=input_frames.cuda(), 
                                        mesh=input_shape.cuda(), 
                                        bg_mask=input_bg_mask.cuda(), 
                                        bg_mask_erosion=input_bg_mask_erosion,
                                        last_hidden_state=hidden, 
                                        return_status=True)
    
        Norm_rev = Normalize((-1, -1, -1), (2, 2, 2), device='cuda') 
        fake = Norm_rev(fake[0])
        predict_video_frames = (255*fake.permute(0,2,3,1).detach().cpu().numpy()).astype(np.uint8)

        return predict_video_frames
        

def writeVideo(audio_path, video_path, composite_seq, audio_sample_frequency=16000):
    target_audio_clip = AudioFileClip(audio_path, fps=audio_sample_frequency)
    print('Write to video, audio_duration:%.2f, image sequence length:%d' % (target_audio_clip.duration, len(composite_seq)))
    os.makedirs(os.path.split(video_path)[0], exist_ok=True)
    imgseqclip = ImageSequenceClip(composite_seq, fps=25)
    temp = imgseqclip.set_audio(target_audio_clip)
    temp.write_videofile(video_path, logger=None)

class edit_video_generation(object):

    def __init__(self, edit_T, save_path):
        self.save_path = save_path # ~/~.mp4
        self.T = edit_T
    
    def composite_edit_origin_video_frames(self, video_frames, predict_video_frames, start, length):

        Norm_rev = Normalize((-1, -1, -1), (2, 2, 2))
        video_frames = Norm_rev(video_frames)
        Norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        predict_video_frames = _toTensor(predict_video_frames).permute(0,3,1,2)/255.0
        composite_video_frames = torch.cat((
            video_frames[:start,:,:,:],
            predict_video_frames[math.ceil((self.T-length)//2):math.ceil((self.T-length)//2)+length,:,:,:],
            video_frames[start:,:,:,:]
        ),0)

        composite_video_frames = composite_video_frames.mul_(255).clamp_(0, 255).permute(0, 2, 3, 1).type(torch.uint8).numpy()
        # composite_video_frames = cv2.cvtColor(composite_video_frames, cv2.COLOR_RGB2BGR)
        composite_video_frames = list(composite_video_frames)
        # composite_video_frames (510, 256, 256, 3)
        return composite_video_frames
    
    def predict_video_generation(self, predict_video_frames):
        video_path = '.../....mp4'
        predict_video_frames = list(predict_video_frames)
        imgseqclip = ImageSequenceClip(predict_video_frames, fps=25)
        imgseqclip.write_videofile(video_path, logger=None)

    
    def composite_edit_origin_audio(self, ori_audio, edit_part_audio, wav_out, start):

        ori_wav = AudioSegment.from_wav(ori_audio)
        ori_before_edit = ori_wav[:start*(1/25)*1000]
        ori_after_edit = ori_wav[start*(1/25)*1000:]
        edit_part_wav = AudioSegment.from_wav(edit_part_audio)
        edit_wav = ori_before_edit + edit_part_wav + ori_after_edit
        edit_wav.export(wav_out, format='wav')

    def edit_video_generation(self,edit_audio_path, composite_video_frames):

        writeVideo(edit_audio_path, self.save_path, composite_video_frames)
        print('done!')

if __name__ == '__main__':
    
    edit_prepare = edit_prepare_data(
                        # for 2min
                        video_path='.../obama_256/obama.mp4', 
                        frames_path='.../frames/obama',
                        audio_path='.../obama/obama.wav',
                        edit_T=51,
                        ori_dict_path='.../dataset/ori_dict.json', 
                        edit_text_path='.../dataset/edit_text.txt', 
                        edit_part_audio_path='.../dataset/edit_part_audio.wav', 
                        edit_part_dict_path='.../dataset/edit_part_dict.json', 
                        num_frames_ori=2736, 
                        duration_total= 109.48)
    
    edit_predict = edit_inference(
                se2e_model_path='.../snapshot.epoch:14000',
                e2v_model_path='.../snapshot.epoch.480')

    edit_video = edit_video_generation(
                edit_T=51, 
                save_path='.../Obama.mp4')
    
    audio_frames = edit_prepare.get_audio_ds_frames()
    expression_frames = edit_prepare.get_expression_frames()
    edit_audio_frames, start, length = edit_prepare.get_edit_audio_frames_start_length()
    video_frames = edit_prepare.get_video_frames()
    video_params = edit_prepare.get_video_params()
    edit_audio_frames = edit_prepare.get_edit_audio_frames(audio_frames, edit_audio_frames, start, length) 
    edit_expression_frames = edit_prepare.get_edit_expression_frames(expression_frames, start, length) 
    predict_exp_pose = edit_predict.expression_predict(edit_audio_frames, edit_expression_frames) 

    edit_video_frames = edit_prepare.get_edit_video_frames(video_frames, start, length) # [T,3,256,256]
    edit_video_params = edit_prepare.get_edit_video_params(video_params, start, length, predict_exp_pose)
    predict_video_frames = edit_predict.video_predict(edit_video_params,edit_video_frames)

    edit_video.composite_edit_origin_audio(ori_audio='.../Obama_25.wav', 
                                        edit_part_audio='.../edit_part_audio.wav', 
                                        wav_out='.../edit_audio.wav', 
                                        start = start)
    composite_video_frames = edit_video.composite_edit_origin_video_frames(video_frames, predict_video_frames, start, length)
    edit_video.edit_video_generation(edit_audio_path='.../edit_audio.wav', 
                                    composite_video_frames=composite_video_frames)
    edit_video.predict_video_generation(predict_video_frames)