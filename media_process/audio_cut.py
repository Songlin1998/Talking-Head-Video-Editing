from moviepy.editor import AudioFileClip, concatenate_audioclips
import os

id = 'female_black'
save_folder = '.../dataset/edit_' + id
audio = AudioFileClip('.../dataset/female_black/0/aud.wav',fps=512000)

insert_audio = concatenate_audioclips([audio.subclip('0:7.9','0:8.4')]) 

new_audio = concatenate_audioclips([audio.subclip('0:0','0:7.9'),audio.subclip('0:3.5','0:4.32'),audio.subclip('0:8.4','0:10')])

insert_audio.write_audiofile(os.path.join(save_folder,'audio.wav'))
new_audio.write_audiofile(os.path.join(save_folder,'audio_new.wav'))