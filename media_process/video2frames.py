import cv2
import os
vid_file = '.../obama.mp4'
ori_imgs_dir = '.../imgs'
cap = cv2.VideoCapture(vid_file)
frame_num = 0
while(True):
    _, frame = cap.read()
    if frame is None:
        break
    cv2.imwrite(os.path.join(ori_imgs_dir, str(frame_num) + '.jpg'), frame)
    frame_num = frame_num + 1
cap.release()