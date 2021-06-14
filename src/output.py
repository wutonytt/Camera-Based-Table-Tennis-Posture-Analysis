import cv2
import pandas as pd
import numpy as np
import os
import sys
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
from PIL import Image 
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from tqdm import tqdm, trange


def foreback_output(imagepath, front_name, rightTestData, leftTestData, modelRight, modelLeft, model_table, model_ball, out_name):
    numOfFiles = len([name for name in os.listdir(imagepath) if os.path.isfile(os.path.join(imagepath, name))])
    
    L = [None] * numOfFiles
    R = [None] * numOfFiles
    
    yright = modelRight.predict(rightTestData.iloc[:, 1:-1])
    yright = np.where(yright == 1, 'Fore', 'Back')
    
    yleft = modelLeft.predict(leftTestData.iloc[:, 1:-1])
    yleft = np.where(yleft == 1, 'Fore', 'Back')
    
    rightTestData['left/right'] = 1
    leftTestData['left/right'] = 0
    
    y = pd.concat([pd.Series(yright), pd.Series(yleft)], ignore_index=True)
    testData = pd.concat([rightTestData, leftTestData], ignore_index=True)
    
    h = pd.DataFrame(y, columns = ['fore/back_pred'])
    pred = pd.concat([testData, h], axis=1)
    for index, row in pred.iterrows():
        hand = row['fore/back_pred']
        ind = int(row['file_num'])
        pos = row['left/right']
        if pos == 1:
            for i in range(15):
                R[ind+i] = hand
        else:
            for i in range(15):
                L[ind+i] = hand
    ball_arr = []
    table_arr = []
    for i in trange(numOfFiles):
        path = os.path.join(imagepath, front_name + str(i).zfill(12) + '_rendered.jpg')
        img = cv2.imread(path)
        ball_pred, table_pred = pred_ball_n_table(img, model_table, model_ball)
        if i == 0:
            ball_sum = np.zeros(shape=ball_pred.shape)
        mask = np.copy(ball_pred)
        mask *= 1800
        mask[mask >= 255] = 255
        mask[mask < 255] = 0
        ball_sum += mask
        table_arr.append(table_pred)
        ball_arr.append(ball_pred)
    
    real_ball_mask = ball_sum / (numOfFiles+1)
    real_ball_mask[real_ball_mask <= 15] = 1
    real_ball_mask[real_ball_mask > 15] = 0
    
    l_fore_cnt = 0
    l_back_cnt = 0
    r_fore_cnt = 0
    r_back_cnt = 0
    
    for i in trange(numOfFiles):
        path = os.path.join(imagepath, front_name + str(i).zfill(12) + '_rendered.jpg')
        img = cv2.imread(path)
        if i == 0:
            height, width, channels = img.shape
            video_out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (width, height))
        
        if L[i] == 'Fore':
            l_fore_cnt += 1
        else:
            l_back_cnt += 1
        if R[i] == 'Fore':
            r_fore_cnt += 1
        else:
            r_back_cnt += 1
            
        l_fore_ratio = l_fore_cnt / (l_fore_cnt + l_back_cnt)
        l_back_ratio = l_back_cnt / (l_fore_cnt + l_back_cnt)
        r_fore_ratio = r_fore_cnt / (r_fore_cnt + r_back_cnt)
        r_back_ratio = r_back_cnt / (r_fore_cnt + r_back_cnt)
        
        img = plot_ball_n_table(img, real_ball_mask * ball_arr[i], table_arr[i])
        img = plot_foreback(img, width, height, L[i], R[i], l_fore_ratio, l_back_ratio, r_fore_ratio, r_back_ratio)
        video_out.write(img)
    video_out.release()
    
    return R, L

def plot_foreback(img, width, height, L_events, R_events, l_fore_ratio, l_back_ratio, r_fore_ratio, r_back_ratio):
    L_event = 'L: {}'.format(L_events)
    R_event = 'R: {}'.format(R_events)
    l_fore_ratio = 'Fore: %.2f%%' % (l_fore_ratio*100)
    l_back_ratio = 'Back: %.2f%%' % (l_back_ratio*100)
    r_fore_ratio = 'Fore: %.2f%%' % (r_fore_ratio*100)
    r_back_ratio = 'Back: %.2f%%' % (r_back_ratio*100)
    
    img = cv2.putText(img, L_event, (int(width/20), int(14*height/15)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    img = cv2.putText(img, R_event, (int(15*width/20), int(14*height/15)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    
    img = cv2.putText(img, l_fore_ratio, (int(6*width/20), int(13*height/15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    img = cv2.putText(img, l_back_ratio, (int(6*width/20), int(14*height/15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    img = cv2.putText(img, r_fore_ratio, (int(11*width/20), int(13*height/15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    img = cv2.putText(img, r_back_ratio, (int(11*width/20), int(14*height/15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img
    
    
def pred_ball_n_table(image_base, model_table, model_ball):
    val_aug = get_validation_augmentation()
    
    image_base = cv2.cvtColor(image_base, cv2.COLOR_BGR2RGB)
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn('efficientnet-b4', 'imagenet')
    
    image = cv2.resize(image_base, (480, 320))
    frame = val_aug(image=image)['image']
    frame = preprocessing_fn(frame)
    frame = to_tensor(frame)
    
    frame = torch.from_numpy(frame).unsqueeze(0)
    
    ball_mask = model_ball.cpu().predict(frame.cpu()).squeeze().numpy()
    ball_mask_full = cv2.resize(ball_mask, (image_base.shape[1], image_base.shape[0]))
    ball_mask_full = np.expand_dims(ball_mask_full, 2)
    
    transparency = 0.7
    ball_mask_full *= transparency
    
    
    table_mask = model_table.cpu().predict(frame.cpu()).squeeze().numpy()
    table_mask_full = cv2.resize(table_mask, (image_base.shape[1], image_base.shape[0]))
    table_mask_full = np.expand_dims(table_mask_full, 2)
    
    transparency = 0.4
    table_mask_full *= transparency
    
    
    return ball_mask_full, table_mask_full


def plot_ball_n_table(image_base, ball_mask_full, table_mask_full):
    image_base = cv2.cvtColor(image_base, cv2.COLOR_BGR2RGB)
    
    green = np.ones(image_base.shape, dtype=np.float) * (0,1,0)
    out = green * ball_mask_full + image_base * (1.0 - ball_mask_full) / 255
    out = np.clip(out, 0, 1)
    
    out = (255.0 * out).astype(np.uint8)
    
    red = np.ones(image_base.shape, dtype=np.float) * (1,0,0)
    out = red * table_mask_full + out * (1.0 - table_mask_full) / 255
    out = np.clip(out, 0, 1)
    
    out = cv2.cvtColor((255.0 * out).astype(np.uint8),cv2.COLOR_RGB2BGR)
    return out
  
    
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(320, 320)
    ]
    return albu.Compose(test_transform)
    
    
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
    

# def make_clip()