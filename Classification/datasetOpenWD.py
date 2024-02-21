import torch
from PIL import Image
import cv2
import random
import numpy as np


def rotate_frame(frame, angle):
    rows = frame.shape[0]
    cols = frame.shape[1]
    center = (cols / 2, rows / 2)

    R = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_img = cv2.warpAffine(frame, R, (cols, rows))

    return rotated_img


def paste_image_into_backg(img, bk):
    h, w = bk.shape[:2]
    h1, w1 = img.shape[:2]
    # let store center coordinate as cx,cy
    cx, cy = (h - h1) // 2, (w - w1) // 2
    # use numpy indexing to place the resized image in the center of
    # background image
    bk[cy:cy + h1, cx:cx + w1] = img
    return bk



def rescale_img(img, factor):
    original_w = img.shape[1]
    original_h = img.shape[0]
    # percent of original size
    width = int(original_w * factor / 100)
    height = int(original_h * factor / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation= cv2.INTER_LINEAR)
    
    if factor < 100:
        mask = np.zeros((original_w, original_h, 3), np.uint8)
        img = paste_image_into_backg(img, mask)
    else:
        img = center_crop(img, original_w)

    return img



def get_fps_from_filename(filename):
    return int(filename.replace('.avi', '').split('fps_')[-1])


def center_crop(im, size):
    height, width = im.shape[0:2]
    cx = int(width / 2)
    cy = int(height / 2)

    y1 = int(cy - size / 2)
    y2 = int(cy + size / 2)
    x1 = int(cx - size / 2)
    x2 = int(cx + size / 2)

    im = im[y1:y2, x1:x2]

    return im



def get_frames_imgs(filename,  frame_list, dim_resize = (224, 224)):
    frames = []
    v_cap = cv2.VideoCapture(filename)

    start_frame_number = frame_list[0]
    last_frame_number = frame_list[-1]

    v_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    
    for fn in range(start_frame_number, last_frame_number + 1):
        success, frame = v_cap.read()
        size = 480
        try:
            frame = center_crop(frame, size)
            # resize image
            frame = cv2.resize(frame, dim_resize, interpolation = cv2.INTER_AREA)            

        
        except Exception as e:
            print(e)
            print(filename)
              
        if success is False:
            continue
            
        if fn in frame_list:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
                   
    v_cap.release()
    return frames



###########################################################################################################################################################
class HealthspanDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, dataset_size, seq_length,  transform=None, augmentation=None, img_size=(480,480)):
        self.all_videos = data_list
        self.dataset_size = dataset_size
        self.seq_length = seq_length
        self.transform = transform
        self.augmentation = augmentation
        self.img_size = img_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        seq_dir, frames_list = self.all_videos[idx]
        
        if seq_dir.split('/')[-1].startswith('N2'):
            label = 0
        else:
            label = 1

        frames = get_frames_imgs(seq_dir, frames_list, self.img_size)
        list_imgs = []

        if len(frames) > self.seq_length:
            frames = frames[:self.seq_length]

        if self.augmentation == True:
            scale_factor = random.randint(50, 150)
            degrees = random.randint(0, 359)

        for frame in frames:

            if self.augmentation == True:
                frame = rescale_img(frame, scale_factor)
                frame = rotate_frame(frame, degrees)

                
            frame = Image.fromarray(frame)
            img_tens = self.transform(frame)
            
            try:
                list_imgs.append(img_tens)
            except:
                print(seq_dir)

        if len(list_imgs) < self.seq_length:
            for rep in range(0, (self.seq_length - len(list_imgs))):
                try:
                    list_imgs.append(img_tens)
                except:
                    print(seq_dir)
                    

        label = torch.tensor(label)
        stacked_set = torch.stack(list_imgs)
        composed_sample = [stacked_set, label]

        return composed_sample
    
    

    
    
    

