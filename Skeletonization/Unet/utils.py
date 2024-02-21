import re
import cv2
import random
import config
import numpy as np

##Functions to sort folder, files in the "natural" way:
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]    
    
def lambda_(epoch):
    return pow((1 - (epoch / 500)), 0.9)   
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
                
                
class Flip(object):
    def __call__(self, sample):
        image, label = sample
        mode = random.choice([-1, 0, 1, 2])
        if mode != 2:
            image = cv2.flip(image, mode)
            label = cv2.flip(label, mode)

        return (image, label)


class Rotate(object):
    def __call__(self, sample):
        image, label = sample
        r = random.choice([0, 1, 2, 3])
        if r == 1:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if r == 2:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)
        if r == 3:
            image = cv2.rotate(image, cv2.ROTATE_180)
            label = cv2.rotate(label, cv2.ROTATE_180)

        return (image, label)
    
from torchvision import transforms

class Normalize(object):
    def __call__(self, sample):
        image, label = sample
        # Define the transformation pipeline using torchvision.transforms.Compose
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
            transforms.ToTensor()
        ])

        # Apply the transformation to both image and label
        image = transform(image)
        label = transform(label)

        return (image, label)
