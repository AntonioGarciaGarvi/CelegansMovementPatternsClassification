# import the necessary packages
import torch
import os

# base path of the dataset
DATASET_PATH = '/home/jovyan/UNET/train/'

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")

# define the test split
TEST_SPLIT = 0.2

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.0001
NUM_EPOCHS = 100
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-8

# define the input image dimensions
INPUT_IMAGE_WIDTH = 296
INPUT_IMAGE_HEIGHT = 296

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory to save results
BASE_OUTPUT = "/home/jovyan/UMF/out/"

# define the path to the output serialized model, model training
# plot, and testing image paths

MODEL_PATH = os.path.join(BASE_OUTPUT, "umf")

# PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "loss_curve"])

TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths"])
