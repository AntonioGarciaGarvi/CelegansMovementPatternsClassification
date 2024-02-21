import numpy as np
import statistics as stat
import math
import time
import datetime
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import modelsNN
import datasetOpenWD as datasets
from torchvision import transforms
from glob import glob
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import random
import utils
from utils import EarlyStopping
from tqdm import tqdm
import os

## In order to ensure reproducible experiments, we must set random seeds.
seed = 42
np.random.seed(seed)
random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## Training hyperparameters
batch_size = 2
learning_rate = 1e-5
number_epochs = 20

skipped_frames = 1  # frames discarded between sampled images
duration = 30  # durantion in seconds of subvideos
captured_fps = 30  # fps of the original videos
sample_freq = captured_fps / (skipped_frames + 1)
seq_length = int(sample_freq * duration)  # total number of frames
frames_per_int = int(captured_fps * duration)  # frames at each interval of the duration of the subvideos

n_classes = 2  # N2 and mutant
dim_resize = (224, 224)  # Neural network input image size

## Transformer hyperparameters
dim = 64
depth = 2
heads = 8
mlp_dim = 128
dim_head = 64
dropout = 0
emb_dropout = 0


## Load Model
model_name = "CNNTrHsOWD" + str(dim_resize[0]) + "skf_" + str(skipped_frames) + "d_" + str(duration)
results_folder = '/results/'
saving_folder = results_folder + "W" + str(duration) + "s/seed" + str(seed) + "/" + model_name + "/"
print(saving_folder)
if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)

nw = modelsNN.CNN_Transformer(seq_length=seq_length, n_classes=n_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dim_head=dim_head, dropout=0., emb_dropout=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nw.to(device)

## Loss function and optimizer
loss_func = torch.nn.CrossEntropyLoss()
wd = 0.1
optimizer = optim.Adam(nw.parameters(), lr=learning_rate, weight_decay=wd)
early_stopping = EarlyStopping(patience=6, min_delta=0)

# learning_rate scheduler, each patience epochs without improvement varies lr by factor
using_scheduler = True
if using_scheduler:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)
    print('using lr scheduler')


# Define base paths and folder names
# The videos are divided into 4 folders :
# base_path/N2_train'
# base_path/N2_val'
# base_path/unc_train'
# base_path/unc_val'

base_path = '/home/jovyan/datasetN2vsUnc/'
n2_folder = 'N2'
unc_folder = 'unc'

# Prepare N2 dataset paths
videos_pathsN2T, videos_pathsN2V = utils.prepare_dataset_paths(base_path, f'{n2_folder}_train', f'{n2_folder}_val')

# Prepare UNC dataset paths
videos_pathsUNCT, videos_pathsUNCV = utils.prepare_dataset_paths(base_path, f'{unc_folder}_train', f'{unc_folder}_val')

# Merge all paths in one list
videos_paths = videos_pathsN2T + videos_pathsN2V + videos_pathsUNCT + videos_pathsUNCV

dataset_list, idx_list = utils.divide_videos_into_subvideos(videos_paths, skipped_frames, seq_length, frames_per_int)
# get train and val idx
all_train_idx, all_val_idx = utils.process_data_indices(idx_list)
max_samp  = utils.get_max_samp_per_sample(idx_list)

idx_train_list, idx_val_list = utils.random_sample_max_subvideos(max_samp, idx_list, all_train_idx, all_val_idx)
dataset_size = int(len(idx_train_list) + len(idx_val_list))
print('dataset size: ' + str(dataset_size))

train_sampler = torch.utils.data.SubsetRandomSampler(idx_train_list)
val_sampler = torch.utils.data.SubsetRandomSampler(idx_val_list)

## Load data
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])


data_train = datasets.HealthspanDataset(data_list=dataset_list, dataset_size=dataset_size, seq_length=seq_length,
                                        transform=data_transform, augmentation=True, img_size=dim_resize)
dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, sampler=train_sampler, num_workers=4)

data_val = datasets.HealthspanDataset(data_list=dataset_list, dataset_size=dataset_size, seq_length=seq_length,
                                      transform=data_transform, augmentation=False, img_size=dim_resize)
dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, sampler=val_sampler, shuffle=False,
                                             num_workers=4)



## Training and validation loop
start_train = time.time()
tmstart = time.strftime("%Y-%d-%b-%Hh-%Mm", time.localtime(time.time()))

print("__________Training started at: " + str(datetime.datetime.now()) + " ________________")

loss_train = []
loss_val = []

best_loss = 100000.0
best_nw = nw.cpu()
nw = nw.cuda()

for epoch in range(number_epochs):
    print("___________New Epoch______________")
    start_epoch = time.time()

    loss_list = []
    loss_list_val = []

    print("Updating weights...")
    print("Learning rate: " + str(learning_rate))
    nw.train()

    for batch in tqdm(dataloader_train):
        optimizer.zero_grad()
        imgs_batch = batch[0].to(device)
        labels = batch[1].to(device)
        pre = nw(imgs_batch)
        train_loss = loss_func(pre, labels)
        train_loss.backward()  # Calculate all gradients by backpropagation
        optimizer.step()  # Optimize
        loss_list.append(train_loss.item())

    print("Calculating cross entropy loss for eval set....")
    nw.eval()

    with torch.no_grad():
        # val data
        y_true = []
        y_pred = []
        for batch in tqdm(dataloader_val):
            imgs_batch = batch[0].to(device)
            labels = batch[1].to(device)
            y_true.extend(labels.cpu().numpy())
            pre = nw(imgs_batch)
            _, predicted = torch.max(pre, 1)
            y_pred.extend(predicted.cpu().numpy())
            loss = loss_func(pre, labels)
            loss_list_val.append(loss.item())

    cf_matrix_val = confusion_matrix(y_true, y_pred)
    class_names = ('N2', 'others')

    # Create Confusion Matrix and calculate metrics
    print('----------------------------------------')
    print("Confusion Matrix  for validation data")
    print('----------------------------------------')
    dataframe = pd.DataFrame(cf_matrix_val, index=class_names, columns=class_names)
    print(dataframe)
    print('----------------------------------------')
    print("Metrics  for validation data")
    print(classification_report(y_true, y_pred, target_names=class_names))

    loss_ep_train = stat.mean(loss_list)
    loss_ep_val = stat.mean(loss_list_val)

    loss_train.append(loss_ep_train)
    loss_val.append(loss_ep_val)

    if using_scheduler:
        scheduler.step(loss_ep_val)

    if loss_ep_val < best_loss:
        print('This is the best cross entropy loss obtained: ' + str(loss_ep_val))
        best_nw = nw.cpu()
        # Save model
        torch.save(best_nw.cpu().state_dict(),
                   saving_folder + model_name + "_" + tmstart + ".pth")
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_nw.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_ep_val
            }, saving_folder + model_name + "_" + tmstart + "v2.pt")
        except:
            print('error saving')

        nw = nw.cuda()
        best_loss = loss_ep_val

        plt.figure(figsize=(8, 8))

        # Create heatmap
        sns.heatmap(dataframe, annot=True, cbar=None, cmap="YlGnBu", fmt="d")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.savefig(saving_folder + 'confmat' + model_name + "_" + tmstart + ".jpg", dpi=96)
        plt.close()

    print("Epoch: " + str(epoch) + "\nLoss in this epoch on train data: " + str(
        loss_ep_train) + "\nLoss in this epoch on test data: " + str(loss_ep_val) + "\nbest test loss obtained: " + str(
        best_loss))
    print("Epoch took " + str(int((time.time() - start_epoch) / 60)) + " mins to run.\n")
    early_stopping(loss_ep_val)
    if early_stopping.early_stop:
        break

train_time = int(math.floor((time.time() - start_train) / 60))
print("\n\n--------Training finished-----------\nExecution time in minutes: " + str(train_time))


ts = time.strftime("%Y-%d-%b-%Hh-%Mm", time.localtime(time.time()))

# Generate training curve
txt = "Batch size=" + str(batch_size) + "; Sequence length=" + str(seq_length) + "; Skipped_frames=" + str(
    skipped_frames) + "; Initial learning rate=" + str(learning_rate) + " ; Best cross entropy on val set: " + str(
    best_loss) + "; Total training time=" + str(train_time) + "mins"
txt_params = "Resnet18 pretrained" + "; weight decay: " + str(wd) + "; TRdim: " + str(dim) + ";TRdepth: " + str(
    depth) + ";TRheads: " + str(heads) + ";TRdim_head: " + str(dim_head) + ";TRmlp_dim: " + str(
    mlp_dim) + "; Dropout:" + str(dropout)

plt.figure(figsize=(1855 / 96, 986 / 96), dpi=96)
plt.plot(loss_train, "ro-", label="Train data")
plt.plot(loss_val, "bx-", label="Val data")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Cross Entropy loss for the train and val data over all training epochs, trained Model: " + model_name)
plt.figtext(0.5, 0.03, txt, wrap=True, horizontalalignment='center', fontsize=10)
plt.figtext(0.5, 0.01, txt_params, wrap=True, horizontalalignment='center', fontsize=8)
plt.grid(True, axis="y")
plt.ylim((0, int(max(np.amax(loss_train), np.amax(loss_val))) + 1))
plt.legend()
plt.savefig(saving_folder + model_name + "_" + ts + ".jpg", dpi=96)
plt.close()