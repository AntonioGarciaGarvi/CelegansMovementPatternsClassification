# USAGE
# python train.py
# import the necessary packages
import dataset
from UMFmodel import UMF
import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
from glob import glob
from loss import Loss 
import utils


print("[INFO] loading image paths...")

# load the image and mask filepaths in a sorted manner
imagePaths =glob(config.IMAGE_DATASET_PATH + '/*.jpg')
maskPaths =glob(config.MASK_DATASET_PATH + '/*.jpg')
imagePaths.sort(key=utils.natural_keys)
maskPaths.sort(key=utils.natural_keys)

# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(imagePaths, maskPaths,
    test_size=config.TEST_SPLIT, random_state=42)

# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]

# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS + "_" + time.strftime("%Y-%d-%b-%Hh-%Mm", time.localtime(time.time())) + ".txt", "w")
f.write("\n".join(testImages))
f.close()

# define transformations
transforms_train = transforms.Compose([utils.Flip(),
            utils.Rotate(),utils.Normalize()
   ])

transforms_val = utils.Normalize()


# create the train and test datasets
trainDS = dataset.SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
    transforms=transforms_train)
testDS = dataset.SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms_val)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")

# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
    batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
    num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
    batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
    num_workers=os.cpu_count())

# initialize our UNet model
unet = UMF(3, config.NUM_CLASSES).to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = Loss()
opt = Adam(unet.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY)

using_scheduler = True
if using_scheduler:
    scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=utils.lambda_)
    print('using lr scheduler')

using_early_stopping = False
if using_early_stopping:
    early_stopping = utils.EarlyStopping(patience=6, min_delta=0)
    print('using early stopping')    
    
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

best_loss = 1000000.0

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
starttim = time.strftime("%Y-%d-%b-%Hh-%Mm", time.localtime(time.time()))
for e in tqdm(range(config.NUM_EPOCHS)):
    # set the model in training mode
    unet.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalTrainDiceLoss = 0
    totalTrainBceLoss = 0
    
    totalTestLoss = 0
    # loop over the training set
    for (i, (x, y)) in enumerate(trainLoader):
        # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
        # perform a forward pass and calculate the training loss
        pred = unet(x)

        soft_dice_loss, bce_loss = lossFunc(pred, y)

        loss = soft_dice_loss + bce_loss
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far
        totalTrainLoss += loss
        totalTrainDiceLoss += soft_dice_loss
        totalTrainBceLoss += bce_loss
        
    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        unet.eval()
        # loop over the validation set
        for (x, y) in testLoader:
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # make the predictions and calculate the validation loss
            pred = unet(x)
            soft_dice_loss, bce_loss = lossFunc(pred, y)
            lossTest = soft_dice_loss + bce_loss
            totalTestLoss += lossTest

    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgTestLoss = totalTestLoss / testSteps
    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(
        avgTrainLoss, avgTestLoss))
    
    
    avgTrainDiceLoss = totalTrainDiceLoss / trainSteps
    avgTrainBceLoss = totalTrainBceLoss / trainSteps
    print("TrainDiceLoss: {:.6f}, TrainBceLoss: {:.4f}".format(
    avgTrainDiceLoss, avgTrainBceLoss))

    if using_scheduler:
        scheduler.step(avgTestLoss)

    if avgTestLoss < best_loss:
        print('This is the best loss obtained: ' + str(avgTestLoss.item()))

        # serialize the model to disk
        torch.save(unet.cpu().state_dict(),
                   config.MODEL_PATH + "_" + starttim + ".pth")
        unet = unet.cuda()
        best_loss = avgTestLoss

    
    if using_early_stopping:
        early_stopping(avgTestLoss.cpu().detach().numpy())
        if early_stopping.early_stop:
            break
        
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch ")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH + "UMF_" + starttim  + ".png")