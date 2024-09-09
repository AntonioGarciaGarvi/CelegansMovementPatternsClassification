# 💡 Usage
First the dataset (in .hdf5 format) must be downloaded following the [instructions](https://github.com/AntonioGarciaGarvi/CelegansMovementPatternsClassification/tree/main/dataset)). The videos are then converted to .avi format [[1]](https://github.com/Tierpsy/tierpsy-tracker/blob/development/tierpsy/analysis/vid_subsample/createSampleVideo.py) and saved with the following nomenclature:\
**VideoName_fps_Xnf_Y**\
Where **X** is the number of fps at which the video was captured and **Y** is the total number of frames in the video.

The videos are splited into 4 folders :
-  base_path/N2_train/'
-  base_path/N2_val/'
-  base_path/unc_train/'
-  base_path/unc_val/'

At the beginning of the **train.py** script, the paths and hyperparameters (of the model, of the training and of the frequency and spatial resolution of the sampled videos) are set.\
Once everything is set, the **train.py** script is executed.

