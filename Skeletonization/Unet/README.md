# 💡 Usage
First of all, the dataset must be created with a folder structure in which the original images are on one side and the skeleton masks on the other side.
- **Dataset**
  - **/images/**
    - *Description:* Directory containing image files.
      - Example: `image001.jpg`, `image002.jpg`, ...

  - **/masks/**
    - *Description:* Directory containing mask files.
      - Example: `mask001.jpg`, `mask002.jpg`, ..


Afterwards, the **config.py** file must be edited to adjust the paths and some hyperparameters.
Finally, run the script **train.py**


# 📝 References 
* [https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/).
* [https://github.com/namdvt/skeletonization](https://github.com/namdvt/skeletonization).
* [High-throughput segmentation of unmyelinated axons by deep learning](https://doi.org/10.1038/s41598-022-04854-3).
* [Skeletonizing Caenorhabditis elegans Based on U-Net Architectures Trained with a Multi-worm Low-Resolution Synthetic Dataset](https://doi.org/10.1007/s11263-023-01818-6).

