# Skeletonization 

In this work we decided to compare the model used in Wormpose [1] with the UNet UMF [2], which in our previous work [3] gave us good results for segmenting skeletons in low-resolution *C. elegans* images. 

First, we used the proposed simulator of Wormpose [1], as it allows us to synthetically generate image datasets very similar to those of the Open Worm Movement Database [4]. 

Following their proposed methodology, we generated 500k synthetic images, which were used to train both models. 

## References 
[1] [WormPose: Image synthesis and convolutional networks for pose estimation in C. elegans](https://doi.org/10.1371/journal.pcbi.1008914)

[2] [High-throughput segmentation of unmyelinated axons by deep learning](https://doi.org/10.1038/s41598-022-04854-3).

[3] [Skeletonizing Caenorhabditis elegans Based on U-Net Architectures Trained with a Multi-worm Low-Resolution Synthetic Dataset](https://doi.org/10.1007/s11263-023-01818-6).

[4] [An open source platform for analyzing and sharing worm behavior data](https://doi.org/10.1038%2Fs41592-018-0112-1)

