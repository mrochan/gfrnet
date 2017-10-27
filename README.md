# G-FRNet: Gated Feedback Refinement Network for Dense Image Labeling

This repository contains code for the paper  

**[Gated Feedback Refinement Network for Dense Image Labeling](http://www.cs.umanitoba.ca/~ywang/papers/cvpr17.pdf)**,
<br>
Presented at [CVPR 2017](http://cvpr2017.thecvf.com/)

The paper addresses the problem of **dense image labeling**, where the goal is to label each pixel in an image. The proposed model is an encoder-decoder-based deep convolutional neural network that is trained in an end-to-end fashion. <!-- The network architecture is inspired by the *[SegNet](https://github.com/alexgkendall/caffe-segnet)* architecture. -->

If you find this code useful in your research, please cite:

    @InProceedings{Islam_2017_CVPR,
	author = {Amirul Islam, Md and Rochan, Mrigank and Bruce, Neil D. B. and Wang, Yang},
	title = {Gated Feedback Refinement Network for Dense Image Labeling},
	booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {July},
	year = {2017}
    }

## Setup
Download and compile caffe-gfrnet which is a modified version of [Caffe](https://github.com/BVLC/caffe).

One of the datasets we experimented with was CamVid dataset (11 class version), which contains 367 training and 233 testing images of road scenes. Each image is of  size 360 by 480. Download the dataset from [this](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid) GitHub repository.

Modify data/camvid/train.txt and data/camvid/test.txt so that G-FRNet can locate the data. The text files contains white-space separated paths to images (.jpg or .png) and their corresponding label images (.png), e.g., 

	/path/to/image1.png /another/path/to/label1.png 
	/path/to/image2.png /path/label2.png ...

Update these two files with the paths where you stored the dataset.
 
## Training G-FRNet

Modify the source under data layer of model file models/camvid/train_camvid_gate.prototxt and the inference model file models/camvid/test_camvid_gate.prototxt. Also, update the net and snapshot_prefix in the solver file 
models/camvid/solver_camvid_gate.prototxt. Note that we initialize our model with pretrained VGG-16 caffemodel which can be downloaded from [here](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). 

Run the following command to train:

    sh run_camvid_train.sh
    
## Testing G-FRNet

Run scripts/compute_bn_statistics_camvid.py and followed by scripts/test_segmentation_camvid.py. You would need to update the paths in these two files in order to successfully run them.

Then, run the following command to test:

    sh run_camvid_test.sh


## Pretrained Models
You can download the pretrained model from [here](https://drive.google.com/open?id=0B4FSw1mplCQTblNkQmlzTU9ROTQ). 

    
