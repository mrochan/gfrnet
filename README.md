# G-FRNet: Gated Feedback Refinement Network for Dense Image Labeling

This repository contains code for the paper  

**[Gated Feedback Refinement Network for Dense Image Labeling](http://www.cs.umanitoba.ca/~ywang/papers/cvpr17.pdf)**,
<br>
Presented at [CVPR 2017](http://cvpr2017.thecvf.com/)

The paper addresses the problem of **dense image labeling**, where the goal is to label each pixel in an image. The proposed model is an encoder-decoder-based deep convolutional neural network that is trained in an end-to-end fashion. The network architecture is inspired by the *[SegNet](https://github.com/alexgkendall/caffe-segnet)* architecture.

If you find this code useful in your research, please cite:

    @InProceedings{Islam_2017_CVPR,
	author = {Amirul Islam, Md and Rochan, Mrigank and Bruce, Neil D. B. and Wang, Yang},
	title = {Gated Feedback Refinement Network for Dense Image Labeling},
	booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {July},
	year = {2017}
    }

## Setting up Caffe-gfrnet and the Dataset
G-FRNet requires a modified version of caffe to run. Please download and compile caffe-gfrnet to use these models.

We used the CamVid dataset which contains 367 training and 233 testing images of road scenes. We used an 11 class version with an image size of 360 by 480. Download this data in the format required for G-FRNet from [this](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid) GitHub repository.

You now need to modify CamVid/train.txt and CamVid/test.txt so that G-FRNet knows where to find the data. G-FRNet requires a text file of white-space separated paths to images (.jpg or .png) and corresponding label images (.png) alternatively, e.g. 

	/path/to/image1.png /another/path/to/label1.png 
	/path/to/image2.png /path/label2.png ...

Please open up these two files in a text editor and use the find & replace tool to change to the absolute path of your data.
 
## Training G-FRNet

The next step is to set up a model for training. First, open the model file Models/CamVid/train_camvid_gate.prototxt and inference model file Models/CamVid/test_camvid_gate.prototxt. You will need to modify the data input source line in all the model's data layers. Replace this with the absolute directory to your data file. Secondly, please open the solver file 
Models/CamVid/solver_camvid_gate.prototxt and change two lines; the net and snapshot_prefix directories should match the directory to your data.

We are now ready to train G-FRNet! Open up a terminal and issue these commands:

    sh run_camvid_train.sh
    
## Testing G-FRNet

First, open up the scripts Scripts/compute_bn_statistics_camvid.py and Scripts/test_segmentation_camvid.py and change to the directory to your G-FRNet Caffe installation. 

We are now ready to test G-FRNet! Open up a terminal and issue this command:

    sh run_camvid_test.sh
We use the Batch Normalization layers in G-FRNet that shift the input feature maps according to their mean and variance statistics for each mini batch. The test script firstly run the script 	Scripts/compute_bn_statistics_camvid.py and saves the final test weights in the output directory as /Models/CamVid/Inference/test_weights.caffemodel. Finally the test script run test_segmentation_camvid.py which save the final predictions for each test image.

## Initialization & Pretrained Models
The initialization and pretrained models are provided [here](https://drive.google.com/open?id=0B4FSw1mplCQTblNkQmlzTU9ROTQ). Please download and put it in the relative path of your directory.  



    
